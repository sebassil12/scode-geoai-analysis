"""Model construction, hyperparameter tuning and artifact serialisation.

This module owns the candidate registry and the search plumbing; ``benchmark``
builds on it to race the candidates against each other. Keeping both in one
place means a model is tuned exactly the same way whether it is being compared
or being trained for production.

CLI::

    python -m ml_project.modeling.train --data data/raw/ --model xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ml_project.config import (
    CHAMPION_PATH,
    CLASS_LABELS,
    CV_FOLDS,
    DATA_DIR,
    LABEL_COLUMN,
    MODEL_N_JOBS,
    RANDOM_STATE,
    S1_BANDS_ORDER,
    SEARCH_ITERATIONS,
    SEARCH_N_JOBS,
    TEST_SIZE,
)
from ml_project.features.sar_engineering import SarFeatureEngineer, build_selector
from ml_project.modeling.evaluate import PRIMARY_METRIC, build_scorers, evaluate
from ml_project.modeling.wrappers import ZeroIndexedClassifier

log = logging.getLogger(__name__)

BALANCE_STRATEGIES = ("class_weight", "smotetomek")

_SHARED_PARAMS: dict[str, list[Any]] = {"selection__k": [10, 20, "all"]}


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
def _random_forest(balance: str) -> tuple[Any, dict[str, list[Any]]]:
    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS,
        class_weight="balanced_subsample" if balance == "class_weight" else None,
    )
    params = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [15, 25, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }
    return estimator, params


def _xgboost(balance: str) -> tuple[Any, dict[str, list[Any]]]:
    from xgboost import XGBClassifier

    if balance == "class_weight":
        # XGBoost has no multiclass class_weight; scale_pos_weight is binary only.
        log.warning("xgboost: no native multiclass class weighting — use --balance smotetomek")

    estimator = ZeroIndexedClassifier(
        XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=MODEL_N_JOBS,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
    )
    params = {
        "model__estimator__n_estimators": [300, 500, 800],
        "model__estimator__max_depth": [4, 6, 10],
        "model__estimator__learning_rate": [0.03, 0.1, 0.2],
        "model__estimator__subsample": [0.7, 0.9, 1.0],
        "model__estimator__colsample_bytree": [0.7, 0.9, 1.0],
    }
    return estimator, params


def _lightgbm(balance: str) -> tuple[Any, dict[str, list[Any]]]:
    from lightgbm import LGBMClassifier

    estimator = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS,
        verbose=-1,
        class_weight="balanced" if balance == "class_weight" else None,
    )
    params = {
        "model__n_estimators": [300, 500, 800],
        "model__num_leaves": [31, 63, 127],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__min_child_samples": [5, 20, 50],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }
    return estimator, params


MODEL_BUILDERS: dict[str, Callable[[str], tuple[Any, dict[str, list[Any]]]]] = {
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def available_models() -> list[str]:
    """Registry entries whose backing library is actually importable."""
    available = []
    for name, builder in MODEL_BUILDERS.items():
        try:
            builder("class_weight")
        except ImportError:
            log.warning("%s is not installed — skipping (pip install %s)", name, name)
        else:
            available.append(name)
    return available


def build_candidate(
    name: str,
    balance: str = "class_weight",
) -> tuple[Pipeline, dict[str, list[Any]]]:
    """Assemble the full pipeline and search space for one named model.

    ``balance='class_weight'`` uses each library's native weighting and adds no
    dependency. ``balance='smotetomek'`` resamples inside every CV fold instead,
    which works uniformly across all three models but requires imbalanced-learn.
    """
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODEL_BUILDERS)}")
    if balance not in BALANCE_STRATEGIES:
        raise ValueError(f"Unknown balance strategy {balance!r}. Known: {BALANCE_STRATEGIES}")

    estimator, params = MODEL_BUILDERS[name](balance)
    steps: list[tuple[str, Any]] = [
        ("features", SarFeatureEngineer()),
        ("selection", build_selector()),
    ]

    if balance == "smotetomek":
        from imblearn.combine import SMOTETomek
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("balance", SMOTETomek(random_state=RANDOM_STATE)))
        steps.append(("model", estimator))
        return ImbPipeline(steps), params | _SHARED_PARAMS

    steps.append(("model", estimator))
    return Pipeline(steps), params | _SHARED_PARAMS


# ---------------------------------------------------------------------------
# Data and search
# ---------------------------------------------------------------------------
def load_dataset(data_path: Path | str) -> pd.DataFrame:
    """Load one CSV, or concatenate every CSV in a directory."""
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.csv")) if data_path.is_dir() else [data_path]
    if not files:
        raise FileNotFoundError(f"No CSV training data found in {data_path}")

    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = set(S1_BANDS_ORDER + [LABEL_COLUMN]) - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    frame = frame[S1_BANDS_ORDER + [LABEL_COLUMN]].dropna()
    log.info(
        "Loaded %d samples from %d file(s): %s",
        len(frame),
        len(files),
        frame[LABEL_COLUMN].value_counts().to_dict(),
    )
    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the label column."""
    X = frame[S1_BANDS_ORDER]
    y = frame[LABEL_COLUMN].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def run_search(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    balance: str = "class_weight",
    n_iter: int = SEARCH_ITERATIONS,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune one candidate and return its fitted best estimator plus CV scores."""
    pipeline, param_distributions = build_candidate(name, balance)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=build_scorers(),
        refit=PRIMARY_METRIC,  # PR-AUC is reported alongside; macro F1 picks the winner
        n_jobs=SEARCH_N_JOBS,
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    log.info("Tuning %s (%d iterations, %d-fold stratified)", name, n_iter, CV_FOLDS)
    search.fit(X_train, y_train)

    best = search.best_index_
    cv_scores = {
        "cv_f1_macro": float(search.cv_results_[f"mean_test_{PRIMARY_METRIC}"][best]),
        "cv_pr_auc": float(search.cv_results_["mean_test_pr_auc"][best]),
        "best_params": search.best_params_,
    }
    log.info(
        "%s: CV macro-F1=%.4f | CV PR-AUC=%.4f",
        name,
        cv_scores["cv_f1_macro"],
        cv_scores["cv_pr_auc"],
    )
    return search.best_estimator_, cv_scores


def run_ensemble(
    fitted: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict[str, Any]]:
    """Soft-vote the already-tuned candidates into one calibrated classifier.

    The base pipelines each carry their own tuned hyper-parameters and balance
    strategy, so cloning them preserves the tuning; voting only averages the
    class probabilities. Averaging beats any single tree family here because RF
    leads on macro F1 while the boosters lead on PR-AUC — the vote keeps both.
    CV uses the same splitter and scorers as ``run_search`` so its numbers sit
    on the same scale as every other candidate's.
    """
    estimators = [(name, clone(pipe)) for name, pipe in fitted.items()]
    ensemble = VotingClassifier(estimators, voting="soft")

    scores = cross_validate(
        ensemble,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=build_scorers(),
        n_jobs=SEARCH_N_JOBS,
        error_score="raise",
    )
    cv_scores = {
        "cv_f1_macro": float(scores[f"test_{PRIMARY_METRIC}"].mean()),
        "cv_pr_auc": float(scores["test_pr_auc"].mean()),
        "best_params": {"components": list(fitted)},
    }

    ensemble.fit(X_train, y_train)
    log.info(
        "ensemble: CV macro-F1=%.4f | CV PR-AUC=%.4f (%s)",
        cv_scores["cv_f1_macro"],
        cv_scores["cv_pr_auc"],
        ", ".join(fitted),
    )
    return ensemble, cv_scores


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _feature_pipeline(model: Any) -> Pipeline:
    """The Pipeline that owns the ``features``/``selection`` steps.

    For a single tuned model that is the model itself; for the soft-voting
    ensemble every base shares the same feature schema, so the first one is a
    faithful representative for the sidecar metadata.
    """
    if isinstance(model, Pipeline) and "features" in model.named_steps:
        return model
    if isinstance(model, VotingClassifier):
        return model.estimators_[0]
    raise TypeError(f"Cannot locate feature pipeline in {type(model).__name__}")


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    model_path: Path | str = CHAMPION_PATH,
) -> Path:
    """Write the ``.joblib`` model and its ``<stem>_metadata.json`` sidecar."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    feature_pipeline = _feature_pipeline(pipeline)
    engineer: SarFeatureEngineer = feature_pipeline.named_steps["features"]
    support = feature_pipeline.named_steps["selection"].get_support()

    payload = {
        **metadata,
        "input_bands": S1_BANDS_ORDER,
        "engineered_features": engineer.feature_names_,
        "selected_features": np.asarray(engineer.feature_names_)[support].tolist(),
        "scaler": {
            "type": "RobustScaler",
            "center": engineer.scaler_.center_.tolist(),
            "scale": engineer.scaler_.scale_.tolist(),
        },
        "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Saved %s and %s", model_path, metadata_path)
    return model_path


def train_model(
    frame: pd.DataFrame,
    name: str = "random_forest",
    balance: str = "class_weight",
    n_iter: int = SEARCH_ITERATIONS,
    test_size: float = TEST_SIZE,
    output_dir: Path | str | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune and evaluate one model; return the fitted pipeline and its metadata."""
    X_train, X_test, y_train, y_test = split_dataset(frame, test_size)
    pipeline, cv_scores = run_search(name, X_train, y_train, balance, n_iter)
    holdout = evaluate(pipeline, X_test, y_test, output_dir, prefix=f"{name} ")

    metadata = {
        "model": name,
        "balance": balance,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        **cv_scores,
        "holdout": holdout,
    }
    return pipeline, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production SAR vegetation classifier.")
    parser.add_argument("--data", default=str(DATA_DIR), help="CSV file or directory of CSVs")
    parser.add_argument("--model", default="random_forest", choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--output", default=str(CHAMPION_PATH), help="Path for the .joblib model")
    parser.add_argument("--balance", default="class_weight", choices=BALANCE_STRATEGIES)
    parser.add_argument("--n-iter", type=int, default=SEARCH_ITERATIONS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    model_path = Path(args.output)
    pipeline, metadata = train_model(
        load_dataset(args.data),
        args.model,
        args.balance,
        args.n_iter,
        args.test_size,
        output_dir=model_path.parent,
    )
    save_artifacts(pipeline, metadata, model_path)


if __name__ == "__main__":
    main()
