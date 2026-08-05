"""Champion-vs-challenger benchmarking: Random Forest against XGBoost and LightGBM.

Every candidate is tuned with the same search budget, scored with the same
metrics, and — critically — judged on one train/test split made *once* before
any model is built. Re-splitting per model would let a lucky partition, rather
than a better algorithm, pick the champion.

CLI::

    python -m ml_project.modeling.benchmark --data data/raw/ --output models/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_project.config import (
    BENCHMARK_REPORT,
    CHAMPION_PATH,
    DATA_DIR,
    MODEL_DIR,
    SEARCH_ITERATIONS,
)
from ml_project.modeling.evaluate import PRIMARY_METRIC, evaluate
from ml_project.modeling.train import (
    BALANCE_STRATEGIES,
    MODEL_BUILDERS,
    available_models,
    load_dataset,
    run_ensemble,
    run_search,
    save_artifacts,
    split_dataset,
)

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """One candidate's scores. Ordered so the champion sorts first."""

    model: str
    holdout_f1_macro: float
    holdout_pr_auc: float
    cv_f1_macro: float
    cv_pr_auc: float
    fit_seconds: float
    best_params: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_key(self) -> tuple[float, float]:
        """Macro F1 decides; PR-AUC breaks ties.

        F1 leads because a false work-order dispatch and a missed encroachment
        both cost real money, so precision and recall matter equally.
        """
        return (self.holdout_f1_macro, self.holdout_pr_auc)


def run_benchmark(
    frame: pd.DataFrame,
    models: list[str] | None = None,
    balance: str = "class_weight",
    n_iter: int = SEARCH_ITERATIONS,
    output_dir: Path | str = MODEL_DIR,
) -> tuple[list[BenchmarkResult], str, Pipeline]:
    """Tune and score every candidate; return results, champion name and pipeline."""
    models = models or available_models()
    if not models:
        raise RuntimeError("No model libraries available — pip install xgboost lightgbm")

    output_dir = Path(output_dir)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    log.info("Benchmarking %s on %d train / %d test samples", models, len(X_train), len(X_test))

    results: list[BenchmarkResult] = []
    fitted: dict[str, Pipeline] = {}

    for name in models:
        started = time.perf_counter()
        try:
            pipeline, cv_scores = run_search(name, X_train, y_train, balance, n_iter)
        except ImportError:
            log.warning("%s is not installed — skipping", name)
            continue

        holdout = evaluate(pipeline, X_test, y_test, output_dir, prefix=f"{name} ")
        fitted[name] = pipeline
        results.append(
            BenchmarkResult(
                model=name,
                holdout_f1_macro=holdout[PRIMARY_METRIC],
                holdout_pr_auc=holdout["pr_auc"],
                cv_f1_macro=cv_scores["cv_f1_macro"],
                cv_pr_auc=cv_scores["cv_pr_auc"],
                fit_seconds=round(time.perf_counter() - started, 2),
                best_params=cv_scores["best_params"],
            )
        )

    if not results:
        raise RuntimeError("Every candidate failed to train — see the warnings above")

    # Race a soft-voting ensemble of the tuned candidates as one more contender.
    # A vote of one is just that model, so it only enters with two or more bases.
    if len(fitted) >= 2:
        started = time.perf_counter()
        ensemble, cv_scores = run_ensemble(fitted, X_train, y_train)
        holdout = evaluate(ensemble, X_test, y_test, output_dir, prefix="ensemble ")
        fitted["ensemble"] = ensemble
        results.append(
            BenchmarkResult(
                model="ensemble",
                holdout_f1_macro=holdout[PRIMARY_METRIC],
                holdout_pr_auc=holdout["pr_auc"],
                cv_f1_macro=cv_scores["cv_f1_macro"],
                cv_pr_auc=cv_scores["cv_pr_auc"],
                fit_seconds=round(time.perf_counter() - started, 2),
                best_params=cv_scores["best_params"],
            )
        )

    results.sort(key=lambda r: r.rank_key, reverse=True)
    champion = results[0].model
    log.info("Champion: %s (holdout macro-F1=%.4f)", champion, results[0].holdout_f1_macro)

    return results, champion, fitted[champion]


def comparison_table(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Human-readable scoreboard, best model first."""
    return pd.DataFrame(
        [
            {
                "model": r.model,
                "holdout_f1_macro": round(r.holdout_f1_macro, 4),
                "holdout_pr_auc": round(r.holdout_pr_auc, 4),
                "cv_f1_macro": round(r.cv_f1_macro, 4),
                "cv_pr_auc": round(r.cv_pr_auc, 4),
                "fit_seconds": r.fit_seconds,
            }
            for r in results
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark candidate models and save the champion."
    )
    parser.add_argument("--data", default=str(DATA_DIR), help="CSV file or directory of CSVs")
    parser.add_argument("--output", default=str(MODEL_DIR), help="Directory for model artifacts")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_BUILDERS),
        help="Subset to benchmark (default: every installed model)",
    )
    parser.add_argument("--balance", default="class_weight", choices=BALANCE_STRATEGIES)
    parser.add_argument("--n-iter", type=int, default=SEARCH_ITERATIONS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    output_dir = Path(args.output)
    results, champion, pipeline = run_benchmark(
        load_dataset(args.data), args.models, args.balance, args.n_iter, output_dir
    )

    table = comparison_table(results)
    log.info("Benchmark results:\n%s", table.to_string(index=False))

    report_path = output_dir / BENCHMARK_REPORT.name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "champion": champion,
                "balance": args.balance,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            default=str,
        )
    )
    log.info("Wrote %s", report_path)

    save_artifacts(
        pipeline,
        {"selected_by": "benchmark", **asdict(results[0])},
        output_dir / CHAMPION_PATH.name,
    )


if __name__ == "__main__":
    main()
