"""Shared scoring: the metric definitions every model in the pipeline is judged by.

Both ``train`` and ``benchmark`` import their scorers from here, so a model can
never be tuned against one definition of macro F1 and reported against another.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs on servers, not in a notebook

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

from ml_project.config import CLASS_IDS, CLASS_LABELS  # noqa: E402

log = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [f"{i}: {CLASS_LABELS[i]}" for i in CLASS_IDS]
PRIMARY_METRIC = "f1_macro"


def macro_pr_auc(y_true: Any, y_score: np.ndarray) -> float:
    """Macro-averaged average precision.

    ``average_precision_score`` only handles binary or multilabel targets, so
    the multiclass labels are binarised against the full class list first —
    using ``CLASS_IDS`` rather than the labels present, so a fold that happens
    to miss a class still scores on the same scale as every other fold.
    """
    binarised = label_binarize(y_true, classes=CLASS_IDS)
    return float(average_precision_score(binarised, y_score, average="macro"))


def build_scorers() -> dict[str, Any]:
    """The two metrics GridCanopy selects on: macro F1 and macro PR-AUC."""
    return {
        PRIMARY_METRIC: make_scorer(f1_score, average="macro"),
        "pr_auc": make_scorer(macro_pr_auc, response_method="predict_proba"),
    }


def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | str | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Score a fitted model on held-out data, optionally writing plots and JSON.

    Returns a JSON-serialisable report suitable for embedding in model metadata.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_binary = label_binarize(y_test, classes=CLASS_IDS)

    matrix = confusion_matrix(y_test, y_pred, labels=CLASS_IDS)
    average_precision = {
        CLASS_LABELS[class_id]: float(average_precision_score(y_binary[:, i], y_proba[:, i]))
        for i, class_id in enumerate(CLASS_IDS)
    }

    summary: dict[str, Any] = {
        PRIMARY_METRIC: float(f1_score(y_test, y_pred, average="macro")),
        "pr_auc": float(np.mean(list(average_precision.values()))),
        "average_precision_per_class": average_precision,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=CLASS_IDS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    log.info(
        "%smacro-F1=%.4f | macro PR-AUC=%.4f", prefix, summary[PRIMARY_METRIC], summary["pr_auc"]
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix.strip("_ ").replace(" ", "_") or "model"
        _plot_confusion_matrix(matrix, output_dir / f"{stem}_confusion_matrix.png")
        _plot_precision_recall(
            y_binary, y_proba, average_precision, output_dir / f"{stem}_precision_recall.png"
        )
        (output_dir / f"{stem}_metrics.json").write_text(json.dumps(summary, indent=2))
        log.info("Evaluation artifacts written to %s", output_dir)

    return summary


def _plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    # Row-normalised shading: raw counts hide recall problems in small classes.
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        normalised,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "recall"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts, shaded by row recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_precision_recall(
    y_binary: np.ndarray,
    y_proba: np.ndarray,
    average_precision: dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, class_id in enumerate(CLASS_IDS):
        precision, recall, _ = precision_recall_curve(y_binary[:, i], y_proba[:, i])
        label = CLASS_LABELS[class_id]
        ax.plot(recall, precision, label=f"{label} (AP={average_precision[label]:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class precision-recall")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
