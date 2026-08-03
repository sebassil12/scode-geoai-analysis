"""Estimator adapters that end up inside a pickled pipeline.

This module deliberately has no ``__main__`` block and is never executed with
``python -m``. Anything pickled into a model artifact must live somewhere that
is only ever *imported*: a class defined in a module run as ``__main__`` is
recorded by pickle as ``__main__.ClassName``, and every other entry point then
fails to load the model with ``AttributeError``. Do not fold these back into
``train.py`` — it has a CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ZeroIndexedClassifier(ClassifierMixin, BaseEstimator):
    """Adapt a classifier that demands labels ``0..n-1`` to our 1-4 schema.

    XGBoost rejects any target whose classes are not zero-indexed and
    contiguous. Remapping the labels globally would leak that constraint into
    the config, the metadata and the inference code; confining it to a wrapper
    keeps class ``1`` meaning "Dense Vegetation" everywhere else.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: Any, **fit_params: Any) -> ZeroIndexedClassifier:
        self.classes_ = np.unique(y)
        self.estimator.fit(X, np.searchsorted(self.classes_, y), **fit_params)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
