"""SAR feature engineering, server-side (Earth Engine) and client-side (sklearn).

Server-side helpers add backscatter ratios, RVI and GLCM textures to an
``ee.Image``. Client-side, ``SarFeatureEngineer`` expands those bands into
polynomial interactions and scales them; it is the single transformer used at
both training and inference time, so the two can never drift apart.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from ml_project.config import DB_MAX, DB_MIN, GLCM_WINDOW_PX, S1_BANDS_ORDER

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps `ee` an optional dependency
    import ee

log = logging.getLogger(__name__)

_EPS = 1e-6
_GLCM_MEASURES = ("contrast", "ent")


# ---------------------------------------------------------------------------
# Server-side (Google Earth Engine)
# ---------------------------------------------------------------------------
def add_sar_indices(image: ee.Image) -> ee.Image:
    """Add the VV/VH ratio and the Radar Vegetation Index to a dB-scale image.

    Both indices are computed in linear power rather than dB: a ratio of
    logarithms is meaningless. VV and VH themselves stay in dB, which is what
    the classifier trains on.
    """
    import ee

    vv_db, vh_db = image.select("VV"), image.select("VH")
    vv = ee.Image(10.0).pow(vv_db.divide(10.0))
    vh = ee.Image(10.0).pow(vh_db.divide(10.0))

    ratio = vv.divide(vh.add(_EPS)).rename("VV_VH_ratio")
    rvi = vh.multiply(8.0).divide(vv.add(vh).add(_EPS)).rename("RVI")

    return image.addBands([ratio, rvi])


def add_glcm_textures(
    image: ee.Image,
    bands: tuple[str, ...] = ("VV", "VH"),
    window: int = GLCM_WINDOW_PX,
) -> ee.Image:
    """Add GLCM contrast and entropy for each named dB band.

    Earth Engine's ``glcmTexture`` needs integer input, so dB is rescaled to
    0-255 over ``[DB_MIN, DB_MAX]``. Keep ``window`` at 3 or 5; wider
    neighbourhoods smear the corridor/forest boundary the model must find.
    """
    import ee

    textures = []
    for band in bands:
        scaled = (
            image.select(band)
            .unitScale(DB_MIN, DB_MAX)
            .multiply(255)
            .clamp(0, 255)
            .toUint8()
            .rename(band)
        )
        glcm = scaled.glcmTexture(size=window)
        textures.append(
            ee.Image.cat([glcm.select(f"{band}_{measure}") for measure in _GLCM_MEASURES])
        )

    return image.addBands(textures)


# ---------------------------------------------------------------------------
# Client-side (scikit-learn)
# ---------------------------------------------------------------------------
class SarFeatureEngineer(TransformerMixin, BaseEstimator):
    """Polynomial interactions over the SAR bands, then a RobustScaler.

    RobustScaler rather than StandardScaler because speckle leaves heavy tails
    in the backscatter distribution that drag a mean/std fit around.

    Missing input columns are filled with zero and column order is restored on
    transform, so a raster whose bands arrive in a different order cannot
    silently shift the feature vector.
    """

    def __init__(self, degree: int = 2, interaction_only: bool = True) -> None:
        self.degree = degree
        self.interaction_only = interaction_only

    def fit(self, X: pd.DataFrame, y: Any = None) -> SarFeatureEngineer:
        X = self._align(X)
        self.input_names_: list[str] = X.columns.tolist()
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False,
        )
        expanded = self.poly_.fit_transform(X)
        self.feature_names_: list[str] = list(self.poly_.get_feature_names_out(self.input_names_))
        self.scaler_ = RobustScaler().fit(expanded)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._align(X, expected=self.input_names_)
        scaled = self.scaler_.transform(self.poly_.transform(X))
        return pd.DataFrame(scaled, columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)

    @staticmethod
    def _align(X: pd.DataFrame, expected: list[str] | None = None) -> pd.DataFrame:
        expected = expected or [b for b in S1_BANDS_ORDER if b in X.columns] or list(X.columns)
        X = X.copy()
        for missing in set(expected) - set(X.columns):
            log.warning("Feature %s absent from input — filling with 0", missing)
            X[missing] = 0.0
        return X[expected].astype("float32")


def build_selector(k: int | str = "all") -> SelectKBest:
    """Mutual-information feature selection.

    Mutual information rather than an F-test: SAR class separability is not
    monotonic — built-up and dense canopy both give high VV — and an F-test
    only sees linear association.
    """
    return SelectKBest(score_func=mutual_info_classif, k=k)


def demo() -> None:
    """Self-check: output shape, column-order invariance, missing-band handling."""
    n_bands = len(S1_BANDS_ORDER)
    expected_width = n_bands + n_bands * (n_bands - 1) // 2

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, n_bands)), columns=S1_BANDS_ORDER)

    engineer = SarFeatureEngineer().fit(X)
    out = engineer.transform(X)
    assert out.shape == (50, expected_width), out.shape
    assert list(out.columns) == engineer.feature_names_

    shuffled = X[list(reversed(S1_BANDS_ORDER))]
    assert np.allclose(engineer.transform(shuffled).to_numpy(), out.to_numpy())

    dropped = engineer.transform(X.drop(columns=["RVI"]))
    assert dropped.shape == out.shape
    assert not np.allclose(dropped["RVI"].to_numpy(), out["RVI"].to_numpy())

    print("sar_engineering demo OK")


if __name__ == "__main__":
    demo()
