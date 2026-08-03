"""Centralised configuration for the GridCanopy SAR risk engine.

Every tunable value lives here. No hardcoded paths, dates or magic numbers
anywhere else in the package. Environment variables override the defaults so
the same code runs on a laptop and in a container without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Google Earth Engine -----------------------------------------------------
GEE_PROJECT: str = os.getenv("GEE_PROJECT", "ee-gis-scode")

# --- Utility corridors -------------------------------------------------------
# Vertices along the transmission line, WGS84 (lon, lat). ``buffer_m`` is applied
# on each side, so the exported ROI is a thin ribbon rather than a whole province.
CORRIDORS: dict[str, dict] = {
    "cuenca-molleturo": {
        "line": [
            (-79.005, -2.900),
            (-79.060, -2.850),
            (-79.150, -2.800),
            (-79.250, -2.780),
            (-79.375, -2.803),
        ],
        "buffer_m": 500,
        "crs": "EPSG:32717",  # UTM 17S — metric CRS for buffering and export
    },
}

# --- Sentinel-1 acquisition --------------------------------------------------
S1_COLLECTION: str = "COPERNICUS/S1_GRD"
S1_INSTRUMENT_MODE: str = "IW"
S1_POLARISATIONS: tuple[str, ...] = ("VV", "VH")
S1_ORBIT_PASS: str | None = "DESCENDING"  # None keeps both passes
S1_RESOLUTION_M: int = 10

# 3-month window: temporal median compositing is what suppresses SAR speckle.
COMPOSITE_START: str = os.getenv("COMPOSITE_START", "2024-06-01")
COMPOSITE_END: str = os.getenv("COMPOSITE_END", "2024-09-01")

SPECKLE_FILTER_RADIUS_PX: int = 1  # focal mean 3x3
GLCM_WINDOW_PX: int = 3  # 3x3 — wider windows blur the right-of-way / forest edge

# dB range used to rescale backscatter to 0-255 before GLCM (GEE needs integers).
DB_MIN: float = -30.0
DB_MAX: float = 5.0

# --- Feature schema ----------------------------------------------------------
# Single source of truth. Must match the GEE export band order exactly.
S1_BANDS_ORDER: list[str] = [
    "VV",
    "VH",
    "VV_VH_ratio",
    "RVI",
    "VV_contrast",
    "VV_ent",
    "VH_contrast",
    "VH_ent",
]

# --- Classes -----------------------------------------------------------------
CLASS_LABELS: dict[int, str] = {
    1: "Dense Vegetation",
    2: "Sparse/Low Vegetation",
    3: "Soil/Bare",
    4: "Built-up",
}
CLASS_IDS: list[int] = sorted(CLASS_LABELS)
HIGH_RISK_CLASS: int = 1  # dense canopy beside a corridor is the clearance risk
NODATA_VALUE: int = 0  # classes start at 1, so 0 is free for nodata

# ESA WorldCover v200 -> our 4 classes. Unlisted codes (70 snow, 80 water) are
# dropped from training: SAR over water is specular and would poison class 3.
WORLDCOVER_COLLECTION: str = "ESA/WorldCover/v200"
WORLDCOVER_REMAP: dict[int, int] = {
    10: 1,
    95: 1,  # tree cover, mangroves
    20: 2,
    30: 2,
    40: 2,
    90: 2,
    100: 2,  # shrub, grass, crop, wetland, moss
    60: 3,  # bare / sparse vegetation
    50: 4,  # built-up
}

# --- Modelling ---------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.3
CV_FOLDS: int = 5
SEARCH_ITERATIONS: int = 25
LABEL_COLUMN: str = "class"

# Parallelise at the search level only. Setting n_jobs=-1 on both the search and
# the estimator oversubscribes the machine — 16 cores becomes 16x16 threads
# fighting for them, and the benchmark runs slower than single-threaded.
SEARCH_N_JOBS: int = -1
MODEL_N_JOBS: int = 1

# --- Paths -------------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data/raw"))
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))
CHAMPION_PATH: Path = MODEL_DIR / "champion.joblib"
BENCHMARK_REPORT: Path = MODEL_DIR / "benchmark_report.json"

# --- Inference ---------------------------------------------------------------
INFERENCE_BLOCK_PX: int = 512  # chunk size for windowed prediction
MIN_POLYGON_AREA_M2: float = 100.0  # one 10 m pixel; anything smaller is speckle
