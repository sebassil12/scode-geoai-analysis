"""Local GeoTIFF classification using the trained champion model.

CLI::

    python -m ml_project.inference.classify --input data/raw/corridor.tif \\
        --model models/champion.joblib --geojson risk.geojson --stats stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_project.config import (
    CHAMPION_PATH,
    CLASS_LABELS,
    CORRIDORS,
    HIGH_RISK_CLASS,
    INFERENCE_BLOCK_PX,
    NODATA_VALUE,
    S1_BANDS_ORDER,
)
from ml_project.core.geotiff_io import blocked_apply, raster_nodata
from ml_project.inference.vectorizer import compute_stats, vectorize

log = logging.getLogger(__name__)


def load_model(model_path: Path | str = CHAMPION_PATH) -> Pipeline:
    """Load a serialised pipeline, failing loudly if it is missing."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def classify_raster(
    tiff_path: Path | str,
    model: Pipeline,
    output_path: Path | str | None = None,
    bands: list[str] = S1_BANDS_ORDER,
    block: int = INFERENCE_BLOCK_PX,
) -> Path:
    """Classify a multiband SAR GeoTIFF block by block.

    Returns the path to the single-band uint8 output (classes 1-4, nodata 0).
    """
    tiff_path = Path(tiff_path)
    output_path = (
        Path(output_path)
        if output_path
        else tiff_path.with_name(f"{tiff_path.stem}_classified.tif")
    )
    nodata = raster_nodata(tiff_path)

    def predict_block(block_data: np.ndarray) -> np.ndarray:
        return _predict_block(block_data, model, bands, nodata)

    return blocked_apply(tiff_path, output_path, predict_block, block, expected_bands=len(bands))


def _predict_block(
    block_data: np.ndarray,
    model: Pipeline,
    bands: list[str],
    nodata: float | None,
) -> np.ndarray:
    """Predict one ``(bands, h, w)`` tile, leaving invalid pixels as nodata."""
    n_bands, height, width = block_data.shape
    flat = block_data.reshape(n_bands, -1).T

    valid = np.isfinite(flat).all(axis=1)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.all(flat == nodata, axis=1)

    prediction = np.full(height * width, NODATA_VALUE, dtype=np.uint8)
    if valid.any():
        prediction[valid] = model.predict(pd.DataFrame(flat[valid], columns=bands)).astype(np.uint8)

    return prediction.reshape(height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a Sentinel-1 GeoTIFF.")
    parser.add_argument("--input", required=True, help="Multiband SAR GeoTIFF")
    parser.add_argument("--model", default=str(CHAMPION_PATH))
    parser.add_argument("--output", help="Classified GeoTIFF (default: <input>_classified.tif)")
    parser.add_argument("--geojson", help="Also write high-risk polygons here")
    parser.add_argument("--stats", help="Also write per-class area statistics here")
    parser.add_argument(
        "--corridor", choices=sorted(CORRIDORS), help="Filter polygons to this corridor"
    )
    parser.add_argument(
        "--max-distance-m", type=float, help="Distance from the corridor centreline"
    )
    parser.add_argument("--block", type=int, default=INFERENCE_BLOCK_PX)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    classified = classify_raster(args.input, load_model(args.model), args.output, block=args.block)

    if args.geojson:
        collection: dict[str, Any] = vectorize(
            classified, HIGH_RISK_CLASS, args.corridor, args.max_distance_m
        )
        Path(args.geojson).write_text(json.dumps(collection))
        log.info(
            "Wrote %d %s polygons to %s",
            len(collection["features"]),
            CLASS_LABELS[HIGH_RISK_CLASS],
            args.geojson,
        )

    if args.stats:
        Path(args.stats).write_text(json.dumps(compute_stats(classified), indent=2))
        log.info("Wrote %s", args.stats)


if __name__ == "__main__":
    main()
