"""All raster and GEE-export I/O. No module outside this one opens a GeoTIFF.

Keeping every ``rasterio.open`` behind this boundary means the classifier and
the vectoriser deal in arrays, and block sizes, COG profiles and nodata
conventions are decided in exactly one place.

CLI::

    python -m ml_project.core.geotiff_io --corridor cuenca-molleturo --output data/raw/
    python -m ml_project.core.geotiff_io --corridor cuenca-molleturo --mode samples
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from ml_project.config import (
    COMPOSITE_END,
    COMPOSITE_START,
    CORRIDORS,
    DATA_DIR,
    INFERENCE_BLOCK_PX,
    NODATA_VALUE,
    S1_RESOLUTION_M,
)

log = logging.getLogger(__name__)

BlockFunction = Callable[[np.ndarray], np.ndarray]

_COG_PROFILE: dict[str, Any] = {
    "driver": "COG",
    "compress": "DEFLATE",
    "BIGTIFF": "IF_SAFER",
}
_CLASSIFIED_PROFILE: dict[str, Any] = {
    "count": 1,
    "dtype": "uint8",
    "nodata": NODATA_VALUE,
    "driver": "GTiff",
    "compress": "DEFLATE",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}
# Set by the COG/GTiff driver itself; carrying them over from the source
# profile makes GDAL reject the creation options.
_DERIVED_KEYS = ("tiled", "blockxsize", "blockysize", "interleave", "BIGTIFF")


class RasterBand(NamedTuple):
    """A single raster band with the georeferencing needed to vectorise it."""

    data: np.ndarray
    transform: Any
    crs: Any
    pixel_area_m2: float


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_band(path: Path | str, index: int = 1) -> RasterBand:
    """Read one band plus its georeferencing."""
    try:
        with rasterio.open(path) as src:
            return RasterBand(
                data=src.read(index),
                transform=src.transform,
                crs=src.crs,
                pixel_area_m2=abs(src.transform.a * src.transform.e),
            )
    except RasterioIOError as exc:
        raise RasterioIOError(f"Could not read raster {path}: {exc}") from exc


def raster_nodata(path: Path | str) -> float | None:
    """The source raster's declared nodata value, if it has one."""
    with rasterio.open(path) as src:
        return src.nodata


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def blocked_apply(
    src_path: Path | str,
    dst_path: Path | str,
    block_fn: BlockFunction,
    block: int = INFERENCE_BLOCK_PX,
    expected_bands: int | None = None,
) -> Path:
    """Stream a raster through ``block_fn`` and write a single-band uint8 result.

    Blocked rather than whole-image because the engineered feature matrix, not
    the raster itself, is what exhausts RAM: every pixel expands to dozens of
    float64 interaction terms.

    ``block_fn`` receives a ``(bands, h, w)`` float32 array and must return an
    ``(h, w)`` uint8 array using ``NODATA_VALUE`` for pixels it skipped.
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(src_path) as src:
            if expected_bands is not None and src.count != expected_bands:
                raise ValueError(
                    f"{src_path.name} has {src.count} bands, expected {expected_bands}"
                )

            profile = {k: v for k, v in src.profile.items() if k not in _DERIVED_KEYS}
            with rasterio.open(dst_path, "w", **(profile | _CLASSIFIED_PROFILE)) as dst:
                for row in range(0, src.height, block):
                    for col in range(0, src.width, block):
                        window = Window(
                            col,
                            row,
                            min(block, src.width - col),
                            min(block, src.height - row),
                        )
                        result = block_fn(src.read(window=window).astype("float32"))
                        dst.write(result, 1, window=window)
    except RasterioIOError as exc:
        raise RasterioIOError(f"Raster I/O failed for {src_path}: {exc}") from exc

    log.info("Wrote %s", dst_path)
    return dst_path


def to_cog(src_path: Path | str, dst_path: Path | str) -> Path:
    """Rewrite a GeoTIFF with the COG driver (tiled, overviews, deflate)."""
    src_path, dst_path = Path(src_path), Path(dst_path)
    with rasterio.open(src_path) as src:
        profile = {k: v for k, v in src.profile.items() if k not in _DERIVED_KEYS}
        with rasterio.open(dst_path, "w", **(profile | _COG_PROFILE)) as dst:
            dst.descriptions = src.descriptions
            for index in range(1, src.count + 1):
                dst.write(src.read(index), index)
    return dst_path


# ---------------------------------------------------------------------------
# Earth Engine exports
# ---------------------------------------------------------------------------
def export_composite(
    corridor: str,
    output_dir: Path | str = DATA_DIR,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
    scale: int = S1_RESOLUTION_M,
) -> Path:
    """Download the Sentinel-1 feature composite for a corridor as a COG."""
    import geemap

    from ml_project.core.gee_extractor import build_feature_image, corridor_geometry, init_ee

    init_ee()
    aoi = corridor_geometry(corridor)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_path = output_dir / f"{corridor}_staging.tif"
    cog_path = output_dir / f"{corridor}.tif"

    log.info("Downloading %s at %d m (this can take several minutes)", corridor, scale)
    geemap.download_ee_image(
        image=build_feature_image(aoi, start, end),
        filename=str(staging_path),
        region=aoi,
        crs=CORRIDORS[corridor]["crs"],
        scale=scale,
        dtype="float32",
    )

    to_cog(staging_path, cog_path)
    staging_path.unlink()
    log.info("Wrote %s", cog_path)
    return cog_path


def export_samples(
    corridor: str,
    output_dir: Path | str = DATA_DIR,
    points_per_class: int = 1000,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
) -> Path:
    """Export a stratified training sample (features plus label) as CSV."""
    from ml_project.core.gee_extractor import corridor_geometry, init_ee, sample_training_points

    init_ee()
    frame = sample_training_points(corridor_geometry(corridor), points_per_class, start, end)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{corridor}_samples.csv"
    frame.to_csv(csv_path, index=False)
    log.info("Wrote %d samples to %s", len(frame), csv_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sentinel-1 SAR data from Earth Engine.")
    parser.add_argument("--corridor", required=True, choices=sorted(CORRIDORS))
    parser.add_argument("--output", default=str(DATA_DIR), help="Output directory")
    parser.add_argument("--mode", default="composite", choices=["composite", "samples"])
    parser.add_argument("--start", default=COMPOSITE_START)
    parser.add_argument("--end", default=COMPOSITE_END)
    parser.add_argument("--scale", type=int, default=S1_RESOLUTION_M, help="Export resolution (m)")
    parser.add_argument("--points-per-class", type=int, default=1000, help="samples mode only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.mode == "composite":
        export_composite(args.corridor, args.output, args.start, args.end, args.scale)
    else:
        export_samples(args.corridor, args.output, args.points_per_class, args.start, args.end)


if __name__ == "__main__":
    main()
