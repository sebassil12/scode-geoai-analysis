"""Classified raster to vector boundaries for GridCanopy.

Turns high-risk pixels into polygons in EPSG:4326 — GeoJSON for the API, or a
GeoDataFrame ready for ``to_postgis``. Also reports per-class areas, which is
what the work-order side of GridCanopy actually budgets against.

CLI::

    python -m ml_project.inference.vectorizer --input corridor_classified.tif --geojson risk.geojson
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import LineString, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from ml_project.config import (
    CLASS_LABELS,
    CORRIDORS,
    HIGH_RISK_CLASS,
    MIN_POLYGON_AREA_M2,
    NODATA_VALUE,
)
from ml_project.core.geotiff_io import read_band

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"


def vectorize(
    classified_path: Path | str,
    target_class: int = HIGH_RISK_CLASS,
    corridor: str | None = None,
    max_distance_m: float | None = None,
    min_area_m2: float = MIN_POLYGON_AREA_M2,
) -> dict[str, Any]:
    """Convert pixels of ``target_class`` into a GeoJSON FeatureCollection.

    Given ``corridor`` and ``max_distance_m``, only polygons within that
    distance of the centreline survive — canopy outside the right-of-way is
    somebody else's budget. Areas are measured in the raster's projected CRS
    before reprojection, because area in degrees is meaningless.
    """
    band = read_band(classified_path)
    keep_zone = _corridor_zone(corridor, max_distance_m, band.crs)
    to_wgs84 = Transformer.from_crs(band.crs, WGS84, always_xy=True).transform

    features = []
    for geometry, _ in shapes(band.data, mask=band.data == target_class, transform=band.transform):
        polygon = shape(geometry)
        if polygon.area < min_area_m2:
            continue
        if keep_zone is not None and not polygon.intersects(keep_zone):
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely_transform(to_wgs84, polygon)),
                "properties": {
                    "class": target_class,
                    "label": CLASS_LABELS[target_class],
                    "area_ha": round(polygon.area / 10_000, 4),
                },
            }
        )

    log.info("Vectorised %d polygons of class %d", len(features), target_class)
    return {"type": "FeatureCollection", "features": features}


def to_geodataframe(collection: dict[str, Any]) -> gpd.GeoDataFrame:
    """Wrap a FeatureCollection as a GeoDataFrame for ``GeoDataFrame.to_postgis``."""
    import geopandas as gpd

    return gpd.GeoDataFrame.from_features(collection["features"], crs=WGS84)


def compute_stats(classified_path: Path | str) -> dict[str, Any]:
    """Per-class pixel counts, hectares and share of the classified area."""
    band = read_band(classified_path)
    total = int(np.sum(band.data != NODATA_VALUE))

    classes = []
    for class_id, label in CLASS_LABELS.items():
        count = int(np.sum(band.data == class_id))
        classes.append(
            {
                "id": class_id,
                "label": label,
                "pixels": count,
                "area_ha": round(count * band.pixel_area_m2 / 10_000, 2),
                "percentage": round(count / total * 100, 2) if total else 0.0,
            }
        )

    return {
        "total_valid_pixels": total,
        "pixel_area_m2": band.pixel_area_m2,
        "classes": classes,
    }


def _corridor_zone(
    corridor: str | None,
    max_distance_m: float | None,
    raster_crs: Any,
) -> BaseGeometry | None:
    """The corridor centreline buffered by ``max_distance_m``, in the raster's CRS."""
    if not corridor or max_distance_m is None:
        return None

    to_raster = Transformer.from_crs(WGS84, raster_crs, always_xy=True).transform
    line = LineString(CORRIDORS[corridor]["line"])
    return shapely_transform(to_raster, line).buffer(max_distance_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorise a classified SAR raster.")
    parser.add_argument("--input", required=True, help="Classified single-band GeoTIFF")
    parser.add_argument("--geojson", help="Write the FeatureCollection here")
    parser.add_argument("--stats", help="Write per-class area statistics here")
    parser.add_argument(
        "--target-class", type=int, default=HIGH_RISK_CLASS, choices=sorted(CLASS_LABELS)
    )
    parser.add_argument("--corridor", choices=sorted(CORRIDORS))
    parser.add_argument(
        "--max-distance-m", type=float, help="Distance from the corridor centreline"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.geojson:
        collection = vectorize(args.input, args.target_class, args.corridor, args.max_distance_m)
        Path(args.geojson).write_text(json.dumps(collection))
        log.info("Wrote %s", args.geojson)

    if args.stats:
        Path(args.stats).write_text(json.dumps(compute_stats(args.input), indent=2))
        log.info("Wrote %s", args.stats)


if __name__ == "__main__":
    main()
