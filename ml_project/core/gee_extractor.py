"""Sentinel-1 GRD acquisition and pre-processing on Google Earth Engine.

Earth Engine's ``COPERNICUS/S1_GRD`` is already delivered with the orbit file
applied, GRD border noise removed, thermal noise removed, radiometric
calibration to sigma-nought (dB), and range-doppler terrain correction against
SRTM. The only pre-processing left to this module is speckle filtering and
temporal compositing — re-implementing the rest would undo work already done.

Requires the optional GEE extra: ``pip install earthengine-api geemap``.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING

import pandas as pd

from ml_project.config import (
    COMPOSITE_END,
    COMPOSITE_START,
    CORRIDORS,
    GEE_PROJECT,
    LABEL_COLUMN,
    RANDOM_STATE,
    S1_BANDS_ORDER,
    S1_COLLECTION,
    S1_INSTRUMENT_MODE,
    S1_ORBIT_PASS,
    S1_POLARISATIONS,
    S1_RESOLUTION_M,
    SPECKLE_FILTER_RADIUS_PX,
    WORLDCOVER_COLLECTION,
    WORLDCOVER_REMAP,
)
from ml_project.features.sar_engineering import add_glcm_textures, add_sar_indices

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee

log = logging.getLogger(__name__)

# ``getInfo`` caps a FeatureCollection at 5000 rows across all classes.
MAX_POINTS_PER_CLASS = 1200


class EarthEngineError(RuntimeError):
    """Raised when an Earth Engine call fails or the client is unavailable."""


def _ee() -> ModuleType:
    """Import Earth Engine, turning a missing dependency into a clear error."""
    try:
        import ee
    except ImportError as exc:
        raise EarthEngineError(
            "Earth Engine support is not installed: pip install earthengine-api geemap"
        ) from exc
    return ee


def init_ee(project: str = GEE_PROJECT) -> None:
    """Initialise Earth Engine, falling back to the OAuth flow when needed."""
    ee = _ee()
    try:
        ee.Initialize(project=project)
    except ee.EEException:
        log.info("No cached Earth Engine credentials — starting authentication")
        ee.Authenticate()
        ee.Initialize(project=project)
    log.info("Earth Engine initialised (project=%s)", project)


def corridor_geometry(name: str) -> ee.Geometry:
    """Return the buffered transmission-line corridor as an ``ee.Geometry``."""
    ee = _ee()
    if name not in CORRIDORS:
        raise KeyError(f"Unknown corridor {name!r}. Known corridors: {sorted(CORRIDORS)}")
    corridor = CORRIDORS[name]
    return ee.Geometry.LineString(corridor["line"]).buffer(corridor["buffer_m"])


def speckle_filter(image: ee.Image, radius: int = SPECKLE_FILTER_RADIUS_PX) -> ee.Image:
    """Focal-mean speckle filter applied in linear power, returned in dB.

    Averaging dB values directly is wrong: the mean of logarithms is not the
    logarithm of the mean, and it biases the result low. Convert to linear
    power, average, convert back.
    """
    ee = _ee()
    linear = ee.Image(10.0).pow(image.divide(10.0))
    smoothed = linear.focal_mean(radius, "square", "pixels")
    return smoothed.log10().multiply(10.0).rename(image.bandNames())


def s1_collection(
    aoi: ee.Geometry,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
) -> ee.ImageCollection:
    """IW dual-polarisation GRD scenes intersecting the AOI in the date window."""
    ee = _ee()
    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", S1_INSTRUMENT_MODE))
        .filter(ee.Filter.eq("resolution_meters", S1_RESOLUTION_M))
    )
    for polarisation in S1_POLARISATIONS:
        collection = collection.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", polarisation)
        )

    if S1_ORBIT_PASS:
        # Mixing orbit passes mixes incidence-angle geometries, which shifts
        # backscatter by several dB and shows up downstream as false change.
        collection = collection.filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT_PASS))

    return collection.select(list(S1_POLARISATIONS))


def sar_composite(
    aoi: ee.Geometry,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
) -> ee.Image:
    """Speckle-filtered median composite over the date window, clipped to the AOI."""
    return s1_collection(aoi, start, end).map(speckle_filter).median().clip(aoi)


def build_feature_image(
    aoi: ee.Geometry,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
) -> ee.Image:
    """Full feature image — VV, VH, ratios, GLCM textures — in ``S1_BANDS_ORDER``."""
    composite = sar_composite(aoi, start, end)
    return add_glcm_textures(add_sar_indices(composite)).select(S1_BANDS_ORDER).toFloat()


def worldcover_labels(aoi: ee.Geometry) -> ee.Image:
    """ESA WorldCover remapped to the 4-class schema; unmapped codes are masked."""
    ee = _ee()
    source, target = zip(*sorted(WORLDCOVER_REMAP.items()), strict=True)
    worldcover = ee.ImageCollection(WORLDCOVER_COLLECTION).first().select("Map").clip(aoi)
    return worldcover.remap(list(source), list(target)).rename(LABEL_COLUMN).toInt8()


def sample_training_points(
    aoi: ee.Geometry,
    points_per_class: int = MAX_POINTS_PER_CLASS,
    start: str = COMPOSITE_START,
    end: str = COMPOSITE_END,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified sample of feature values plus labels, as a DataFrame.

    WorldCover is a coarse proxy for a field survey — good enough for a
    baseline, not good enough to report as validated accuracy.
    """
    ee = _ee()

    if points_per_class > MAX_POINTS_PER_CLASS:
        raise ValueError(
            f"points_per_class={points_per_class} risks exceeding the 5000-row getInfo "
            f"limit across 4 classes; export the table to Drive instead."
        )

    stacked = build_feature_image(aoi, start, end).addBands(worldcover_labels(aoi))
    samples = stacked.stratifiedSample(
        numPoints=points_per_class,
        classBand=LABEL_COLUMN,
        region=aoi,
        scale=S1_RESOLUTION_M,
        seed=seed,
        geometries=False,
        tileScale=4,
    )

    try:
        rows = [feature["properties"] for feature in samples.getInfo()["features"]]
    except ee.EEException as exc:
        raise EarthEngineError(
            f"Sampling failed (often a memory or 5000-row limit): {exc}"
        ) from exc

    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        raise EarthEngineError("Sampling returned no usable points — check the AOI and dates")

    log.info(
        "Sampled %d points | class counts: %s",
        len(frame),
        frame[LABEL_COLUMN].value_counts().to_dict(),
    )
    return frame[S1_BANDS_ORDER + [LABEL_COLUMN]]
