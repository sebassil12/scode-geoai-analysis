"""End-to-end self-check on synthetic SAR data. No Earth Engine, no network.

    python -m ml_project.test_pipeline        (or: pytest ml_project/test_pipeline.py)

Covers the things that break silently: band ordering between training and
inference, the 1-4 label schema surviving XGBoost's zero-index requirement,
blocked prediction agreeing with whole-image prediction, nodata handling, and
champion selection.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from ml_project.config import CLASS_IDS, LABEL_COLUMN, NODATA_VALUE, S1_BANDS_ORDER
from ml_project.inference.classify import classify_raster, load_model
from ml_project.inference.vectorizer import compute_stats, to_geodataframe, vectorize
from ml_project.modeling.benchmark import comparison_table, run_benchmark
from ml_project.modeling.train import (
    available_models,
    build_candidate,
    load_dataset,
    save_artifacts,
)


def synthetic_samples(n_per_class: int = 120, seed: int = 0) -> pd.DataFrame:
    """Four separable Gaussian blobs in SAR feature space."""
    rng = np.random.default_rng(seed)
    frames = []
    for offset, class_id in enumerate(CLASS_IDS):
        values = rng.normal(loc=offset * 3.0, scale=1.0, size=(n_per_class, len(S1_BANDS_ORDER)))
        frame = pd.DataFrame(values, columns=S1_BANDS_ORDER)
        frame[LABEL_COLUMN] = class_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_raster(path: Path, frame: pd.DataFrame, height: int = 20, width: int = 20) -> None:
    """Write a 10 m UTM raster of feature values, with a nodata gutter."""
    pixels = frame[S1_BANDS_ORDER].sample(height * width, replace=True, random_state=1).to_numpy()
    data = pixels.T.reshape(len(S1_BANDS_ORDER), height, width).astype("float32")
    data[:, :, :5] = np.nan  # nodata gutter — must come back as NODATA_VALUE

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(S1_BANDS_ORDER),
        "dtype": "float32",
        "crs": "EPSG:32717",
        "transform": from_origin(700000, 9700000, 10, 10),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_label_schema_survives_every_model() -> None:
    """Each candidate must predict our 1-4 labels, not zero-indexed ones."""
    frame = synthetic_samples(60)
    X, y = frame[S1_BANDS_ORDER], frame[LABEL_COLUMN]

    for name in available_models():
        pipeline, _ = build_candidate(name)
        pipeline.fit(X, y)
        predictions = set(np.unique(pipeline.predict(X)).tolist())
        assert predictions <= set(CLASS_IDS), f"{name} predicted {predictions}"
        assert list(pipeline.classes_) == CLASS_IDS, f"{name} classes_={pipeline.classes_}"


def test_benchmark_picks_a_champion() -> None:
    frame = synthetic_samples(120)
    with tempfile.TemporaryDirectory() as tmp:
        results, champion, pipeline = run_benchmark(frame, n_iter=1, output_dir=Path(tmp))

        assert champion in {r.model for r in results}
        assert results[0].model == champion
        assert results == sorted(results, key=lambda r: r.rank_key, reverse=True)
        assert not comparison_table(results).empty
        assert all(0.0 <= r.holdout_f1_macro <= 1.0 for r in results)
        assert pipeline.predict(frame[S1_BANDS_ORDER]) is not None


def test_inference_round_trip() -> None:
    frame = synthetic_samples()
    pipeline, _ = build_candidate("random_forest")
    pipeline.fit(frame[S1_BANDS_ORDER], frame[LABEL_COLUMN])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        frame.to_csv(tmp / "samples.csv", index=False)
        assert len(load_dataset(tmp)) == len(frame)

        model_path = save_artifacts(pipeline, {"note": "self-check"}, tmp / "champion.joblib")
        metadata = model_path.with_name("champion_metadata.json").read_text()
        assert "selected_features" in metadata and "RobustScaler" in metadata

        raster = tmp / "corridor.tif"
        write_raster(raster, frame)
        classified = classify_raster(raster, load_model(model_path), block=16)

        with rasterio.open(classified) as src:
            out = src.read(1)
            assert src.count == 1 and src.dtypes[0] == "uint8"
            assert src.crs.to_string() == "EPSG:32717"

        assert np.all(out[:, :5] == NODATA_VALUE), "nodata gutter leaked a prediction"
        assert set(np.unique(out[:, 5:]).tolist()) <= set(CLASS_IDS), "unexpected class in output"

        # Block size must not change the answer.
        whole = classify_raster(raster, load_model(model_path), tmp / "whole.tif", block=4096)
        with rasterio.open(whole) as src:
            assert np.array_equal(out, src.read(1)), "blocked output differs from whole-image"

        stats = compute_stats(classified)
        assert stats["total_valid_pixels"] == int(np.sum(out != NODATA_VALUE))
        assert stats["pixel_area_m2"] == 100.0
        assert abs(sum(c["percentage"] for c in stats["classes"]) - 100.0) < 0.1

        collection = vectorize(classified)
        assert collection["type"] == "FeatureCollection"
        for feature in collection["features"]:
            lon, lat = feature["geometry"]["coordinates"][0][0]
            assert -82 < lon < -74 and -6 < lat < 2, "geometry is not in EPSG:4326"

        if collection["features"]:
            assert to_geodataframe(collection).crs.to_string() == "EPSG:4326"


def test_cli_artifacts_load_across_entry_points() -> None:
    """A model trained by one ``python -m`` CLI must load in another.

    Regression test for pickle's ``__main__`` trap: a class defined in the
    module being run as a script is recorded as ``__main__.ClassName`` and is
    unloadable everywhere else. Only a real subprocess round-trip catches it,
    which is why this test shells out instead of calling the functions.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "raw").mkdir()
        frame = synthetic_samples(60)
        frame.to_csv(tmp / "raw" / "samples.csv", index=False)
        write_raster(tmp / "raw" / "corridor.tif", frame)

        model = tmp / "xgb.joblib"
        _run_module(
            "ml_project.modeling.train",
            "--data",
            str(tmp / "raw"),
            "--model",
            "xgboost",  # the only candidate needing a pickled wrapper
            "--output",
            str(model),
            "--n-iter",
            "1",
        )
        assert model.exists()

        _run_module(
            "ml_project.inference.classify",
            "--input",
            str(tmp / "raw" / "corridor.tif"),
            "--model",
            str(model),
            "--geojson",
            str(tmp / "risk.geojson"),
        )
        assert (tmp / "raw" / "corridor_classified.tif").exists()
        assert (tmp / "risk.geojson").exists()


def _run_module(module: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, f"{module} failed:\n{result.stderr[-2000:]}"


if __name__ == "__main__":
    from ml_project.features.sar_engineering import demo

    demo()
    test_label_schema_survives_every_model()
    print("label schema OK")
    test_inference_round_trip()
    print("inference round-trip OK")
    test_benchmark_picks_a_champion()
    print("benchmark OK")
    test_cli_artifacts_load_across_entry_points()
    print("cli artifact round-trip OK")
