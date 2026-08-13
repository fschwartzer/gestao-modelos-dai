from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEMO_SAMPLES
from src.data import load_csv_source, load_demo_data, normalize_coordinate_pair, unique_work_points


def test_normalize_original_coordinates() -> None:
    latitude, longitude, status = normalize_coordinate_pair(-51.20, -30.03)
    assert latitude == -30.03
    assert longitude == -51.20
    assert status == "original"


def test_normalize_inverted_coordinates() -> None:
    latitude, longitude, status = normalize_coordinate_pair(-30.03, -51.20)
    assert latitude == -30.03
    assert longitude == -51.20
    assert status == "invertida_corrigida"


def test_demo_data_loads_and_has_valid_points() -> None:
    works, catalog, samples = load_demo_data()
    assert not works.empty
    assert not catalog.empty
    assert not samples.empty
    points = unique_work_points(works)
    assert points["latitude"].between(-30.35, -29.80).all()
    assert points["longitude"].between(-51.40, -50.95).all()
    assert works["modelo_nome"].nunique() >= 10


def test_gzip_csv_can_be_loaded_from_uploaded_bytes() -> None:
    frame = load_csv_source(Path(DEMO_SAMPLES).read_bytes())
    assert not frame.empty
    assert {"modelo_nome", "latitude", "longitude"}.issubset(frame.columns)
