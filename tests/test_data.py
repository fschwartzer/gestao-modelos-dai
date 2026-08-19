from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DEMO_SAMPLES
from src.data import (
    _read_sqlite_connection,
    analysis_availability,
    load_csv_source,
    load_demo_data,
    load_sqlite_bytes,
    normalize_coordinate_pair,
    standardize_samples,
    unique_work_points,
)


def test_sqlite_optional_work_date_is_loaded_when_available() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE trabalhos (
            trabalho_id TEXT PRIMARY KEY, nome TEXT, nome_original TEXT,
            tipo_codigo TEXT, tipo_label TEXT, ano INTEGER, data_emissao TEXT,
            total_registros INTEGER, total_imoveis INTEGER, total_modelos INTEGER
        );
        CREATE TABLE trabalho_imoveis (
            imovel_id INTEGER PRIMARY KEY, trabalho_id TEXT, endereco TEXT,
            numero TEXT, label TEXT, coord_x REAL, coord_y REAL
        );
        CREATE TABLE trabalho_imovel_modelos (imovel_id INTEGER, modelo_nome TEXT);
        INSERT INTO trabalhos VALUES
            ('T1', 'Trabalho', 'Trabalho', 'LA', 'Laudo', 2026, '2026-07-15', 1, 1, 1);
        INSERT INTO trabalho_imoveis VALUES
            (1, 'T1', 'Rua A', '1', 'Rua A, 1', -51.20, -30.03);
        INSERT INTO trabalho_imovel_modelos VALUES (1, 'MOD_V_TER_Z1_001A');
        """
    )
    try:
        frame = _read_sqlite_connection(connection)
    finally:
        connection.close()
    assert frame.iloc[0]["data_trabalho"] == pd.Timestamp("2026-07-15")


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
    assert "data_trabalho" in works


def test_gzip_csv_can_be_loaded_from_uploaded_bytes() -> None:
    frame = load_csv_source(Path(DEMO_SAMPLES).read_bytes())
    assert not frame.empty
    assert {"modelo_nome", "latitude", "longitude"}.issubset(frame.columns)


def test_invalid_sqlite_upload_is_rejected_before_opening() -> None:
    try:
        load_sqlite_bytes(b"arquivo-invalido")
    except ValueError as error:
        assert "cabeçalho SQLite" in str(error)
    else:
        raise AssertionError("Um arquivo sem cabeçalho SQLite foi aceito.")


def test_demo_sqlite_can_be_loaded_from_uploaded_bytes() -> None:
    from src.config import DEMO_DB

    frame = load_sqlite_bytes(Path(DEMO_DB).read_bytes())
    assert not frame.empty
    assert frame["trabalho_id"].nunique() == 650


def test_uploaded_sample_coordinates_are_normalized() -> None:
    frame = pd.DataFrame(
        {
            "modelo_nome": ["MODELO_A"],
            "latitude": [-51.20],
            "longitude": [-30.03],
        }
    )
    normalized = standardize_samples(frame)
    assert normalized.iloc[0]["latitude"] == -30.03
    assert normalized.iloc[0]["longitude"] == -51.20
    assert normalized.iloc[0]["status_coordenada"] == "invertida_corrigida"


def test_analysis_availability_with_independent_sources() -> None:
    empty = pd.DataFrame()
    works = pd.DataFrame({"trabalho_id": ["T1"]})
    catalog = pd.DataFrame({"modelo_nome": ["M1"]})
    samples = pd.DataFrame(
        {"modelo_nome": ["M1"], "latitude": [-30.03], "longitude": [-51.20]}
    )

    sqlite_only = analysis_availability(works, empty, empty)
    assert sqlite_only == {
        "Visão geral": True,
        "Modelos": False,
        "Cobertura": False,
        "Prioridades": False,
        "Metodologia": True,
    }

    dai_only = analysis_availability(empty, catalog, samples)
    assert dai_only == {
        "Visão geral": False,
        "Modelos": True,
        "Cobertura": False,
        "Prioridades": False,
        "Metodologia": True,
    }

    complete = analysis_availability(works, catalog, samples)
    assert all(complete.values())

    samples_and_sqlite = analysis_availability(works, empty, samples)
    assert samples_and_sqlite["Cobertura"] is False
    assert samples_and_sqlite["Prioridades"] is False
