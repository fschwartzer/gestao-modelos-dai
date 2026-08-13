from __future__ import annotations

import io
import sqlite3
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from src.config import (
    DEMO_CATALOG,
    DEMO_DB,
    DEMO_SAMPLES,
    POA_LATITUDE_RANGE,
    POA_LONGITUDE_RANGE,
)


WORK_QUERY = """
SELECT
    t.trabalho_id,
    t.nome,
    t.nome_original,
    t.tipo_codigo,
    t.tipo_label,
    t.ano,
    t.total_registros,
    t.total_imoveis,
    t.total_modelos,
    i.imovel_id,
    i.endereco,
    i.numero,
    i.label AS imovel_label,
    i.coord_x,
    i.coord_y,
    im.modelo_nome
FROM trabalhos t
JOIN trabalho_imoveis i
  ON i.trabalho_id = t.trabalho_id
JOIN trabalho_imovel_modelos im
  ON im.imovel_id = i.imovel_id
"""


def _in_range(value: float, limits: tuple[float, float]) -> bool:
    return limits[0] <= value <= limits[1]


def normalize_coordinate_pair(coord_x: object, coord_y: object) -> tuple[float, float, str]:
    """Retorna latitude, longitude e status de normalização.

    O banco histórico contém a maior parte dos registros como X=longitude e
    Y=latitude, mas há registros antigos invertidos. A decisão usa limites
    territoriais amplos de Porto Alegre.
    """

    try:
        x = float(coord_x)
        y = float(coord_y)
    except (TypeError, ValueError):
        return np.nan, np.nan, "ausente"

    xy_ok = _in_range(x, POA_LONGITUDE_RANGE) and _in_range(y, POA_LATITUDE_RANGE)
    yx_ok = _in_range(y, POA_LONGITUDE_RANGE) and _in_range(x, POA_LATITUDE_RANGE)

    if xy_ok and not yx_ok:
        return y, x, "original"
    if yx_ok and not xy_ok:
        return x, y, "invertida_corrigida"
    return np.nan, np.nan, "fora_limites"


def normalize_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    normalized = result.apply(
        lambda row: normalize_coordinate_pair(row.get("coord_x"), row.get("coord_y")),
        axis=1,
        result_type="expand",
    )
    normalized.columns = ["latitude", "longitude", "status_coordenada"]
    return pd.concat([result, normalized], axis=1)


def _read_sqlite_connection(connection: sqlite3.Connection) -> pd.DataFrame:
    available = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    required = {"trabalhos", "trabalho_imoveis", "trabalho_imovel_modelos"}
    missing = required - available
    if missing:
        raise ValueError(f"Banco incompatível. Tabelas ausentes: {', '.join(sorted(missing))}")
    frame = pd.read_sql_query(WORK_QUERY, connection)
    return normalize_coordinates(frame)


def load_sqlite_path(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        return _read_sqlite_connection(connection)
    finally:
        connection.close()


def load_sqlite_bytes(raw: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        temporary.write(raw)
        temporary.flush()
        return load_sqlite_path(temporary.name)


def load_csv_source(source: str | Path | bytes | BinaryIO | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, bytes):
        compression = "gzip" if source.startswith(b"\x1f\x8b") else "infer"
        return pd.read_csv(io.BytesIO(source), compression=compression)
    if hasattr(source, "read"):
        return pd.read_csv(source)
    path = Path(source)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def standardize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "modelo_nome" not in result.columns:
        raise ValueError("O catálogo deve conter a coluna 'modelo_nome'.")
    for column in ("data_inicial", "data_final", "artifact_mtime"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    for column in ("n_modelo", "n_completo", "n_outliers", "r2_ajustado"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(subset=["modelo_nome"], keep="last")


def standardize_samples(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"modelo_nome", "latitude", "longitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Amostras incompatíveis. Colunas ausentes: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    if "data_ref" in result.columns:
        result["data_ref"] = pd.to_datetime(result["data_ref"], errors="coerce")
    return result.dropna(subset=["latitude", "longitude"])


def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    works = load_sqlite_path(DEMO_DB)
    catalog = standardize_catalog(load_csv_source(DEMO_CATALOG))
    samples = standardize_samples(load_csv_source(DEMO_SAMPLES))
    return works, catalog, samples


def unique_work_points(works: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trabalho_id",
        "nome",
        "tipo_codigo",
        "tipo_label",
        "ano",
        "imovel_id",
        "endereco",
        "numero",
        "latitude",
        "longitude",
        "status_coordenada",
    ]
    available = [column for column in columns if column in works.columns]
    return works[available].drop_duplicates(subset=["trabalho_id", "imovel_id"])
