from __future__ import annotations

import io

import joblib
import pandas as pd
import pytest

from src.dai import extract_dai_bytes, extract_many_dai_bytes


def synthetic_dai_bytes() -> bytes:
    model_frame = pd.DataFrame(
        {
            "lat": [-30.03, -30.04, -30.02],
            "lon": [-51.20, -51.19, -51.18],
            "data_negocio": ["2025-01-10", "2025-02-10", "2025-03-10"],
            "valor": [100.0, 110.0, 120.0],
        }
    )
    package = {
        "versao": 2,
        "dados": {"df": model_frame, "df_completo": model_frame, "outliers_excluidos": []},
        "transformacoes": {
            "X": model_frame[["valor"]],
            "y": model_frame["valor"],
            "info": [],
        },
        "modelo": {"diagnosticos": {"gerais": {"r2_ajustado": 0.91}}},
        "periodo_dados_mercado": {
            "data_inicial": "2025-01-01",
            "data_final": "2025-03-31",
            "coluna_data": "data_negocio",
        },
    }
    buffer = io.BytesIO()
    joblib.dump(package, buffer)
    return buffer.getvalue()


def test_dai_requires_explicit_trust() -> None:
    with pytest.raises(PermissionError, match="origem"):
        extract_dai_bytes(synthetic_dai_bytes(), "MOD_V_TER_Z1_001.dai")


def test_dai_upload_extracts_catalog_and_spatial_sample() -> None:
    record, samples = extract_dai_bytes(
        synthetic_dai_bytes(),
        "MOD_V_TER_Z1_001.dai",
        trust_source=True,
    )
    assert record["modelo_nome"] == "MOD_V_TER_Z1_001"
    assert record["n_modelo"] == 3
    assert record["r2_ajustado"] == 0.91
    assert len(record["artifact_sha256"]) == 64
    assert len(samples) == 3
    assert samples[0]["data_ref"] == "2025-01-10"


def test_dai_batch_keeps_valid_files_and_reports_invalid_ones() -> None:
    catalog, samples, errors = extract_many_dai_bytes(
        (
            ("MOD_V_TER_Z1_001.dai", synthetic_dai_bytes()),
            ("corrompido.dai", b"nao-e-joblib"),
        ),
        trust_source=True,
    )
    assert len(catalog) == 1
    assert len(samples) == 3
    assert errors["arquivo"].tolist() == ["corrompido.dai"]


def test_dai_batch_rejects_duplicate_canonical_model_names() -> None:
    raw = synthetic_dai_bytes()
    catalog, samples, errors = extract_many_dai_bytes(
        (
            ("MOD_V_TER_Z1_001.dai", raw),
            ("MOD_V_TER_Z1_001(1).dai", raw),
        ),
        trust_source=True,
    )
    assert len(catalog) == 1
    assert len(samples) == 3
    assert "duplicado" in errors.iloc[0]["erro"]
