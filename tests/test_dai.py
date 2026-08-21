from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.dai import (
    extract_dai_bytes,
    extract_dai_path,
    extract_many_dai_bytes,
    restore_model_names_from_artifacts,
    sha256_bytes,
)


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


def test_dai_path_preserves_logical_name_when_resolved_path_is_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob_path = tmp_path / ("a" * 64)
    blob_path.write_bytes(synthetic_dai_bytes())
    logical_path = tmp_path / "snapshot" / "MOD_V_TER_Z1_001.dai"
    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        if path == logical_path:
            return blob_path
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    record, samples = extract_dai_path(logical_path, trust_source=True)

    assert record["modelo_nome"] == "MOD_V_TER_Z1_001"
    assert record["arquivo"] == "MOD_V_TER_Z1_001.dai"
    assert {sample["modelo_nome"] for sample in samples} == {"MOD_V_TER_Z1_001"}


def test_preextracted_hash_names_are_restored_without_deserializing(
    tmp_path: Path,
) -> None:
    raw = synthetic_dai_bytes()
    artifact_path = tmp_path / "MOD_V_TER_Z1_001.dai"
    artifact_path.write_bytes(raw)
    digest = sha256_bytes(raw)
    catalog = pd.DataFrame(
        {
            "modelo_nome": [digest],
            "arquivo": [digest],
            "artifact_sha256": [digest],
        }
    )
    samples = pd.DataFrame(
        {"modelo_nome": [digest], "latitude": [-30.03], "longitude": [-51.20]}
    )

    repaired_catalog, repaired_samples, audit = restore_model_names_from_artifacts(
        catalog, samples, (artifact_path,)
    )

    assert repaired_catalog.iloc[0]["modelo_nome"] == "MOD_V_TER_Z1_001"
    assert repaired_catalog.iloc[0]["arquivo"] == "MOD_V_TER_Z1_001.dai"
    assert repaired_samples.iloc[0]["modelo_nome"] == "MOD_V_TER_Z1_001"
    assert len(audit) == 1


def test_dai_infers_period_and_uses_equivalent_model_metrics() -> None:
    model_frame = pd.DataFrame(
        {
            "DATA": ["01/02/2025", "15/03/2025"],
            "LATITUDE": [-30.03, -30.04],
            "LONGITUDE": [-51.20, -51.19],
            "VALOR": [100.0, 110.0],
        }
    )
    package = {
        "dados": {"df": model_frame, "df_completo": model_frame},
        "transformacoes": {
            "X": model_frame[["VALOR"]],
            "y": model_frame["VALOR"],
        },
        "periodo_dados_mercado": {"coluna_data": "DATA"},
        "modelo": {"metrics": {"rsquared": 0.92, "rsquared_adj": 0.88, "mse_resid": 9.0}},
    }
    buffer = io.BytesIO()
    joblib.dump(package, buffer)

    record, samples = extract_dai_bytes(
        buffer.getvalue(), "MOD_V_TER_Z1_002.dai", trust_source=True
    )

    assert record["data_inicial"] == pd.Timestamp("2025-02-01")
    assert record["data_final"] == pd.Timestamp("2025-03-15")
    assert record["r2_ajustado"] == 0.88
    assert record["desvio_padrao_residuos"] == 3.0
    assert record["variavel_alvo"] == "VALOR"
    assert [sample["data_ref"] for sample in samples] == ["2025-02-01", "2025-03-15"]
