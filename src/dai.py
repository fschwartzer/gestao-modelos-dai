from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def canonical_name(source_name: str | Path) -> str:
    """Retorna o identificador histórico do modelo a partir do arquivo."""

    stem = Path(str(source_name)).stem
    return re.sub(r"\(\d+\)$", "", stem).strip()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> object | None:
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        if alias.casefold() in normalized:
            return normalized[alias.casefold()]
    return None


def extract_package(
    package: object,
    *,
    source_name: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_mtime: str | None = None,
    include_personal_metadata: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Extrai somente metadados e coordenadas necessárias ao diagnóstico espacial."""

    if not isinstance(package, Mapping):
        raise TypeError("O pacote não contém um dicionário no nível raiz.")

    data_block = _mapping(package.get("dados"))
    transformations = _mapping(package.get("transformacoes"))
    model_block = _mapping(package.get("modelo"))
    diagnostics = _mapping(model_block.get("diagnosticos"))
    general = _mapping(diagnostics.get("gerais"))
    period = _mapping(package.get("periodo_dados_mercado"))
    appraisal = _mapping(package.get("avaliacao"))
    author = _mapping(package.get("elaborador"))

    model_frame = data_block.get("df")
    complete_frame = data_block.get("df_completo")
    excluded = data_block.get("outliers_excluidos", [])
    predictors = transformations.get("X")
    target = transformations.get("y")

    n_model = len(model_frame) if isinstance(model_frame, pd.DataFrame) else None
    n_complete = len(complete_frame) if isinstance(complete_frame, pd.DataFrame) else n_model
    try:
        n_outliers = len(excluded) if excluded is not None else None
    except TypeError:
        n_outliers = None
    pct_outliers = (
        100 * n_outliers / n_complete
        if n_complete and n_outliers is not None
        else None
    )
    model_name = canonical_name(source_name)

    record: dict[str, object] = {
        "modelo_nome": model_name,
        "arquivo": Path(source_name).name,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_mtime": artifact_mtime,
        "versao_formato": package.get("versao"),
        "data_inicial": period.get("data_inicial"),
        "data_final": period.get("data_final"),
        "coluna_data": period.get("coluna_data"),
        "n_modelo": n_model,
        "n_completo": n_complete,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "variavel_alvo": getattr(target, "name", None),
        "preditoras_json": json.dumps(
            list(predictors.columns) if isinstance(predictors, pd.DataFrame) else [],
            ensure_ascii=False,
        ),
        "transformacoes_json": json.dumps(
            transformations.get("info", []), ensure_ascii=False, default=str
        ),
        "r2": _python_scalar(general.get("r2")),
        "r2_ajustado": _python_scalar(general.get("r2_ajustado")),
        "desvio_padrao_residuos": _python_scalar(general.get("desvio_padrao_residuos")),
        "mse": _python_scalar(general.get("mse")),
        "equacao": diagnostics.get("equacao"),
        "tipo_y": appraisal.get("tipo_y"),
        "coluna_area": appraisal.get("coluna_area"),
        "elaborador_nome": author.get("nome_completo") if include_personal_metadata else None,
        "elaborador_lotacao": author.get("lotacao") if include_personal_metadata else None,
        "status": "a_classificar",
    }

    samples: list[dict[str, object]] = []
    if not isinstance(model_frame, pd.DataFrame):
        return record, samples

    latitude_column = _find_column(model_frame, ("lat", "latitude"))
    longitude_column = _find_column(model_frame, ("lon", "lng", "longitude"))
    if latitude_column is None or longitude_column is None:
        return record, samples

    coordinates = pd.DataFrame(
        {
            "latitude": pd.to_numeric(model_frame[latitude_column], errors="coerce"),
            "longitude": pd.to_numeric(model_frame[longitude_column], errors="coerce"),
        },
        index=model_frame.index,
    )
    date_column = period.get("coluna_data")
    dates = (
        pd.to_datetime(model_frame[date_column], errors="coerce")
        if date_column in model_frame.columns
        else pd.Series(pd.NaT, index=model_frame.index)
    )
    for sequence, (index, row) in enumerate(coordinates.dropna().iterrows(), start=1):
        value_date = dates.loc[index] if index in dates.index else pd.NaT
        samples.append(
            {
                "modelo_nome": model_name,
                "sample_id": f"{model_name}-{sequence:05d}",
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "data_ref": None if pd.isna(value_date) else value_date.date().isoformat(),
            }
        )
    return record, samples


def extract_dai_bytes(
    raw: bytes,
    source_name: str,
    *,
    trust_source: bool = False,
    include_personal_metadata: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Desserializa um upload confiável; joblib/pickle pode executar código."""

    if not trust_source:
        raise PermissionError("Confirme explicitamente que a origem do arquivo .DAI é confiável.")
    if not raw:
        raise ValueError("O arquivo .DAI está vazio.")
    package = joblib.load(io.BytesIO(raw))
    return extract_package(
        package,
        source_name=source_name,
        artifact_sha256=sha256_bytes(raw),
        artifact_size_bytes=len(raw),
        include_personal_metadata=include_personal_metadata,
    )


def extract_dai_path(
    path: str | Path,
    *,
    trust_source: bool = False,
    include_personal_metadata: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Desserializa um arquivo local confiável e registra sua proveniência."""

    if not trust_source:
        raise PermissionError("Confirme explicitamente que a origem do arquivo .DAI é confiável.")
    resolved = Path(path).resolve()
    package = joblib.load(resolved)
    return extract_package(
        package,
        source_name=resolved.name,
        artifact_sha256=sha256_file(resolved),
        artifact_size_bytes=resolved.stat().st_size,
        artifact_mtime=pd.Timestamp(resolved.stat().st_mtime, unit="s").isoformat(),
        include_personal_metadata=include_personal_metadata,
    )


def extract_many_dai_bytes(
    sources: Sequence[tuple[str, bytes]],
    *,
    trust_source: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processa um lote sem interromper os arquivos válidos quando um item falha."""

    if not trust_source:
        raise PermissionError("Confirme explicitamente que a origem dos arquivos .DAI é confiável.")
    catalog: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    seen_model_names: set[str] = set()
    for source_name, raw in sources:
        try:
            record, extracted_samples = extract_dai_bytes(
                raw,
                source_name,
                trust_source=True,
            )
            model_key = str(record["modelo_nome"]).casefold()
            if model_key in seen_model_names:
                raise ValueError(
                    "Identificador de modelo duplicado no lote; renomeie ou envie somente uma versão."
                )
            seen_model_names.add(model_key)
            catalog.append(record)
            samples.extend(extracted_samples)
        except Exception as error:  # mantém o lote e torna incompatibilidades auditáveis
            errors.append(
                {"arquivo": Path(source_name).name, "erro": f"{type(error).__name__}: {error}"}
            )
    return (
        pd.DataFrame(catalog),
        pd.DataFrame(samples),
        pd.DataFrame(errors, columns=["arquivo", "erro"]),
    )
