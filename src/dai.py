from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DAI_EXTRACTOR_VERSION = "3.1"
SHA256_IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


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


def _first_available(*values: object) -> object | None:
    """Retorna o primeiro escalar efetivamente informado."""

    for value in values:
        value = _python_scalar(value)
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return None


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> object | None:
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        if alias.casefold() in normalized:
            return normalized[alias.casefold()]
    return None


def is_sha256_identifier(value: object) -> bool:
    """Identifica o nome opaco introduzido pelo cache de blobs."""

    return bool(SHA256_IDENTIFIER_PATTERN.fullmatch(str(value or "").strip()))


def restore_model_names_from_artifacts(
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    artifact_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restaura nomes lógicos em catálogos antigos gerados de links do HF.

    A restauração só ocorre quando o identificador atual parece um SHA-256 e o
    hash de conteúdo registrado no catálogo corresponde de forma não ambígua a
    um `.dai` disponível. O conteúdo pickle não é desserializado nesta etapa.
    """

    repaired_catalog = catalog.copy()
    repaired_samples = samples.copy()
    audit_columns = ["modelo_nome_anterior", "modelo_nome", "arquivo"]
    if (
        repaired_catalog.empty
        or "modelo_nome" not in repaired_catalog
        or "artifact_sha256" not in repaired_catalog
        or not artifact_paths
    ):
        return repaired_catalog, repaired_samples, pd.DataFrame(columns=audit_columns)

    artifacts_by_hash: dict[str, list[tuple[str, str]]] = {}
    for artifact_path in artifact_paths:
        logical_path = Path(artifact_path)
        digest = sha256_file(logical_path).casefold()
        artifacts_by_hash.setdefault(digest, []).append(
            (canonical_name(logical_path.name), logical_path.name)
        )

    replacements: dict[str, str] = {}
    audit_records: list[dict[str, str]] = []
    for index, row in repaired_catalog.iterrows():
        current_name = str(row.get("modelo_nome") or "").strip()
        digest = str(row.get("artifact_sha256") or "").strip().casefold()
        candidates = artifacts_by_hash.get(digest, [])
        if not is_sha256_identifier(current_name) or len(candidates) != 1:
            continue
        restored_name, logical_filename = candidates[0]
        repaired_catalog.at[index, "modelo_nome"] = restored_name
        if "arquivo" in repaired_catalog:
            repaired_catalog.at[index, "arquivo"] = logical_filename
        replacements[current_name.casefold()] = restored_name
        audit_records.append(
            {
                "modelo_nome_anterior": current_name,
                "modelo_nome": restored_name,
                "arquivo": logical_filename,
            }
        )

    if replacements and "modelo_nome" in repaired_samples:
        repaired_samples["modelo_nome"] = repaired_samples["modelo_nome"].map(
            lambda value: replacements.get(str(value).casefold(), value)
        )

    return (
        repaired_catalog,
        repaired_samples,
        pd.DataFrame(audit_records, columns=audit_columns),
    )


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
    model_metrics = _mapping(model_block.get("metrics"))
    period = _mapping(package.get("periodo_dados_mercado"))
    appraisal = _mapping(package.get("avaliacao"))
    author = _mapping(package.get("elaborador"))

    model_frame = data_block.get("df")
    complete_frame = data_block.get("df_completo")
    excluded = data_block.get("outliers_excluidos", [])
    predictors = transformations.get("X")
    target = transformations.get("y")

    n_model = (
        len(model_frame)
        if isinstance(model_frame, pd.DataFrame)
        else _first_available(model_metrics.get("nobs"), general.get("n"))
    )
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

    date_column = period.get("coluna_data")
    if isinstance(model_frame, pd.DataFrame) and date_column not in model_frame.columns:
        date_column = _find_column(
            model_frame,
            ("data", "data_negocio", "data_ref", "data_observacao", "date"),
        )
    dates = (
        pd.to_datetime(
            model_frame[date_column],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
        if isinstance(model_frame, pd.DataFrame) and date_column in model_frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    valid_dates = dates.dropna()
    data_initial = _first_available(
        period.get("data_inicial"),
        valid_dates.min() if not valid_dates.empty else None,
    )
    data_final = _first_available(
        period.get("data_final"),
        valid_dates.max() if not valid_dates.empty else None,
    )

    target_name = getattr(target, "name", None)
    if target_name is None and isinstance(target, pd.DataFrame) and len(target.columns) == 1:
        target_name = target.columns[0]
    target_name = _first_available(
        target_name,
        appraisal.get("variavel_alvo"),
        appraisal.get("coluna_y"),
    )

    predictor_columns = (
        list(predictors.columns)
        if isinstance(predictors, pd.DataFrame)
        else [predictors.name]
        if isinstance(predictors, pd.Series) and predictors.name is not None
        else []
    )
    mse = _first_available(general.get("mse"), model_metrics.get("mse_resid"))
    residual_std = _first_available(general.get("desvio_padrao_residuos"))
    if residual_std is None and isinstance(mse, (int, float)) and mse >= 0:
        residual_std = math.sqrt(mse)

    record: dict[str, object] = {
        "modelo_nome": model_name,
        "arquivo": Path(source_name).name,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_mtime": artifact_mtime,
        "versao_formato": package.get("versao"),
        "data_inicial": data_initial,
        "data_final": data_final,
        "coluna_data": date_column,
        "n_modelo": n_model,
        "n_completo": n_complete,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "variavel_alvo": target_name,
        "preditoras_json": json.dumps(
            predictor_columns,
            ensure_ascii=False,
        ),
        "transformacoes_json": json.dumps(
            transformations.get("info", []), ensure_ascii=False, default=str
        ),
        "r2": _first_available(general.get("r2"), model_metrics.get("rsquared")),
        "r2_ajustado": _first_available(
            general.get("r2_ajustado"), model_metrics.get("rsquared_adj")
        ),
        "desvio_padrao_residuos": residual_std,
        "mse": mse,
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
    if dates.empty:
        dates = pd.Series(pd.NaT, index=model_frame.index)
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
    logical_path = Path(path)
    resolved = logical_path.resolve(strict=True)
    artifact_stat = resolved.stat()
    package = joblib.load(resolved)
    return extract_package(
        package,
        # `resolved.name` pode ser o SHA do blob quando o arquivo veio de um
        # snapshot do Hugging Face. O nome lógico é parte do contrato do modelo.
        source_name=logical_path.name,
        artifact_sha256=sha256_file(resolved),
        artifact_size_bytes=artifact_stat.st_size,
        artifact_mtime=pd.Timestamp(artifact_stat.st_mtime, unit="s").isoformat(),
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
