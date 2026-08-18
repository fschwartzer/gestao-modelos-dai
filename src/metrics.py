from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from src.config import SCORE_WEIGHTS, TIPOLOGIA_LABELS


HAVERSINE_MAX_PAIR_CELLS = 1_000_000


def canonical_model_name(name: object) -> str:
    value = str(name or "").strip()
    value = re.sub(r"\.dai$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\(\d+\)$", "", value)
    return value


def parse_model_dimensions(name: object) -> dict[str, str]:
    value = canonical_model_name(name).upper()
    match = re.match(r"^MOD_([AV])_([A-Z]+)", value)
    if match:
        market_code, type_code = match.groups()
    else:
        market_code, type_code = "?", "OUTRO"
    market = {"A": "Aluguel", "V": "Venda"}.get(market_code, "Não classificado")
    type_label = TIPOLOGIA_LABELS.get(type_code, type_code.title())
    zones = sorted(set(re.findall(r"Z[1-5]", value)))
    family_code = f"{market_code}_{type_code}" if market_code != "?" else "OUTRO"
    family_label = f"{market} — {type_label}" if market_code != "?" else "Não classificado"
    return {
        "modelo_nome_canonico": canonical_model_name(name),
        "mercado_codigo": market_code,
        "mercado": market,
        "tipologia_codigo": type_code,
        "tipologia": type_label,
        "familia_codigo": family_code,
        "familia": family_label,
        "zonas_nome": ", ".join(zones) if zones else "Não informada",
    }


def add_model_dimensions(frame: pd.DataFrame, column: str = "modelo_nome") -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    dimensions = pd.DataFrame(result[column].map(parse_model_dimensions).tolist(), index=result.index)
    for name in dimensions.columns:
        result[name] = dimensions[name]
    return result


def haversine_nearest_km(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
) -> np.ndarray:
    if origins.empty or destinations.empty:
        return np.full(len(origins), np.nan)
    origin = origins[["latitude", "longitude"]].to_numpy(dtype=float)
    destination = destinations[["latitude", "longitude"]].to_numpy(dtype=float)
    valid_origins = np.isfinite(origin).all(axis=1)
    destination = destination[np.isfinite(destination).all(axis=1)]
    result = np.full(len(origin), np.nan)
    if not valid_origins.any() or not len(destination):
        return result

    valid_origin_positions = np.flatnonzero(valid_origins)
    chunk_size = max(1, HAVERSINE_MAX_PAIR_CELLS // len(destination))
    lat2 = np.radians(destination[:, 0])[None, :]
    lon2 = np.radians(destination[:, 1])[None, :]
    for start in range(0, len(valid_origin_positions), chunk_size):
        positions = valid_origin_positions[start : start + chunk_size]
        current = origin[positions]
        lat1 = np.radians(current[:, 0])[:, None]
        lon1 = np.radians(current[:, 1])[:, None]
        value = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
        result[positions] = np.min(distance, axis=1)
    return result


def model_distance_summary(works: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    if works.empty:
        return pd.DataFrame(columns=["modelo_nome", "dist_mediana_km", "dist_p90_km", "dist_max_km"])
    records: list[dict[str, object]] = []
    for model_name, group in works.groupby("modelo_nome", dropna=False):
        model_samples = samples[samples["modelo_nome"] == model_name] if not samples.empty else pd.DataFrame()
        valid_works = group.dropna(subset=["latitude", "longitude"]).drop_duplicates(
            subset=["trabalho_id", "imovel_id"]
        )
        distances = haversine_nearest_km(valid_works, model_samples)
        finite = distances[np.isfinite(distances)]
        records.append(
            {
                "modelo_nome": model_name,
                "n_trabalhos_distancia": int(len(valid_works)),
                "n_amostra_espacial": int(len(model_samples)),
                "dist_mediana_km": float(np.median(finite)) if finite.size else np.nan,
                "dist_p90_km": float(np.quantile(finite, 0.90)) if finite.size else np.nan,
                "dist_max_km": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(records)


def _normalize_demand(series: pd.Series) -> pd.Series:
    values = np.log1p(series.astype(float))
    maximum = values.max()
    return values / maximum if maximum > 0 else values * 0


def add_temporal_governance(
    frame: pd.DataFrame,
    *,
    today: date | None = None,
    date_column: str = "data_final",
) -> pd.DataFrame:
    """Aplica as regras operacionais de contemporaneidade do modelo.

    Até 6 meses completos: vigente. Acima de 6 e até 12 meses: alerta.
    Acima de 12 meses, ou sem data verificável: não utilizar.
    """

    result = frame.copy()
    reference = pd.Timestamp(today or date.today()).normalize()
    if date_column in result:
        parsed = pd.to_datetime(result[date_column], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        parsed = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    parsed = parsed.dt.normalize()

    alert_cutoff = reference - pd.DateOffset(months=6)
    stop_cutoff = reference - pd.DateOffset(months=12)
    result["status_temporal"] = "Vigente"
    result["motivo_temporal"] = "Dado mais contemporâneo dentro do limite de 6 meses"

    alert_mask = parsed.notna() & (parsed < alert_cutoff) & (parsed >= stop_cutoff)
    stop_mask = parsed.notna() & (parsed < stop_cutoff)
    missing_mask = parsed.isna()
    result.loc[alert_mask, "status_temporal"] = "Alerta"
    result.loc[alert_mask, "motivo_temporal"] = (
        "Dado mais contemporâneo excede 6 meses, sem ultrapassar 12 meses"
    )
    result.loc[stop_mask, "status_temporal"] = "Não utilizar"
    result.loc[stop_mask, "motivo_temporal"] = "Dado mais contemporâneo excede 12 meses"
    result.loc[missing_mask, "status_temporal"] = "Não utilizar"
    result.loc[missing_mask, "motivo_temporal"] = "Data do dado mais contemporâneo ausente"

    month_age = (
        (reference.year - parsed.dt.year) * 12
        + (reference.month - parsed.dt.month)
        - (parsed.dt.day > reference.day).astype("Int64")
    )
    result["idade_dado_meses"] = month_age.clip(lower=0).astype("Int64")
    result["apto_para_uso"] = result["status_temporal"] != "Não utilizar"
    return result


def build_priority_table(
    works: pd.DataFrame,
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    today: date | None = None,
) -> pd.DataFrame:
    """Gera triagem operacional, não um juízo automático sobre validade técnica."""

    if works.empty:
        return pd.DataFrame()
    reference_date = today or date.today()
    max_year = int(pd.to_numeric(works["ano"], errors="coerce").max())
    recent = works[pd.to_numeric(works["ano"], errors="coerce") >= max_year - 1]
    demand = (
        recent.groupby("modelo_nome")["trabalho_id"]
        .nunique()
        .rename("demanda_recente")
        .reset_index()
    )
    total = (
        works.groupby("modelo_nome")["trabalho_id"]
        .nunique()
        .rename("demanda_total")
        .reset_index()
    )
    result = total.merge(demand, on="modelo_nome", how="left")
    result["demanda_recente"] = result["demanda_recente"].fillna(0).astype(int)

    catalog_columns = [column for column in ["modelo_nome", "data_final", "artifact_sha256", "r2_ajustado"] if column in catalog.columns]
    if catalog_columns:
        result = result.merge(catalog[catalog_columns], on="modelo_nome", how="left")
    if "data_final" not in result:
        result["data_final"] = pd.NaT
    result["no_catalogo"] = result["modelo_nome"].isin(set(catalog.get("modelo_nome", [])))

    distances = model_distance_summary(works, samples)
    result = result.merge(distances, on="modelo_nome", how="left")

    result = add_temporal_governance(result, today=reference_date)
    result["score_demanda"] = _normalize_demand(result["demanda_recente"])
    result["score_recencia"] = result["status_temporal"].map(
        {"Vigente": 0.0, "Alerta": 0.5, "Não utilizar": 1.0}
    )
    result["score_suporte"] = (result["dist_p90_km"] / 5.0).clip(upper=1).fillna(1.0)
    result["score_catalogo"] = (~result["no_catalogo"]).astype(float)
    result["score_triagem"] = 100 * (
        SCORE_WEIGHTS["demanda"] * result["score_demanda"]
        + SCORE_WEIGHTS["recencia"] * result["score_recencia"]
        + SCORE_WEIGHTS["suporte"] * result["score_suporte"]
        + SCORE_WEIGHTS["catalogo"] * result["score_catalogo"]
    )
    result["nivel_triagem"] = pd.cut(
        result["score_triagem"],
        bins=[-np.inf, 40, 65, np.inf],
        labels=["Baixa", "Média", "Alta"],
    ).astype(str)
    result.loc[
        (result["status_temporal"] == "Alerta") & (result["nivel_triagem"] == "Baixa"),
        "nivel_triagem",
    ] = "Média"
    result.loc[result["status_temporal"] == "Não utilizar", "nivel_triagem"] = "Alta"
    result["ordem_temporal"] = result["status_temporal"].map(
        {"Não utilizar": 2, "Alerta": 1, "Vigente": 0}
    )
    result = add_model_dimensions(result)
    return (
        result.sort_values(
            ["ordem_temporal", "score_triagem", "demanda_recente"], ascending=False
        )
        .drop(columns="ordem_temporal")
        .reset_index(drop=True)
    )


def distance_bins(distances: pd.Series) -> pd.DataFrame:
    bins = [-np.inf, 0.5, 1, 2, 5, np.inf]
    labels = ["Até 0,5 km", "0,5–1 km", "1–2 km", "2–5 km", "Acima de 5 km"]
    categories = pd.cut(distances, bins=bins, labels=labels)
    return (
        categories.value_counts(sort=False)
        .rename_axis("faixa")
        .reset_index(name="trabalhos")
    )

