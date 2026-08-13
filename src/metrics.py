from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from src.config import SCORE_WEIGHTS, TIPOLOGIA_LABELS


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
    lat1 = np.radians(origin[:, 0])[:, None]
    lon1 = np.radians(origin[:, 1])[:, None]
    lat2 = np.radians(destination[:, 0])[None, :]
    lon2 = np.radians(destination[:, 1])[None, :]
    value = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    return np.nanmin(distance, axis=1)


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


def _staleness_score(dates: pd.Series, today: date) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    reference = pd.Timestamp(today)
    months = (reference.year - parsed.dt.year) * 12 + (reference.month - parsed.dt.month)
    score = (months.clip(lower=0) / 36).clip(upper=1)
    return score.fillna(1.0)


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

    result["score_demanda"] = _normalize_demand(result["demanda_recente"])
    result["score_recencia"] = _staleness_score(result["data_final"], reference_date)
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
    result = add_model_dimensions(result)
    return result.sort_values(["score_triagem", "demanda_recente"], ascending=False)


def distance_bins(distances: pd.Series) -> pd.DataFrame:
    bins = [-np.inf, 0.5, 1, 2, 5, np.inf]
    labels = ["Até 0,5 km", "0,5–1 km", "1–2 km", "2–5 km", "Acima de 5 km"]
    categories = pd.cut(distances, bins=bins, labels=labels)
    return (
        categories.value_counts(sort=False)
        .rename_axis("faixa")
        .reset_index(name="trabalhos")
    )

