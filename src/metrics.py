from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    DEMAND_REFERENCE_WORKS,
    MIN_SPATIAL_DIAGNOSTIC_WORKS,
    MIN_SPATIAL_SAMPLE,
    RECENT_DEMAND_WINDOW_MONTHS,
    SPATIAL_RELATIVE_RISK_FULL,
    SPATIAL_RELATIVE_RISK_START,
    TIPOLOGIA_LABELS,
    TRIAGE_EVIDENCE_WEIGHTS,
    TRIAGE_RULE_VERSION,
)
from src.spatial import add_reach_status


HAVERSINE_MAX_PAIR_CELLS = 1_000_000
MODEL_REVISION_PATTERN = re.compile(
    r"^(?P<lineage>.+?_\d+)(?P<revision>[A-Z])?$", re.IGNORECASE
)


def canonical_model_name(name: object) -> str:
    value = str(name or "").strip()
    value = re.sub(r"\.dai$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\(\d+\)$", "", value)
    return value


def parse_model_revision(name: object) -> dict[str, object]:
    """Separa a linhagem do modelo e ordena seu sufixo alfabético de revisão."""

    canonical = canonical_model_name(name)
    match = MODEL_REVISION_PATTERN.fullmatch(canonical)
    if not match:
        return {
            "modelo_nome_canonico": canonical,
            "modelo_linhagem": canonical.upper(),
            "revisao_modelo": "",
            "ordem_revisao": 0,
        }

    revision = (match.group("revision") or "").upper()
    revision_order = 0
    for character in revision:
        revision_order = revision_order * 26 + (ord(character) - ord("A") + 1)
    return {
        "modelo_nome_canonico": canonical,
        "modelo_linhagem": match.group("lineage").upper(),
        "revisao_modelo": revision,
        "ordem_revisao": revision_order,
    }


def consolidate_latest_model_revisions(
    works: pd.DataFrame,
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mantém a revisão mais recente e agrega usos históricos na mesma linhagem.

    A revisão vencedora é definida pela união dos nomes presentes nas três fontes.
    Trabalhos são remapeados para a vencedora; catálogo e amostra antigos são
    descartados para não combinar metadados ou suporte espacial de versões distintas.
    """

    names: list[str] = []
    for frame in (works, catalog, samples):
        if not frame.empty and "modelo_nome" in frame:
            names.extend(frame["modelo_nome"].dropna().astype(str).tolist())

    audit_columns = [
        "modelo_linhagem",
        "modelo_mais_recente",
        "revisao_mais_recente",
        "versoes_encontradas",
        "n_versoes",
    ]
    if not names:
        return works.copy(), catalog.copy(), samples.copy(), pd.DataFrame(columns=audit_columns)

    revisions = pd.DataFrame([parse_model_revision(name) for name in names]).drop_duplicates(
        subset=["modelo_nome_canonico"]
    )
    revisions = revisions.sort_values(
        ["modelo_linhagem", "ordem_revisao", "modelo_nome_canonico"]
    )
    winners = revisions.groupby("modelo_linhagem", as_index=False).tail(1)
    winner_by_lineage = winners.set_index("modelo_linhagem")[
        "modelo_nome_canonico"
    ].to_dict()

    audit_records: list[dict[str, object]] = []
    for lineage, group in revisions.groupby("modelo_linhagem", sort=True):
        ordered_versions = group["modelo_nome_canonico"].tolist()
        winner = group.iloc[-1]
        audit_records.append(
            {
                "modelo_linhagem": lineage,
                "modelo_mais_recente": winner["modelo_nome_canonico"],
                "revisao_mais_recente": winner["revisao_modelo"] or "base",
                "versoes_encontradas": ", ".join(ordered_versions),
                "n_versoes": len(ordered_versions),
            }
        )

    def winner_for(name: object) -> str:
        parsed = parse_model_revision(name)
        return str(winner_by_lineage[parsed["modelo_linhagem"]])

    consolidated_works = works.copy()
    if not consolidated_works.empty and "modelo_nome" in consolidated_works:
        consolidated_works["modelo_nome_original"] = consolidated_works["modelo_nome"]
        consolidated_works["modelo_nome"] = consolidated_works["modelo_nome"].map(winner_for)
        duplicate_keys = [
            column
            for column in ("trabalho_id", "imovel_id", "modelo_nome")
            if column in consolidated_works
        ]
        if duplicate_keys:
            consolidated_works = consolidated_works.drop_duplicates(subset=duplicate_keys)

    def keep_latest(frame: pd.DataFrame, *, add_revision_columns: bool = False) -> pd.DataFrame:
        if frame.empty or "modelo_nome" not in frame:
            return frame.copy()
        result = frame.copy()
        parsed = pd.DataFrame(
            result["modelo_nome"].map(parse_model_revision).tolist(), index=result.index
        )
        is_latest = [
            canonical_model_name(name).casefold() == winner_for(name).casefold()
            for name in result["modelo_nome"]
        ]
        result = result.loc[is_latest].copy()
        result["modelo_nome"] = result["modelo_nome"].map(canonical_model_name)
        if add_revision_columns and not result.empty:
            result["modelo_linhagem"] = parsed.loc[result.index, "modelo_linhagem"]
            result["revisao_modelo"] = parsed.loc[result.index, "revisao_modelo"].replace(
                "", "base"
            )
        return result

    consolidated_catalog = keep_latest(catalog, add_revision_columns=True)
    consolidated_samples = keep_latest(samples)
    return (
        consolidated_works.reset_index(drop=True),
        consolidated_catalog.reset_index(drop=True),
        consolidated_samples.reset_index(drop=True),
        pd.DataFrame(audit_records, columns=audit_columns),
    )


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


def sample_nearest_neighbor_km(points: pd.DataFrame) -> np.ndarray:
    """Distância ao vizinho da própria amostra, excluindo apenas a mesma linha."""

    result = np.full(len(points), np.nan)
    if points.empty:
        return result
    coordinates = points[["latitude", "longitude"]].to_numpy(dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(coordinates).all(axis=1))
    if len(valid_positions) < 2:
        return result

    valid = coordinates[valid_positions]
    chunk_size = max(1, HAVERSINE_MAX_PAIR_CELLS // len(valid))
    lat2 = np.radians(valid[:, 0])[None, :]
    lon2 = np.radians(valid[:, 1])[None, :]
    for start in range(0, len(valid_positions), chunk_size):
        positions = valid_positions[start : start + chunk_size]
        current = coordinates[positions]
        lat1 = np.radians(current[:, 0])[:, None]
        lon1 = np.radians(current[:, 1])[:, None]
        value = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
        local_rows = np.arange(len(positions))
        own_columns = np.arange(start, start + len(positions))
        distance[local_rows, own_columns] = np.inf
        result[positions] = np.min(distance, axis=1)
    return result


def model_distance_summary(works: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "modelo_nome",
        "n_trabalhos_distancia",
        "n_amostra_espacial",
        "dist_mediana_km",
        "dist_p90_km",
        "dist_max_km",
        "amostra_nn_p90_km",
        "dist_relativa_p90",
        "fora_envoltoria_pct",
    ]
    if works.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for model_name, group in works.groupby("modelo_nome", dropna=False):
        model_samples = samples[samples["modelo_nome"] == model_name] if not samples.empty else pd.DataFrame()
        duplicate_keys = [
            column for column in ("trabalho_id", "imovel_id") if column in group
        ]
        valid_works = group.dropna(subset=["latitude", "longitude"]).drop_duplicates(
            subset=duplicate_keys or None
        )
        distances = haversine_nearest_km(valid_works, model_samples)
        finite = distances[np.isfinite(distances)]
        sample_neighbors = sample_nearest_neighbor_km(model_samples)
        finite_neighbors = sample_neighbors[np.isfinite(sample_neighbors)]
        sample_nn_p90 = (
            float(np.quantile(finite_neighbors, 0.90))
            if finite_neighbors.size
            else np.nan
        )
        work_p90 = float(np.quantile(finite, 0.90)) if finite.size else np.nan
        relative_p90 = (
            work_p90 / sample_nn_p90
            if np.isfinite(work_p90) and np.isfinite(sample_nn_p90) and sample_nn_p90 > 0
            else np.nan
        )
        classified = add_reach_status(valid_works, model_samples)[
            "dentro_alcance"
        ].dropna()
        records.append(
            {
                "modelo_nome": model_name,
                "n_trabalhos_distancia": int(len(valid_works)),
                "n_amostra_espacial": int(len(model_samples)),
                "dist_mediana_km": float(np.median(finite)) if finite.size else np.nan,
                "dist_p90_km": work_p90,
                "dist_max_km": float(np.max(finite)) if finite.size else np.nan,
                "amostra_nn_p90_km": sample_nn_p90,
                "dist_relativa_p90": relative_p90,
                "fora_envoltoria_pct": (
                    float((~classified.astype(bool)).mean() * 100)
                    if not classified.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _score_operational_impact(series: pd.Series) -> pd.Series:
    """Escala fixa: o resultado não depende dos demais modelos carregados."""

    denominator = np.log1p(float(DEMAND_REFERENCE_WORKS))
    return (np.log1p(series.astype(float)) / denominator).clip(lower=0, upper=1)


def select_recent_works(
    works: pd.DataFrame,
    *,
    today: date | None = None,
    months: int = RECENT_DEMAND_WINDOW_MONTHS,
) -> tuple[pd.DataFrame, str]:
    """Seleciona demanda em janela fixa, com fallback explícito quando só há ano."""

    if works.empty:
        return works.copy(), "sem trabalhos"
    reference = pd.Timestamp(today or date.today()).normalize()
    year_source = (
        works["ano"]
        if "ano" in works
        else pd.Series(np.nan, index=works.index, dtype=float)
    )
    years = pd.to_numeric(year_source, errors="coerce")
    fallback = years.isin([reference.year - 1, reference.year])
    if "data_trabalho" not in works:
        return works.loc[fallback].copy(), "aproximação por ano civil"

    parsed = pd.to_datetime(
        works["data_trabalho"],
        errors="coerce",
        format="mixed",
        utc=True,
        dayfirst=True,
    ).dt.tz_convert(None)
    valid_dates = parsed.notna()
    cutoff = reference - pd.DateOffset(months=months)
    by_date = valid_dates & parsed.between(cutoff, reference, inclusive="both")
    if valid_dates.all():
        return works.loc[by_date].copy(), f"janela móvel de {months} meses"
    if valid_dates.any():
        combined = by_date | (~valid_dates & fallback)
        return (
            works.loc[combined].copy(),
            f"janela móvel de {months} meses com fallback por ano",
        )
    return works.loc[fallback].copy(), "aproximação por ano civil"


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
    """Gera fila operacional em camadas, sem confundir ausência com desempenho ruim."""

    if works.empty:
        return pd.DataFrame()
    works, catalog, samples, _ = consolidate_latest_model_revisions(
        works, catalog, samples
    )
    reference_date = today or date.today()
    recent, demand_method = select_recent_works(works, today=reference_date)
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
    model_names = sorted(
        set(works.get("modelo_nome", []))
        | set(catalog.get("modelo_nome", []))
        | set(samples.get("modelo_nome", []))
    )
    result = pd.DataFrame({"modelo_nome": model_names})
    result = result.merge(total, on="modelo_nome", how="left")
    result = result.merge(demand, on="modelo_nome", how="left")
    for column in ("demanda_recente", "demanda_total"):
        result[column] = result[column].fillna(0).astype(int)

    catalog_columns = [column for column in ["modelo_nome", "data_final", "artifact_sha256", "r2_ajustado"] if column in catalog.columns]
    if catalog_columns:
        result = result.merge(catalog[catalog_columns], on="modelo_nome", how="left")
    if "data_final" not in result:
        result["data_final"] = pd.NaT
    result["no_catalogo"] = result["modelo_nome"].isin(set(catalog.get("modelo_nome", [])))

    sample_counts = (
        samples.groupby("modelo_nome")
        .size()
        .rename("n_amostra_espacial")
        .reset_index()
        if not samples.empty
        else pd.DataFrame(columns=["modelo_nome", "n_amostra_espacial"])
    )
    result = result.merge(sample_counts, on="modelo_nome", how="left")
    result["n_amostra_espacial"] = pd.to_numeric(
        result["n_amostra_espacial"], errors="coerce"
    ).fillna(0).astype(int)

    distances = model_distance_summary(recent, samples).drop(
        columns="n_amostra_espacial", errors="ignore"
    )
    result = result.merge(distances, on="modelo_nome", how="left")
    result["n_trabalhos_distancia"] = pd.to_numeric(
        result["n_trabalhos_distancia"], errors="coerce"
    ).fillna(0).astype(int)

    result = add_temporal_governance(result, today=reference_date)
    result["score_impacto"] = _score_operational_impact(result["demanda_recente"])
    result["score_demanda"] = result["score_impacto"]
    result["score_recencia"] = result["status_temporal"].map(
        {"Vigente": 0.0, "Alerta": 0.5, "Não utilizar": 1.0}
    )
    distance_risk = (
        (result["dist_relativa_p90"] - SPATIAL_RELATIVE_RISK_START)
        / (SPATIAL_RELATIVE_RISK_FULL - SPATIAL_RELATIVE_RISK_START)
    ).clip(lower=0, upper=1)
    hull_risk = (result["fora_envoltoria_pct"] / 100).clip(lower=0, upper=1)
    result["score_suporte"] = pd.concat(
        [distance_risk.rename("distancia"), hull_risk.rename("envoltoria")], axis=1
    ).mean(axis=1, skipna=True)
    result["score_suporte"] = pd.to_numeric(
        result["score_suporte"], errors="coerce"
    )
    no_recent_demand = result["demanda_recente"] == 0
    result.loc[no_recent_demand & result["score_suporte"].isna(), "score_suporte"] = 0.0
    result["suporte_aplicavel"] = ~no_recent_demand
    result["confianca_suporte"] = "Não aplicável"
    result.loc[
        result["suporte_aplicavel"] & (result["n_trabalhos_distancia"] == 0),
        "confianca_suporte",
    ] = "Não disponível"
    result.loc[
        result["suporte_aplicavel"]
        & result["n_trabalhos_distancia"].between(
            1, MIN_SPATIAL_DIAGNOSTIC_WORKS - 1
        ),
        "confianca_suporte",
    ] = "Exploratória"
    result.loc[
        result["n_trabalhos_distancia"] >= MIN_SPATIAL_DIAGNOSTIC_WORKS,
        "confianca_suporte",
    ] = "Suficiente"
    result["score_catalogo"] = (~result["no_catalogo"]).astype(float)

    def completeness_reasons(row: pd.Series) -> list[str]:
        reasons: list[str] = []
        if not bool(row["no_catalogo"]):
            reasons.append("catálogo da revisão atual ausente")
        if pd.isna(row["data_final"]):
            reasons.append("data contemporânea ausente")
        if row["demanda_recente"] > 0:
            if row["n_amostra_espacial"] < MIN_SPATIAL_SAMPLE:
                reasons.append("amostra espacial insuficiente")
            if row["n_trabalhos_distancia"] == 0:
                reasons.append("trabalhos recentes sem coordenadas válidas")
        return reasons

    reasons = result.apply(completeness_reasons, axis=1)
    result["motivo_completude"] = reasons.map(
        lambda values: "; ".join(values) if values else "insumos operacionais disponíveis"
    )
    result["status_completude"] = "Avaliável"
    result.loc[reasons.map(bool), "status_completude"] = "Incompleta"
    no_evidence = (
        ~result["no_catalogo"]
        & (result["n_amostra_espacial"] == 0)
    )
    result.loc[no_evidence, "status_completude"] = "Sem evidência"
    result["score_risco_operacional"] = result["status_completude"].map(
        {"Avaliável": 0.0, "Incompleta": 0.5, "Sem evidência": 1.0}
    )

    # As métricas existentes no .DAI não identificam inequivocamente uma base de teste.
    result["score_desempenho"] = np.nan
    result["status_validacao"] = "Não avaliado — base de teste ausente"
    components = {
        "impacto": result["score_impacto"],
        "desempenho": result["score_desempenho"],
        "suporte": result["score_suporte"].where(
            result["confianca_suporte"] == "Suficiente"
        ),
        "operacional": result["score_risco_operacional"],
    }
    weighted_sum = pd.Series(0.0, index=result.index)
    available_weight = pd.Series(0.0, index=result.index)
    for component, values in components.items():
        weight = TRIAGE_EVIDENCE_WEIGHTS[component]
        values = pd.to_numeric(values, errors="coerce")
        available = values.notna()
        weighted_sum = weighted_sum.add(values.fillna(0) * weight)
        available_weight = available_weight.add(available.astype(float) * weight)
    # Mantém a escala planejada de 100 pontos; evidência ausente não recebe risco técnico.
    result["score_auxiliar"] = 100 * weighted_sum
    result["cobertura_evidencias_pct"] = 100 * available_weight
    result["score_conclusivo"] = available_weight >= 1 - 1e-9
    result["status_escore"] = "Provisório — sem validação de desempenho"
    result["score_triagem"] = result["score_auxiliar"]

    high_spatial_risk = (
        (result["confianca_suporte"] == "Suficiente")
        & (result["score_suporte"].fillna(0) >= 2 / 3)
    )
    has_recent_demand = result["demanda_recente"] > 0
    result["prioridade_intervencao"] = "P3"
    result.loc[
        (result["status_temporal"] == "Não utilizar") & ~has_recent_demand,
        "prioridade_intervencao",
    ] = "P1"
    result.loc[
        (result["status_temporal"] == "Não utilizar") & has_recent_demand,
        "prioridade_intervencao",
    ] = "P0"
    result.loc[
        (result["status_temporal"] == "Alerta") & ~has_recent_demand,
        "prioridade_intervencao",
    ] = "P2"
    result.loc[
        (result["status_temporal"] == "Alerta") & has_recent_demand,
        "prioridade_intervencao",
    ] = "P1"
    result.loc[
        (result["status_temporal"] == "Vigente")
        & (
            high_spatial_risk
            | ((result["status_completude"] != "Avaliável") & has_recent_demand)
        ),
        "prioridade_intervencao",
    ] = "P2"

    def recommended_action(row: pd.Series) -> str:
        if row["prioridade_intervencao"] == "P0":
            return "Suspender o uso e atualizar ou substituir imediatamente"
        if row["status_temporal"] == "Não utilizar":
            return "Decidir entre atualização e aposentadoria antes de novo uso"
        if row["prioridade_intervencao"] == "P1":
            return "Planejar atualização antes do limite de 12 meses"
        if row["status_completude"] != "Avaliável":
            return "Completar evidências e revisar o suporte do modelo"
        if high_spatial_risk.at[row.name]:
            return "Revisar extrapolação espacial e domínio das preditoras"
        if row["status_temporal"] == "Alerta":
            return "Programar revisão e confirmar a necessidade de manutenção"
        return "Monitorar na rotina periódica"

    result["acao_recomendada"] = result.apply(recommended_action, axis=1)
    result["motivo_prioridade"] = result.apply(
        lambda row: (
            f"{row['status_temporal']}; {row['demanda_recente']} trabalho(s) na janela; "
            f"completude {str(row['status_completude']).lower()}"
        ),
        axis=1,
    )
    result["nivel_triagem"] = result["prioridade_intervencao"].map(
        {"P0": "Alta", "P1": "Alta", "P2": "Média", "P3": "Baixa"}
    )
    result["metodo_demanda"] = demand_method
    result["janela_demanda_meses"] = RECENT_DEMAND_WINDOW_MONTHS
    result["referencia_demanda_trabalhos"] = DEMAND_REFERENCE_WORKS
    result["regra_triagem_versao"] = TRIAGE_RULE_VERSION
    result["data_referencia_triagem"] = pd.Timestamp(reference_date)
    result["ordem_prioridade"] = result["prioridade_intervencao"].map(
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    )
    result = add_model_dimensions(result)
    return (
        result.sort_values(
            ["ordem_prioridade", "score_auxiliar", "demanda_recente"],
            ascending=[True, False, False],
        )
        .drop(columns="ordem_prioridade")
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

