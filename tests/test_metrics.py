from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data import load_demo_data
from src.metrics import (
    add_temporal_governance,
    build_priority_table,
    consolidate_latest_model_revisions,
    haversine_nearest_km,
    parse_model_dimensions,
    parse_model_revision,
    sample_nearest_neighbor_km,
    select_recent_works,
)


def test_parse_model_dimensions() -> None:
    parsed = parse_model_dimensions("MOD_V_TER_Z2_008C.dai")
    assert parsed["mercado"] == "Venda"
    assert parsed["tipologia"] == "Terreno"
    assert parsed["zonas_nome"] == "Z2"


def test_parse_model_revision_orders_alphabetic_suffixes() -> None:
    older = parse_model_revision("MOD_V_TER_Z1_006I")
    newer = parse_model_revision("MOD_V_TER_Z1_006J.dai")
    assert older["modelo_linhagem"] == newer["modelo_linhagem"]
    assert older["ordem_revisao"] < newer["ordem_revisao"]


def test_haversine_zero_distance() -> None:
    point = pd.DataFrame({"latitude": [-30.03], "longitude": [-51.20]})
    distance = haversine_nearest_km(point, point)
    assert np.isclose(distance[0], 0.0)


def test_haversine_ignores_invalid_coordinates_without_losing_alignment() -> None:
    origins = pd.DataFrame(
        {"latitude": [-30.03, np.nan], "longitude": [-51.20, -51.20]}
    )
    destinations = pd.DataFrame(
        {"latitude": [np.nan, -30.03], "longitude": [-51.20, -51.20]}
    )
    distance = haversine_nearest_km(origins, destinations)
    assert np.isclose(distance[0], 0.0)
    assert np.isnan(distance[1])


def test_sample_nearest_neighbor_excludes_the_same_row() -> None:
    points = pd.DataFrame(
        {
            "latitude": [-30.03, -30.03],
            "longitude": [-51.20, -51.19],
        }
    )
    distances = sample_nearest_neighbor_km(points)
    assert np.isfinite(distances).all()
    assert (distances > 0).all()


def test_recent_window_prefers_complete_work_date() -> None:
    works = pd.DataFrame(
        {
            "trabalho_id": ["fora", "limite", "recente"],
            "ano": [2025, 2025, 2026],
            "data_trabalho": ["2025-08-18", "2025-08-19", "2026-08-01"],
        }
    )
    recent, method = select_recent_works(works, today=date(2026, 8, 19))
    assert recent["trabalho_id"].tolist() == ["limite", "recente"]
    assert method == "janela móvel de 12 meses"


def test_priority_table_is_bounded() -> None:
    works, catalog, samples = load_demo_data()
    table = build_priority_table(works, catalog, samples, today=date(2026, 8, 13))
    assert not table.empty
    assert table["score_triagem"].between(0, 100).all()
    assert set(table["nivel_triagem"]).issubset({"Baixa", "Média", "Alta"})


def test_temporal_governance_respects_six_and_twelve_month_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "data_final": [
                "2026-02-18",  # exatamente 6 meses
                "2026-02-17",  # acima de 6 meses
                "2025-08-18",  # exatamente 12 meses
                "2025-08-17",  # acima de 12 meses
                None,
            ]
        }
    )
    governed = add_temporal_governance(frame, today=date(2026, 8, 18))
    assert governed["status_temporal"].tolist() == [
        "Vigente",
        "Alerta",
        "Alerta",
        "Não utilizar",
        "Não utilizar",
    ]
    assert governed["apto_para_uso"].tolist() == [True, True, True, False, False]


def test_temporal_rule_overrides_qualitative_priority_and_order() -> None:
    works, catalog, samples = load_demo_data()
    catalog = catalog.copy()
    catalog["data_final"] = "2026-08-13"
    blocked_model = catalog.iloc[0]["modelo_nome"]
    alerted_model = catalog.iloc[1]["modelo_nome"]
    catalog.loc[catalog["modelo_nome"] == blocked_model, "data_final"] = "2025-08-12"
    catalog.loc[catalog["modelo_nome"] == alerted_model, "data_final"] = "2026-02-12"

    table = build_priority_table(works, catalog, samples, today=date(2026, 8, 13))
    blocked = table[table["modelo_nome"] == blocked_model].iloc[0]
    alerted = table[table["modelo_nome"] == alerted_model].iloc[0]
    assert table.iloc[0]["modelo_nome"] == blocked_model
    assert blocked["status_temporal"] == "Não utilizar"
    assert blocked["prioridade_intervencao"] == "P0"
    assert blocked["nivel_triagem"] == "Alta"
    assert alerted["status_temporal"] == "Alerta"
    assert alerted["nivel_triagem"] in {"Média", "Alta"}


def test_latest_revision_consolidates_works_and_discards_old_samples() -> None:
    works = pd.DataFrame(
        {
            "trabalho_id": ["T1", "T1", "T2", "T3"],
            "imovel_id": [1, 1, 2, 3],
            "modelo_nome": [
                "MOD_V_TER_Z1_006I",
                "MOD_V_TER_Z1_006J",
                "MOD_V_TER_Z1_006I",
                "MOD_V_TER_Z2_007A",
            ],
        }
    )
    catalog = pd.DataFrame(
        {
            "modelo_nome": [
                "MOD_V_TER_Z1_006I",
                "MOD_V_TER_Z1_006J",
                "MOD_V_TER_Z2_007A",
            ]
        }
    )
    samples = pd.DataFrame(
        {
            "modelo_nome": [
                "MOD_V_TER_Z1_006I",
                "MOD_V_TER_Z1_006I",
                "MOD_V_TER_Z1_006J",
                "MOD_V_TER_Z2_007A",
            ],
            "latitude": [-30.03, -30.04, -30.02, -30.01],
            "longitude": [-51.20, -51.21, -51.19, -51.18],
        }
    )

    consolidated_works, consolidated_catalog, consolidated_samples, audit = (
        consolidate_latest_model_revisions(works, catalog, samples)
    )
    assert set(consolidated_works["modelo_nome"]) == {
        "MOD_V_TER_Z1_006J",
        "MOD_V_TER_Z2_007A",
    }
    assert len(
        consolidated_works[
            consolidated_works["modelo_nome"] == "MOD_V_TER_Z1_006J"
        ]
    ) == 2
    assert set(consolidated_catalog["modelo_nome"]) == {
        "MOD_V_TER_Z1_006J",
        "MOD_V_TER_Z2_007A",
    }
    assert "MOD_V_TER_Z1_006I" not in set(consolidated_samples["modelo_nome"])
    lineage = audit[audit["modelo_linhagem"] == "MOD_V_TER_Z1_006"].iloc[0]
    assert lineage["modelo_mais_recente"] == "MOD_V_TER_Z1_006J"
    assert lineage["n_versoes"] == 2


def test_newer_work_revision_does_not_inherit_old_catalog_metadata() -> None:
    works = pd.DataFrame(
        {"trabalho_id": ["T1"], "modelo_nome": ["MOD_V_TER_Z1_006J"]}
    )
    catalog = pd.DataFrame({"modelo_nome": ["MOD_V_TER_Z1_006I"]})
    samples = pd.DataFrame(
        {
            "modelo_nome": ["MOD_V_TER_Z1_006I"],
            "latitude": [-30.03],
            "longitude": [-51.20],
        }
    )
    _, consolidated_catalog, consolidated_samples, _ = consolidate_latest_model_revisions(
        works, catalog, samples
    )
    assert consolidated_catalog.empty
    assert consolidated_samples.empty


def test_priority_table_counts_lineage_only_once_at_latest_revision() -> None:
    works, catalog, samples = load_demo_data()
    source_model = works.iloc[0]["modelo_nome"]
    lineage = parse_model_revision(source_model)["modelo_linhagem"]
    older = f"{lineage}I"
    newer = f"{lineage}J"

    works = works.copy()
    catalog = catalog.copy()
    samples = samples.copy()
    works.loc[works["modelo_nome"] == source_model, "modelo_nome"] = older
    catalog.loc[catalog["modelo_nome"] == source_model, "modelo_nome"] = newer
    samples.loc[samples["modelo_nome"] == source_model, "modelo_nome"] = newer

    table = build_priority_table(works, catalog, samples, today=date(2026, 8, 18))
    lineage_rows = table[
        table["modelo_nome"].map(
            lambda name: parse_model_revision(name)["modelo_linhagem"] == lineage
        )
    ]
    assert lineage_rows["modelo_nome"].tolist() == [newer]


def test_demand_score_does_not_depend_on_other_loaded_models() -> None:
    works, catalog, samples = load_demo_data()
    complete = build_priority_table(works, catalog, samples, today=date(2026, 8, 19))
    target = complete.sort_values("demanda_recente").iloc[0]["modelo_nome"]
    isolated = build_priority_table(
        works[works["modelo_nome"] == target],
        catalog,
        samples,
        today=date(2026, 8, 19),
    )
    complete_score = complete.loc[
        complete["modelo_nome"] == target, "score_impacto"
    ].iloc[0]
    isolated_score = isolated.loc[
        isolated["modelo_nome"] == target, "score_impacto"
    ].iloc[0]
    assert np.isclose(complete_score, isolated_score)


def test_old_filtered_years_are_not_reclassified_as_recent() -> None:
    works, catalog, samples = load_demo_data()
    old_works = works[pd.to_numeric(works["ano"], errors="coerce") <= 2020]
    table = build_priority_table(old_works, catalog, samples, today=date(2026, 8, 19))
    assert table["demanda_recente"].sum() == 0
    assert set(table["metodo_demanda"]) == {"aproximação por ano civil"}


def test_missing_spatial_evidence_is_not_maximum_technical_risk() -> None:
    works = pd.DataFrame(
        {
            "trabalho_id": ["T1"],
            "imovel_id": [1],
            "modelo_nome": ["MOD_V_TER_Z1_001A"],
            "ano": [2026],
            "latitude": [-30.03],
            "longitude": [-51.20],
        }
    )
    catalog = pd.DataFrame(
        {
            "modelo_nome": ["MOD_V_TER_Z1_001A"],
            "data_final": ["2026-08-01"],
        }
    )
    table = build_priority_table(works, catalog, pd.DataFrame(), today=date(2026, 8, 19))
    record = table.iloc[0]
    assert pd.isna(record["score_suporte"])
    assert record["status_completude"] == "Incompleta"
    assert record["prioridade_intervencao"] == "P2"
    assert record["cobertura_evidencias_pct"] == 50


def test_catalog_only_expired_model_is_included_for_retirement_decision() -> None:
    works = pd.DataFrame(
        {
            "trabalho_id": ["T1"],
            "imovel_id": [1],
            "modelo_nome": ["MOD_V_TER_Z1_001A"],
            "ano": [2026],
            "latitude": [-30.03],
            "longitude": [-51.20],
        }
    )
    catalog = pd.DataFrame(
        {
            "modelo_nome": ["MOD_V_TER_Z1_001A", "MOD_V_TER_Z2_002A"],
            "data_final": ["2026-08-01", "2024-01-01"],
        }
    )
    table = build_priority_table(works, catalog, pd.DataFrame(), today=date(2026, 8, 19))
    unused = table[table["modelo_nome"] == "MOD_V_TER_Z2_002A"].iloc[0]
    assert unused["demanda_recente"] == 0
    assert unused["prioridade_intervencao"] == "P1"
    assert "aposentadoria" in unused["acao_recomendada"]


def test_score_remains_provisional_without_identified_test_metrics() -> None:
    works, catalog, samples = load_demo_data()
    table = build_priority_table(works, catalog, samples, today=date(2026, 8, 19))
    assert table["score_desempenho"].isna().all()
    assert (~table["score_conclusivo"]).all()
    assert table["cobertura_evidencias_pct"].le(65).all()


def test_more_recent_demand_increases_score_within_the_same_class() -> None:
    model_a = "MOD_V_TER_Z1_001A"
    model_b = "MOD_V_TER_Z2_002A"
    works = pd.DataFrame(
        {
            "trabalho_id": ["A1", "B1", "B2", "B3", "B4", "B5"],
            "imovel_id": range(1, 7),
            "modelo_nome": [model_a] + [model_b] * 5,
            "ano": [2026] * 6,
            "latitude": [np.nan] * 6,
            "longitude": [np.nan] * 6,
        }
    )
    catalog = pd.DataFrame(
        {
            "modelo_nome": [model_a, model_b],
            "data_final": ["2026-08-01", "2026-08-01"],
        }
    )
    table = build_priority_table(works, catalog, pd.DataFrame(), today=date(2026, 8, 19))
    a = table[table["modelo_nome"] == model_a].iloc[0]
    b = table[table["modelo_nome"] == model_b].iloc[0]
    assert a["prioridade_intervencao"] == b["prioridade_intervencao"] == "P2"
    assert b["score_impacto"] > a["score_impacto"]
    assert b["score_auxiliar"] > a["score_auxiliar"]


def test_sparse_spatial_diagnostic_does_not_escalate_priority() -> None:
    model = "MOD_V_TER_Z1_001A"
    works = pd.DataFrame(
        {
            "trabalho_id": ["T1"],
            "imovel_id": [1],
            "modelo_nome": [model],
            "ano": [2026],
            "latitude": [-30.30],
            "longitude": [-51.38],
        }
    )
    catalog = pd.DataFrame(
        {"modelo_nome": [model], "data_final": ["2026-08-01"]}
    )
    samples = pd.DataFrame(
        {
            "modelo_nome": [model] * 3,
            "latitude": [-30.03, -30.031, -30.029],
            "longitude": [-51.20, -51.201, -51.202],
        }
    )
    record = build_priority_table(
        works, catalog, samples, today=date(2026, 8, 19)
    ).iloc[0]
    assert record["score_suporte"] >= 2 / 3
    assert record["confianca_suporte"] == "Exploratória"
    assert record["prioridade_intervencao"] == "P3"

