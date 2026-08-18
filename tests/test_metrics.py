from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data import load_demo_data
from src.metrics import (
    add_temporal_governance,
    build_priority_table,
    haversine_nearest_km,
    parse_model_dimensions,
)


def test_parse_model_dimensions() -> None:
    parsed = parse_model_dimensions("MOD_V_TER_Z2_008C.dai")
    assert parsed["mercado"] == "Venda"
    assert parsed["tipologia"] == "Terreno"
    assert parsed["zonas_nome"] == "Z2"


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
    assert blocked["nivel_triagem"] == "Alta"
    assert alerted["status_temporal"] == "Alerta"
    assert alerted["nivel_triagem"] in {"Média", "Alta"}

