from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data import load_demo_data
from src.metrics import (
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


def test_priority_table_is_bounded() -> None:
    works, catalog, samples = load_demo_data()
    table = build_priority_table(works, catalog, samples, today=date(2026, 8, 13))
    assert not table.empty
    assert table["score_triagem"].between(0, 100).all()
    assert set(table["nivel_triagem"]).issubset({"Baixa", "Média", "Alta"})

