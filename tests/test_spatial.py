from __future__ import annotations

import pandas as pd

from src.charts import coverage_map
from src.spatial import add_reach_status, empirical_reach_polygon


def sample_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modelo_nome": ["MODELO_A"] * 3,
            "latitude": [-30.10, -30.10, -30.00],
            "longitude": [-51.20, -51.10, -51.15],
        }
    )


def test_empirical_reach_requires_three_non_collinear_points() -> None:
    assert empirical_reach_polygon(sample_points()) is not None
    assert empirical_reach_polygon(sample_points().iloc[:2]) is None


def test_work_points_are_classified_against_matching_model_reach() -> None:
    works = pd.DataFrame(
        {
            "modelo_nome": ["MODELO_A", "MODELO_A"],
            "latitude": [-30.05, -30.05],
            "longitude": [-51.15, -51.25],
        }
    )
    classified = add_reach_status(works, sample_points())
    assert classified["dentro_alcance"].tolist() == [True, False]
    assert classified["status_alcance"].tolist() == [
        "Dentro da envoltória",
        "Fora da envoltória",
    ]


def test_coverage_map_contains_empirical_polygon() -> None:
    works = pd.DataFrame(
        {
            "modelo_nome": ["MODELO_A"],
            "nome": ["TRABALHO_1"],
            "ano": [2026],
            "latitude": [-30.05],
            "longitude": [-51.15],
            "distancia_km": [0.4],
            "status_alcance": ["Dentro da envoltória"],
        }
    )
    figure = coverage_map(works, sample_points())
    assert any(trace.fill == "toself" for trace in figure.data)
