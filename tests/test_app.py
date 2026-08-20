from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import DEMO_DB
from src.data import load_demo_data
from src.metrics import add_model_dimensions, add_temporal_governance
from tests.test_dai import synthetic_dai_bytes


def test_app_reloads_stale_release_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = (
        "import src.config as config; "
        "import src.metrics as metrics; "
        "config.TRIAGE_RULE_VERSION = '1.0'; "
        "metrics.TRIAGE_RULE_VERSION = '1.0'; "
        "stale = lambda *args, **kwargs: 'stale'; "
        "metrics.build_priority_table = stale; "
        "import app; "
        "assert app.config_module.TRIAGE_RULE_VERSION == '2.0'; "
        "assert app.metrics_module.TRIAGE_RULE_VERSION == '2.0'; "
        "assert callable(app.consolidate_latest_model_revisions); "
        "assert app.build_priority_table is not stale"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_app_starts_in_demo_mode() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Gestão geoespacial de modelos de avaliação"

    expected_titles = {
        "Modelos": "Catálogo de modelos",
        "Cobertura": "Cobertura e suporte espacial",
        "Prioridades": "Fila de intervenção e governança",
        "Metodologia": "Metodologia e segurança",
    }
    for page, title in expected_titles.items():
        app.sidebar.radio[1].set_value(page).run()
        assert not app.exception
        assert app.title[0].value == title


def test_upload_mode_exposes_sqlite_and_dai_inputs() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[0].set_value("Enviar arquivos nesta sessão").run()

    assert not app.exception
    assert [uploader.label for uploader in app.sidebar.file_uploader] == [
        "Trabalhos técnicos (SQLite)",
        "Modelos (.DAI)",
        "Catálogo (.csv)",
        "Amostras (.csv ou .csv.gz)",
    ]
    assert any("ao menos um" in message.value for message in app.info)


def test_sqlite_only_enables_work_analyses() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[0].set_value("Enviar arquivos nesta sessão").run()
    app.sidebar.file_uploader[0].upload(
        "trabalhos.sqlite3", Path(DEMO_DB).read_bytes()
    ).run()

    assert not app.exception
    assert app.title[0].value == "Gestão geoespacial de modelos de avaliação"
    assert list(app.sidebar.radio[1].options) == ["Visão geral", "Metodologia"]


def test_dai_only_enables_model_catalog() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[0].set_value("Enviar arquivos nesta sessão").run()
    app.sidebar.file_uploader[1].upload(
        "MOD_V_TER_Z1_001.dai", synthetic_dai_bytes()
    ).run()
    app.sidebar.checkbox[0].check().run()

    assert not app.exception
    assert app.title[0].value == "Catálogo de modelos"
    assert list(app.sidebar.radio[1].options) == ["Modelos", "Metodologia"]
    assert any("uso histórico" in message.value for message in app.info)


def test_coverage_defaults_to_all_current_and_alerted_models() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[1].set_value("Cobertura").run()

    _, catalog, samples = load_demo_data()
    governed = add_temporal_governance(catalog, today=date.today())
    expected = (
        set(
            governed.loc[
                governed["status_temporal"].isin(["Vigente", "Alerta"]),
                "modelo_nome",
            ]
        )
        & set(samples["modelo_nome"])
    )
    selector = next(
        item for item in app.multiselect if item.label == "Modelos sobrepostos"
    )
    assert not app.exception
    assert set(selector.value) == expected


def test_coverage_model_options_follow_family_filter() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[1].set_value("Cobertura").run()

    _, catalog, samples = load_demo_data()
    catalog = add_model_dimensions(catalog)
    target_family = "Venda — Terreno"
    expected_options = sorted(
        set(
            catalog.loc[catalog["familia"] == target_family, "modelo_nome"]
        )
        & set(samples["modelo_nome"])
    )
    family_filter = next(
        item for item in app.sidebar.multiselect if item.label == "Famílias"
    )
    family_filter.set_value([target_family]).run()
    selector = next(
        item for item in app.multiselect if item.label == "Modelos sobrepostos"
    )

    assert not app.exception
    assert list(selector.options) == expected_options
    assert all(
        catalog.set_index("modelo_nome").loc[model_name, "familia"] == target_family
        for model_name in selector.value
    )


def test_model_catalog_follows_sidebar_family_filter() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    app.sidebar.radio[1].set_value("Modelos").run()
    target_family = "Venda — Terreno"
    family_filter = next(
        item for item in app.sidebar.multiselect if item.label == "Famílias"
    )
    family_filter.set_value([target_family]).run()

    displayed = app.dataframe[-1].value
    assert not app.exception
    assert set(displayed["familia"]) == {target_family}


def test_priority_queue_follows_all_sidebar_filters() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    app.sidebar.radio[1].set_value("Prioridades").run()
    works, _, _ = load_demo_data()
    enriched = add_model_dimensions(works)
    target_family = "Venda — Terreno"
    target_year = int(
        enriched.loc[enriched["familia"] == target_family, "ano"].max()
    )
    target_type = (
        enriched.loc[
            (enriched["familia"] == target_family)
            & (enriched["ano"] == target_year),
            "tipo_label",
        ]
        .value_counts()
        .index[0]
    )

    family_filter = next(
        item for item in app.sidebar.multiselect if item.label == "Famílias"
    )
    family_filter.set_value([target_family]).run()
    year_filter = next(
        item for item in app.sidebar.multiselect if item.label == "Anos"
    )
    year_filter.set_value([target_year]).run()
    type_filter = next(
        item for item in app.sidebar.multiselect if item.label == "Tipos de trabalho"
    )
    type_filter.set_value([target_type]).run()

    displayed = app.dataframe[-1].value
    expected_demand = (
        enriched.loc[
            (enriched["familia"] == target_family)
            & (enriched["ano"] == target_year)
            & (enriched["tipo_label"] == target_type)
        ]
        .groupby("modelo_nome")["trabalho_id"]
        .nunique()
    )

    assert not app.exception
    assert set(displayed["familia"]) == {target_family}
    assert displayed.set_index("modelo_nome")["demanda_total"].to_dict() == {
        model_name: int(expected_demand.get(model_name, 0))
        for model_name in displayed["modelo_nome"]
    }
