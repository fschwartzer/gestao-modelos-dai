from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import DEMO_DB
from tests.test_dai import synthetic_dai_bytes


def test_app_starts_in_demo_mode() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Gestão geoespacial de modelos de avaliação"

    expected_titles = {
        "Modelos": "Catálogo de modelos",
        "Cobertura": "Cobertura e suporte espacial",
        "Prioridades": "Triagem para atualização e auditoria",
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
