from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


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
    ]
