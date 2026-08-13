from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    PLOTLY_CONFIG,
    annual_stacked_bar,
    coverage_map,
    demand_map,
    distance_bar,
    horizontal_bar,
    priority_map,
)
from src.config import PRIVATE_DIR
from src.data import (
    load_csv_source,
    load_demo_data,
    load_sqlite_bytes,
    load_sqlite_path,
    standardize_catalog,
    standardize_samples,
    unique_work_points,
)
from src.metrics import (
    add_model_dimensions,
    build_priority_table,
    distance_bins,
    haversine_nearest_km,
)


st.set_page_config(
    page_title="Gestão de Modelos DAI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_demo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_demo_data()


@st.cache_data(show_spinner=False)
def cached_uploaded_sqlite(raw: bytes) -> pd.DataFrame:
    return load_sqlite_bytes(raw)


@st.cache_data(show_spinner=False)
def cached_uploaded_csv(raw: bytes) -> pd.DataFrame:
    return load_csv_source(raw)


@st.cache_data(show_spinner=False)
def cached_local_data(db_path: str, catalog_path: str, samples_path: str):
    works = load_sqlite_path(db_path)
    catalog = standardize_catalog(load_csv_source(catalog_path))
    samples = standardize_samples(load_csv_source(samples_path))
    return works, catalog, samples


def load_selected_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    local_db = PRIVATE_DIR / "trabalhos_tecnicos.sqlite3"
    local_catalog = PRIVATE_DIR / "catalogo_modelos.csv"
    local_samples = PRIVATE_DIR / "amostras_modelos.csv.gz"
    local_available = local_db.exists()

    options = ["Demonstração"]
    if local_available:
        options.append("Arquivos locais protegidos")
    options.append("Enviar arquivos nesta sessão")
    mode = st.sidebar.radio("Fonte dos dados", options, index=0)

    if mode == "Demonstração":
        works, catalog, samples = cached_demo()
        return works, catalog, samples, "Dados sintéticos de demonstração"

    if mode == "Arquivos locais protegidos":
        works, catalog, samples = cached_local_data(
            str(local_db), str(local_catalog), str(local_samples)
        )
        return works, catalog, samples, "Arquivos locais protegidos"

    st.sidebar.caption("Os arquivos enviados são processados apenas durante a sessão ativa.")
    db_file = st.sidebar.file_uploader("Banco de trabalhos (.sqlite3)", type=["sqlite3", "db"])
    catalog_file = st.sidebar.file_uploader("Catálogo dos modelos (.csv)", type=["csv"])
    samples_file = st.sidebar.file_uploader(
        "Amostras espaciais (.csv ou .csv.gz)", type=["csv", "gz"]
    )
    if db_file is None:
        st.info("Envie o banco SQLite na barra lateral para iniciar a análise.")
        st.stop()
    works = cached_uploaded_sqlite(db_file.getvalue())
    catalog = (
        standardize_catalog(cached_uploaded_csv(catalog_file.getvalue()))
        if catalog_file is not None
        else pd.DataFrame()
    )
    samples = (
        standardize_samples(cached_uploaded_csv(samples_file.getvalue()))
        if samples_file is not None
        else pd.DataFrame()
    )
    return works, catalog, samples, "Arquivos enviados na sessão"


def global_filters(works: pd.DataFrame) -> pd.DataFrame:
    enriched = add_model_dimensions(works)
    st.sidebar.divider()
    st.sidebar.subheader("Filtros")
    years = sorted(pd.to_numeric(enriched["ano"], errors="coerce").dropna().astype(int).unique())
    selected_years = st.sidebar.multiselect("Anos", years, default=years)
    types = sorted(enriched["tipo_label"].dropna().unique())
    selected_types = st.sidebar.multiselect("Tipos de trabalho", types, default=types)
    families = sorted(enriched["familia"].dropna().unique())
    selected_families = st.sidebar.multiselect("Famílias", families, default=families)
    return enriched[
        enriched["ano"].isin(selected_years)
        & enriched["tipo_label"].isin(selected_types)
        & enriched["familia"].isin(selected_families)
    ].copy()


def page_overview(works: pd.DataFrame, source_label: str) -> None:
    st.title("Gestão geoespacial de modelos de avaliação")
    st.caption(source_label)
    points = unique_work_points(works)

    first, second, third, fourth = st.columns(4)
    first.metric("Trabalhos", f"{works['trabalho_id'].nunique():,}".replace(",", "."))
    second.metric("Usos de modelos", f"{len(works):,}".replace(",", "."))
    third.metric("Modelos distintos", f"{works['modelo_nome'].nunique():,}".replace(",", "."))
    corrected = int((points["status_coordenada"] == "invertida_corrigida").sum())
    fourth.metric("Coordenadas corrigidas", f"{corrected:,}".replace(",", "."))

    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.subheader("Onde estão os trabalhos")
        st.plotly_chart(demand_map(points), width="stretch", config=PLOTLY_CONFIG)
    with right:
        st.subheader("Modelos mais utilizados")
        ranking = (
            works.groupby("modelo_nome")["trabalho_id"]
            .nunique()
            .nlargest(12)
            .rename("trabalhos")
            .reset_index()
        )
        st.plotly_chart(
            horizontal_bar(
                ranking,
                "modelo_nome",
                "trabalhos",
                {"modelo_nome": "Modelo", "trabalhos": "Trabalhos"},
                height=530,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    st.subheader("Evolução anual por família")
    top_families = works["familia"].value_counts().head(7).index
    annual = (
        works[works["familia"].isin(top_families)]
        .groupby(["ano", "familia"])
        .size()
        .rename("usos")
        .reset_index()
    )
    st.plotly_chart(annual_stacked_bar(annual), width="stretch", config=PLOTLY_CONFIG)


def page_models(works: pd.DataFrame, catalog: pd.DataFrame) -> None:
    st.title("Catálogo de modelos")
    if catalog.empty:
        st.warning("Envie o catálogo extraído dos `.dai` para consultar métricas e períodos.")
        return
    catalog_view = add_model_dimensions(catalog)
    usage = works.groupby("modelo_nome")["trabalho_id"].nunique().rename("usos_historicos")
    catalog_view = catalog_view.merge(usage, on="modelo_nome", how="left")
    catalog_view["usos_historicos"] = catalog_view["usos_historicos"].fillna(0).astype(int)

    families = sorted(catalog_view["familia"].dropna().unique())
    selected_family = st.selectbox("Família", ["Todas"] + families)
    filtered = catalog_view if selected_family == "Todas" else catalog_view[catalog_view["familia"] == selected_family]

    display_columns = [
        "modelo_nome",
        "familia",
        "zonas_nome",
        "data_inicial",
        "data_final",
        "n_modelo",
        "n_outliers",
        "r2_ajustado",
        "usos_historicos",
    ]
    display_columns = [column for column in display_columns if column in filtered.columns]
    st.dataframe(
        filtered[display_columns].sort_values(["familia", "modelo_nome"]),
        width="stretch",
        hide_index=True,
        column_config={
            "modelo_nome": "Modelo",
            "familia": "Família",
            "zonas_nome": "Zonas",
            "data_inicial": st.column_config.DateColumn("Início dos dados"),
            "data_final": st.column_config.DateColumn("Fim dos dados"),
            "n_modelo": st.column_config.NumberColumn("Amostra", format="%d"),
            "n_outliers": st.column_config.NumberColumn("Excluídos", format="%d"),
            "r2_ajustado": st.column_config.NumberColumn("R² ajustado", format="%.3f"),
            "usos_historicos": st.column_config.NumberColumn("Usos", format="%d"),
        },
    )

    st.subheader("Ficha do modelo")
    selected_model = st.selectbox("Modelo", sorted(filtered["modelo_nome"].unique()))
    record = filtered[filtered["modelo_nome"] == selected_model].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Amostra utilizada", int(record.get("n_modelo", 0) or 0))
    columns[1].metric("Amostra completa", int(record.get("n_completo", 0) or 0))
    columns[2].metric("Outliers", int(record.get("n_outliers", 0) or 0))
    r2 = record.get("r2_ajustado", np.nan)
    columns[3].metric("R² ajustado", "—" if pd.isna(r2) else f"{r2:.3f}")
    if pd.notna(record.get("equacao")):
        st.code(str(record["equacao"]), language=None)
    detail_columns = [
        "data_inicial",
        "data_final",
        "variavel_alvo",
        "tipo_y",
        "coluna_area",
        "preditoras_json",
        "artifact_sha256",
    ]
    detail = {column: record.get(column) for column in detail_columns if column in record.index}
    st.json({key: (value.isoformat() if isinstance(value, pd.Timestamp) else value) for key, value in detail.items()})


def page_coverage(works: pd.DataFrame, catalog: pd.DataFrame, samples: pd.DataFrame) -> None:
    st.title("Cobertura e suporte espacial")
    if samples.empty:
        st.warning("Envie o arquivo de amostras espaciais para calcular suporte e distâncias.")
        return
    candidates = sorted(set(works["modelo_nome"]) & set(samples["modelo_nome"]))
    if not candidates:
        st.warning("Nenhum nome de modelo coincide entre os trabalhos e as amostras.")
        return
    selected = st.selectbox("Modelo", candidates)
    work_points = (
        works[works["modelo_nome"] == selected]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["trabalho_id", "imovel_id"])
        .copy()
    )
    sample_points = samples[samples["modelo_nome"] == selected].copy()
    work_points["distancia_km"] = haversine_nearest_km(work_points, sample_points)

    finite = work_points["distancia_km"].dropna()
    metrics = st.columns(4)
    metrics[0].metric("Dados da amostra", len(sample_points))
    metrics[1].metric("Trabalhos", work_points["trabalho_id"].nunique())
    metrics[2].metric("Distância mediana", "—" if finite.empty else f"{finite.median():.2f} km")
    metrics[3].metric("Distância P90", "—" if finite.empty else f"{finite.quantile(.90):.2f} km")

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.plotly_chart(
            coverage_map(work_points, sample_points),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with right:
        st.subheader("Distância à amostra")
        st.plotly_chart(
            distance_bar(distance_bins(finite)),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Distância é um alerta de suporte, não uma conclusão sobre inadequação do modelo. "
            "As faixas das variáveis explicativas também precisam ser verificadas."
        )

    if not catalog.empty and selected in set(catalog["modelo_nome"]):
        record = catalog[catalog["modelo_nome"] == selected].iloc[0]
        st.info(
            f"Período da amostra: {record.get('data_inicial', '—')} a "
            f"{record.get('data_final', '—')} · R² ajustado: "
            f"{record.get('r2_ajustado', '—')}"
        )


def page_priority(works: pd.DataFrame, catalog: pd.DataFrame, samples: pd.DataFrame) -> None:
    st.title("Triagem para atualização e auditoria")
    st.caption(
        "O escore organiza a revisão; não declara que um modelo é inválido. "
        "Pesos: demanda 35%, recência 25%, suporte espacial 25% e presença no catálogo 15%."
    )
    table = build_priority_table(works, catalog, samples, today=date.today())
    if table.empty:
        st.info("Não há dados suficientes para gerar a triagem.")
        return
    top = table.head(15).copy()
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Modelos prioritários")
        st.plotly_chart(
            horizontal_bar(
                top,
                "modelo_nome",
                "score_triagem",
                {"modelo_nome": "Modelo", "score_triagem": "Escore (0–100)"},
                color="nivel_triagem",
                height=550,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with right:
        st.subheader("Onde ocorreram os usos prioritários")
        mapped = works.merge(
            table[["modelo_nome", "score_triagem", "nivel_triagem"]],
            on="modelo_nome",
            how="left",
        )
        st.plotly_chart(
            priority_map(mapped),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    st.subheader("Componentes da triagem")
    display = table[
        [
            "modelo_nome",
            "familia",
            "demanda_recente",
            "demanda_total",
            "data_final",
            "dist_p90_km",
            "no_catalogo",
            "score_triagem",
            "nivel_triagem",
        ]
    ].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "modelo_nome": "Modelo",
            "familia": "Família",
            "demanda_recente": "Trabalhos recentes",
            "demanda_total": "Trabalhos totais",
            "data_final": st.column_config.DateColumn("Fim dos dados"),
            "dist_p90_km": st.column_config.NumberColumn("Distância P90 (km)", format="%.2f"),
            "no_catalogo": "No catálogo atual",
            "score_triagem": st.column_config.ProgressColumn(
                "Escore", min_value=0, max_value=100, format="%.1f"
            ),
            "nivel_triagem": "Nível",
        },
    )


def page_methodology() -> None:
    st.title("Metodologia e segurança")
    st.markdown(
        """
### O que o MVP faz

- relaciona trabalhos técnicos e nomes históricos de modelos;
- corrige coordenadas antigas invertidas durante a leitura;
- apresenta demanda territorial e temporal;
- compara trabalhos com a amostra espacial extraída dos modelos;
- cria uma triagem transparente para atualização e auditoria.

### O que o MVP ainda não conclui

- distância espacial não substitui análise de extrapolação multivariada;
- R² não determina sozinho a qualidade ou a validade do modelo;
- o escore de triagem não substitui decisão técnica;
- diferenças entre estimativa e valor adotado só poderão ser avaliadas quando esses campos forem registrados.

### Segurança dos dados

Arquivos `.dai` são pacotes `joblib/pickle` e somente devem ser abertos quando sua origem é
confiável. Eles não são carregados pelo aplicativo web. O extrator deve ser executado localmente,
em ambiente controlado, produzindo catálogos CSV sem os registros pessoais da amostra.

O banco real, os `.dai` e os catálogos derivados estão bloqueados no `.gitignore`. Para uma
demonstração pública, mantenha somente os dados sintéticos. Para uma sessão de análise, os arquivos
podem ser enviados pela barra lateral sem serem incorporados ao repositório.
"""
    )


def main() -> None:
    works, catalog, samples, source_label = load_selected_data()
    filtered_works = global_filters(works)
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navegação",
        ["Visão geral", "Modelos", "Cobertura", "Prioridades", "Metodologia"],
    )
    st.sidebar.caption("MVP · Gestão de Modelos DAI")

    if filtered_works.empty:
        st.warning("Nenhum trabalho corresponde aos filtros selecionados.")
        return
    if page == "Visão geral":
        page_overview(filtered_works, source_label)
    elif page == "Modelos":
        page_models(filtered_works, catalog)
    elif page == "Cobertura":
        page_coverage(filtered_works, catalog, samples)
    elif page == "Prioridades":
        page_priority(filtered_works, catalog, samples)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
