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
from src.config import ALLOW_DAI_UPLOADS, PRIVATE_DIR
from src.dai import extract_dai_path, extract_many_dai_bytes
from src.data import (
    analysis_availability,
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
    add_temporal_governance,
    build_priority_table,
    distance_bins,
    haversine_nearest_km,
)
from src.spatial import SPATIAL_CRS, add_reach_status


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
def cached_uploaded_dais(
    sources: tuple[tuple[str, bytes], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog, samples, errors = extract_many_dai_bytes(sources, trust_source=True)
    return standardize_catalog(catalog), standardize_samples(samples), errors


@st.cache_data(show_spinner=False)
def cached_local_data(
    db_path: str | None,
    dai_paths: tuple[str, ...],
    catalog_path: str,
    samples_path: str,
):
    works = load_sqlite_path(db_path) if db_path else pd.DataFrame()
    if not dai_paths:
        catalog = standardize_catalog(load_csv_source(catalog_path))
        samples = standardize_samples(load_csv_source(samples_path))
        return works, catalog, samples, pd.DataFrame(columns=["arquivo", "erro"])

    catalog_records: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    error_records: list[dict[str, str]] = []
    seen_model_names: set[str] = set()
    for dai_path in dai_paths:
        try:
            record, extracted_samples = extract_dai_path(dai_path, trust_source=True)
            model_key = str(record["modelo_nome"]).casefold()
            if model_key in seen_model_names:
                raise ValueError(
                    "Identificador de modelo duplicado no lote; mantenha somente uma versão."
                )
            seen_model_names.add(model_key)
            catalog_records.append(record)
            sample_records.extend(extracted_samples)
        except Exception as error:
            error_records.append(
                {"arquivo": Path(dai_path).name, "erro": f"{type(error).__name__}: {error}"}
            )
    return (
        works,
        standardize_catalog(pd.DataFrame(catalog_records)),
        standardize_samples(pd.DataFrame(sample_records)),
        pd.DataFrame(error_records, columns=["arquivo", "erro"]),
    )


def show_extraction_errors(errors: pd.DataFrame) -> None:
    if errors.empty:
        return
    st.sidebar.warning(f"{len(errors)} arquivo(s) .DAI não puderam ser lidos.")
    with st.sidebar.expander("Ver erros de extração"):
        st.dataframe(errors, hide_index=True, width="stretch")


def load_selected_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    local_db = PRIVATE_DIR / "trabalhos_tecnicos.sqlite3"
    local_catalog = PRIVATE_DIR / "catalogo_modelos.csv"
    local_samples = PRIVATE_DIR / "amostras_modelos.csv.gz"
    local_dais = tuple(
        str(path)
        for path in sorted(
            set(PRIVATE_DIR.glob("*.dai"))
            | set((PRIVATE_DIR / "modelos").glob("*.dai"))
        )
    )
    local_available = (
        local_db.exists()
        or bool(local_dais)
        or local_catalog.exists()
        or local_samples.exists()
    )

    options = ["Demonstração"]
    if local_available:
        options.append("Arquivos locais protegidos")
    options.append("Enviar arquivos nesta sessão")
    mode = st.sidebar.radio("Fonte dos dados", options, index=0)

    if mode == "Demonstração":
        works, catalog, samples = cached_demo()
        return works, catalog, samples, "Dados sintéticos de demonstração"

    if mode == "Arquivos locais protegidos":
        selected_local_dais: tuple[str, ...] = ()
        if local_dais and st.sidebar.checkbox("Processar arquivos .DAI locais"):
            if not st.sidebar.checkbox(
                "Confirmo que os .DAI locais são internos e confiáveis",
                help="Arquivos joblib/pickle podem executar código durante a leitura.",
            ):
                st.warning(
                    "Confirme a origem dos arquivos `.DAI` na barra lateral para processá-los."
                )
                st.stop()
            selected_local_dais = local_dais
        works, catalog, samples, errors = cached_local_data(
            str(local_db) if local_db.exists() else None,
            selected_local_dais,
            str(local_catalog),
            str(local_samples),
        )
        show_extraction_errors(errors)
        return (
            works,
            catalog,
            samples,
            (
                "Arquivos locais protegidos · "
                f"SQLite: {'sim' if not works.empty else 'não'} · "
                f"modelos: {len(catalog)}"
            ),
        )

    st.sidebar.caption("Os uploads são processados em memória durante a sessão ativa.")
    db_file = st.sidebar.file_uploader(
        "Trabalhos técnicos (SQLite)", type=["sqlite", "sqlite3", "db"]
    )
    dai_files = (
        st.sidebar.file_uploader(
            "Modelos (.DAI)",
            type=["dai"],
            accept_multiple_files=True,
            help="Selecione um ou vários modelos. A amostra espacial será extraída do próprio pacote.",
        )
        if ALLOW_DAI_UPLOADS
        else []
    )
    if not ALLOW_DAI_UPLOADS:
        st.sidebar.info("O envio direto de `.DAI` foi desativado nesta implantação.")
    works = cached_uploaded_sqlite(db_file.getvalue()) if db_file else pd.DataFrame()

    if dai_files:
        trusted = st.sidebar.checkbox(
            "Confirmo que os .DAI são internos e confiáveis",
            help="A confirmação é obrigatória porque joblib/pickle pode executar código.",
        )
        if not trusted:
            st.warning(
                "Os arquivos `.DAI` ainda não foram abertos. Confirme a origem na barra lateral."
            )
            st.stop()
        sources = tuple((uploaded.name, uploaded.getvalue()) for uploaded in dai_files)
        catalog, samples, errors = cached_uploaded_dais(sources)
        show_extraction_errors(errors)
        if catalog.empty:
            st.error("Nenhum arquivo `.DAI` pôde ser processado; consulte os erros de extração.")
            if works.empty:
                st.stop()
        else:
            return (
                works,
                catalog,
                samples,
                (
                    "Uploads da sessão · "
                    f"SQLite: {'sim' if not works.empty else 'não'} · "
                    f"modelos: {len(catalog)}"
                ),
            )

    with st.sidebar.expander("Alternativa: catálogos já extraídos"):
        catalog_file = st.file_uploader("Catálogo (.csv)", type=["csv"], key="catalog_csv")
        samples_file = st.file_uploader(
            "Amostras (.csv ou .csv.gz)", type=["csv", "gz"], key="samples_csv"
        )
    catalog = standardize_catalog(cached_uploaded_csv(catalog_file.getvalue())) if catalog_file else pd.DataFrame()
    samples = standardize_samples(cached_uploaded_csv(samples_file.getvalue())) if samples_file else pd.DataFrame()
    if works.empty and catalog.empty:
        st.info("Envie ao menos um banco SQLite, arquivo `.DAI` ou catálogo já extraído.")
        st.stop()
    return (
        works,
        catalog,
        samples,
        (
            "Uploads da sessão · "
            f"SQLite: {'sim' if not works.empty else 'não'} · "
            f"modelos: {len(catalog)}"
        ),
    )


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
        st.warning("Envie arquivos `.DAI` confiáveis para consultar métricas e períodos.")
        return
    catalog_view = add_temporal_governance(
        add_model_dimensions(catalog), today=date.today()
    )
    if works.empty:
        st.info("SQLite não fornecido: o uso histórico dos modelos não será exibido.")
    else:
        usage = works.groupby("modelo_nome")["trabalho_id"].nunique().rename("usos_historicos")
        catalog_view = catalog_view.merge(usage, on="modelo_nome", how="left")
        catalog_view["usos_historicos"] = (
            catalog_view["usos_historicos"].fillna(0).astype(int)
        )

    families = sorted(catalog_view["familia"].dropna().unique())
    selected_family = st.selectbox("Família", ["Todas"] + families)
    filtered = catalog_view if selected_family == "Todas" else catalog_view[catalog_view["familia"] == selected_family]

    display_columns = [
        "modelo_nome",
        "familia",
        "zonas_nome",
        "data_inicial",
        "data_final",
        "idade_dado_meses",
        "status_temporal",
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
            "data_final": st.column_config.DateColumn("Dado mais contemporâneo"),
            "idade_dado_meses": st.column_config.NumberColumn(
                "Meses completos", format="%d"
            ),
            "status_temporal": "Situação temporal",
            "n_modelo": st.column_config.NumberColumn("Amostra", format="%d"),
            "n_outliers": st.column_config.NumberColumn("Excluídos", format="%d"),
            "r2_ajustado": st.column_config.NumberColumn("R² ajustado", format="%.3f"),
            "usos_historicos": st.column_config.NumberColumn("Usos", format="%d"),
        },
    )

    st.subheader("Ficha do modelo")
    selected_model = st.selectbox("Modelo", sorted(filtered["modelo_nome"].unique()))
    record = filtered[filtered["modelo_nome"] == selected_model].iloc[0]
    temporal_status = record["status_temporal"]
    temporal_reason = record["motivo_temporal"]
    if temporal_status == "Não utilizar":
        st.error(f"Não utilizar este modelo. {temporal_reason}.")
    elif temporal_status == "Alerta":
        st.warning(f"Alerta de contemporaneidade. {temporal_reason}.")
    else:
        st.success("Modelo dentro do limite temporal de 6 meses.")
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


def page_coverage(works: pd.DataFrame, samples: pd.DataFrame) -> None:
    st.title("Cobertura e suporte espacial")
    if samples.empty:
        st.warning(
            "Os modelos processados não forneceram coordenadas `lat/lon` válidas para o mapa."
        )
        return
    candidates = sorted(set(works["modelo_nome"]) & set(samples["modelo_nome"]))
    if not candidates:
        st.warning("Nenhum nome de modelo coincide entre os trabalhos e as amostras.")
        return
    selected_models = st.multiselect(
        "Modelos sobrepostos",
        candidates,
        default=[candidates[0]],
        help="Os trabalhos exibidos são aqueles vinculados aos modelos selecionados.",
    )
    if not selected_models:
        st.info("Selecione ao menos um modelo para montar a sobreposição.")
        return
    show_samples = st.toggle("Exibir pontos da amostra", value=False)
    work_points = (
        works[works["modelo_nome"].isin(selected_models)]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["trabalho_id", "imovel_id", "modelo_nome"])
        .copy()
    )
    sample_points = samples[samples["modelo_nome"].isin(selected_models)].copy()
    work_points["distancia_km"] = np.nan
    for model_name, indices in work_points.groupby("modelo_nome").groups.items():
        model_samples = sample_points[sample_points["modelo_nome"] == model_name]
        work_points.loc[indices, "distancia_km"] = haversine_nearest_km(
            work_points.loc[indices], model_samples
        )
    work_points = add_reach_status(work_points, sample_points)

    finite = work_points["distancia_km"].dropna()
    classified = work_points["dentro_alcance"].dropna()
    inside_share = float(classified.mean() * 100) if not classified.empty else np.nan
    metrics = st.columns(5)
    metrics[0].metric("Dados da amostra", len(sample_points))
    metrics[1].metric("Trabalhos", work_points["trabalho_id"].nunique())
    metrics[2].metric("Distância mediana", "—" if finite.empty else f"{finite.median():.2f} km")
    metrics[3].metric("Distância P90", "—" if finite.empty else f"{finite.quantile(.90):.2f} km")
    metrics[4].metric(
        "Pares dentro da envoltória",
        "—" if np.isnan(inside_share) else f"{inside_share:.1f}%",
    )

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.plotly_chart(
            coverage_map(work_points, sample_points, show_samples=show_samples),
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

    st.caption(
        f"CRS: {SPATIAL_CRS}. O alcance desenhado é a envoltória convexa dos pontos da amostra; "
        "pode cobrir lacunas sem observações e não representa limite legal, zona autorizada ou "
        "ausência de extrapolação multivariada."
    )

    summary_records: list[dict[str, object]] = []
    for model_name in selected_models:
        model_works = work_points[work_points["modelo_nome"] == model_name]
        model_samples = sample_points[sample_points["modelo_nome"] == model_name]
        model_classified = model_works["dentro_alcance"].dropna()
        model_distances = model_works["distancia_km"].dropna()
        summary_records.append(
            {
                "modelo_nome": model_name,
                "pontos_amostra": len(model_samples),
                "trabalhos": model_works["trabalho_id"].nunique(),
                "dentro_envoltoria_pct": (
                    float(model_classified.mean() * 100) if not model_classified.empty else np.nan
                ),
                "dist_p90_km": (
                    float(model_distances.quantile(0.90)) if not model_distances.empty else np.nan
                ),
            }
        )
    st.subheader("Resumo por modelo")
    st.dataframe(
        pd.DataFrame(summary_records),
        hide_index=True,
        width="stretch",
        column_config={
            "modelo_nome": "Modelo",
            "pontos_amostra": "Pontos da amostra",
            "trabalhos": "Trabalhos",
            "dentro_envoltoria_pct": st.column_config.NumberColumn(
                "Dentro da envoltória", format="%.1f%%"
            ),
            "dist_p90_km": st.column_config.NumberColumn("Distância P90", format="%.2f km"),
        },
    )


def page_priority(works: pd.DataFrame, catalog: pd.DataFrame, samples: pd.DataFrame) -> None:
    st.title("Triagem para atualização e auditoria")
    st.caption(
        "Regra temporal obrigatória: acima de 6 meses gera alerta; acima de 12 meses, "
        "o modelo não deve ser utilizado. O escore preserva os pesos de demanda 35%, "
        "recência 25%, suporte espacial 25% e presença no catálogo 15%."
    )
    table = build_priority_table(works, catalog, samples, today=date.today())
    if table.empty:
        st.info("Não há dados suficientes para gerar a triagem.")
        return
    status_counts = table["status_temporal"].value_counts()
    temporal_metrics = st.columns(3)
    temporal_metrics[0].metric("Não utilizar", int(status_counts.get("Não utilizar", 0)))
    temporal_metrics[1].metric("Em alerta", int(status_counts.get("Alerta", 0)))
    temporal_metrics[2].metric("Vigentes", int(status_counts.get("Vigente", 0)))
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
                color="status_temporal",
                height=550,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with right:
        st.subheader("Onde ocorreram os usos prioritários")
        mapped = works.merge(
            table[
                ["modelo_nome", "score_triagem", "nivel_triagem", "status_temporal"]
            ],
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
            "idade_dado_meses",
            "status_temporal",
            "motivo_temporal",
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
            "data_final": st.column_config.DateColumn("Dado mais contemporâneo"),
            "idade_dado_meses": st.column_config.NumberColumn(
                "Meses completos", format="%d"
            ),
            "status_temporal": "Situação temporal",
            "motivo_temporal": "Regra aplicada",
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
- aceita SQLite e `.DAI` de forma independente, habilitando apenas análises compatíveis;
- corrige coordenadas antigas invertidas durante a leitura;
- apresenta demanda territorial e temporal;
- recebe diretamente bancos SQLite e múltiplos arquivos `.DAI` confiáveis;
- sobrepõe trabalhos à envoltória convexa da amostra espacial de cada modelo;
- cria uma triagem transparente para atualização e auditoria.

### Regra temporal de uso

- até 6 meses completos desde o dado mais contemporâneo: **Vigente**;
- acima de 6 meses e até 12 meses: **Alerta**;
- acima de 12 meses: **Não utilizar**;
- sem data final verificável: **Não utilizar** por precaução.

Os limites são calculados por mês-calendário: exatamente 6 meses ainda é vigente e exatamente
12 meses permanece em alerta. A regra temporal é obrigatória e prevalece sobre o nível qualitativo
derivado do escore ponderado.

### O que o MVP ainda não conclui

- distância espacial não substitui análise de extrapolação multivariada;
- a envoltória convexa pode preencher lacunas sem dados e não equivale à área autorizada do modelo;
- R² não determina sozinho a qualidade ou a validade do modelo;
- o aplicativo não calcula COD, PRD, mediana das razões ou regressividade sem valores observados
  e estimados em uma base de teste identificada;
- o escore de triagem não substitui decisão técnica;
- diferenças entre estimativa e valor adotado só poderão ser avaliadas quando esses campos forem registrados.

### Segurança dos dados

Arquivos `.DAI` são pacotes `joblib/pickle` e podem executar código ao serem abertos. A aplicação
exige confirmação explícita de origem confiável antes da desserialização, mas essa confirmação não
é uma sandbox. Em implantação pública, prefira desativar o envio direto e usar os CSVs produzidos
pelo extrator local em ambiente controlado.

O banco real, os `.dai` e os catálogos derivados estão bloqueados no `.gitignore`. Para uma
demonstração pública, mantenha somente os dados sintéticos. Para uma sessão de análise, os arquivos
podem ser enviados pela barra lateral sem serem incorporados ao repositório. As geometrias usam
WGS84 (EPSG:4326); as distâncias são calculadas por haversine e apresentadas em quilômetros.

### Interpretação analítica

A sobreposição é um diagnóstico pós-modelagem. Ela não altera treino, validação ou predição e não
gera vazamento entre amostra e trabalhos. Ainda assim, proximidade espacial não demonstra ausência
de viés, overfitting ou regressividade: esses riscos devem ser verificados em base de teste ou
validação espacial/temporal identificada, com COD, PRD e razões por estrato de valor.
"""
    )


def main() -> None:
    try:
        works, catalog, samples, source_label = load_selected_data()
    except (OSError, ValueError) as error:
        st.error(f"Não foi possível carregar os arquivos: {error}")
        st.stop()
    filtered_works = global_filters(works) if not works.empty else works
    availability = analysis_availability(works, catalog, samples)
    page_order = ["Visão geral", "Modelos", "Cobertura", "Prioridades", "Metodologia"]
    available_pages = [page_name for page_name in page_order if availability[page_name]]

    st.sidebar.divider()
    st.sidebar.subheader("Fontes disponíveis")
    st.sidebar.caption(
        f"{'✅' if not works.empty else '—'} SQLite · "
        f"{'✅' if not catalog.empty else '—'} modelos `.DAI`/catálogo · "
        f"{'✅' if not samples.empty else '—'} amostra espacial"
    )
    unavailable_pages = [page_name for page_name in page_order if not availability[page_name]]
    if unavailable_pages:
        st.sidebar.caption(
            "Análises desabilitadas pelas fontes ausentes: " + ", ".join(unavailable_pages)
        )
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navegação",
        available_pages,
    )
    st.sidebar.caption("MVP · Gestão de Modelos DAI")

    if filtered_works.empty and page in {"Visão geral", "Cobertura", "Prioridades"}:
        st.warning("Nenhum trabalho corresponde aos filtros selecionados.")
        return
    if page == "Visão geral":
        page_overview(filtered_works, source_label)
    elif page == "Modelos":
        page_models(filtered_works, catalog)
    elif page == "Cobertura":
        page_coverage(filtered_works, samples)
    elif page == "Prioridades":
        page_priority(filtered_works, catalog, samples)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
