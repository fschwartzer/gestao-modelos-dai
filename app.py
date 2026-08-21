from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import os

# O Streamlit Cloud pode reler app.py sem invalidar módulos já importados. A
# versão esperada funciona como um contrato de implantação: configuração e
# métricas precisam pertencer à mesma versão antes de a interface ser montada.
EXPECTED_TRIAGE_RULE_VERSION = "2.0"
EXPECTED_DAI_EXTRACTOR_VERSION = "3.1"

from src import config as config_module
from src.huggingface_source import download_hf_snapshot, locate_hf_sources

if getattr(config_module, "TRIAGE_RULE_VERSION", None) != EXPECTED_TRIAGE_RULE_VERSION:
    config_module = importlib.reload(config_module)

ALLOW_DAI_UPLOADS = config_module.ALLOW_DAI_UPLOADS
PRIVATE_DIR = config_module.PRIVATE_DIR

HF_REPO_ID = config_module.HF_REPO_ID
HF_REPO_REVISION = config_module.HF_REPO_REVISION
TRUST_HF_DAI = config_module.TRUST_HF_DAI

from src import metrics as metrics_module

if getattr(metrics_module, "TRIAGE_RULE_VERSION", None) != EXPECTED_TRIAGE_RULE_VERSION:
    metrics_module = importlib.reload(metrics_module)

add_model_dimensions = metrics_module.add_model_dimensions
add_temporal_governance = metrics_module.add_temporal_governance
build_priority_table = metrics_module.build_priority_table
consolidate_latest_model_revisions = metrics_module.consolidate_latest_model_revisions
distance_bins = metrics_module.distance_bins
haversine_nearest_km = metrics_module.haversine_nearest_km

from src.charts import (
    PLOTLY_CONFIG,
    annual_stacked_bar,
    coverage_map,
    demand_map,
    distance_bar,
    horizontal_bar,
    priority_map,
)
from src import dai as dai_module

if getattr(dai_module, "DAI_EXTRACTOR_VERSION", None) != EXPECTED_DAI_EXTRACTOR_VERSION:
    dai_module = importlib.reload(dai_module)

extract_dai_path = dai_module.extract_dai_path
extract_many_dai_bytes = dai_module.extract_many_dai_bytes
is_sha256_identifier = dai_module.is_sha256_identifier
restore_model_names_from_artifacts = dai_module.restore_model_names_from_artifacts
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
    extractor_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del extractor_version  # participa da chave do cache e invalida versões anteriores
    catalog, samples, errors = extract_many_dai_bytes(sources, trust_source=True)
    return standardize_catalog(catalog), standardize_samples(samples), errors


@st.cache_data(show_spinner=False)
def cached_restore_hf_model_names(
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    dai_paths: tuple[str, ...],
    extractor_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del extractor_version
    repaired_catalog, repaired_samples, audit = restore_model_names_from_artifacts(
        catalog, samples, dai_paths
    )
    return (
        standardize_catalog(repaired_catalog),
        standardize_samples(repaired_samples),
        audit,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_hf_snapshot(repo_id: str, revision: str) -> str:
    snapshot_path = download_hf_snapshot(
        repo_id=repo_id,
        revision=revision,
        token=os.getenv("HF_TOKEN") or None,
    )
    return str(snapshot_path)


@st.cache_data(show_spinner=False)
def cached_local_data(
    db_path: str | None,
    dai_paths: tuple[str, ...],
    catalog_path: str,
    samples_path: str,
    extractor_version: str,
):
    del extractor_version
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
    if "erro" in errors:
        frequencies = errors["erro"].fillna("Erro não informado").value_counts()
        most_common_error = str(frequencies.index[0])
        st.sidebar.caption(
            f"Causa mais frequente ({int(frequencies.iloc[0])} arquivo(s)): "
            f"{most_common_error}"
        )
    with st.sidebar.expander("Ver erros de extração"):
        ordered_columns = [column for column in ("erro", "arquivo") if column in errors]
        st.dataframe(
            errors[ordered_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "erro": st.column_config.TextColumn("Erro completo", width="large"),
                "arquivo": st.column_config.TextColumn("Arquivo", width="medium"),
            },
        )
        st.download_button(
            "Baixar diagnóstico (.csv)",
            errors.to_csv(index=False).encode("utf-8"),
            file_name="erros_extracao_dai.csv",
            mime="text/csv",
        )


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

    options = [
        "Demonstração",
        "Repositório Hugging Face",
    ]

    if local_available:
        options.append("Arquivos locais protegidos")

    options.append("Enviar arquivos nesta sessão")
    mode = st.sidebar.radio("Fonte dos dados", options, index=0)

    if mode == "Demonstração":
        works, catalog, samples = cached_demo()
        return works, catalog, samples, "Dados sintéticos de demonstração"

    if mode == "Repositório Hugging Face":
        if st.sidebar.button(
            "Atualizar e reprocessar dados",
            help="Limpa o snapshot e os resultados de importação mantidos em cache.",
        ):
            cached_hf_snapshot.clear()
            cached_local_data.clear()
            cached_restore_hf_model_names.clear()
            st.rerun()
        try:
            with st.spinner("Baixando e preparando os arquivos do Hugging Face..."):
                snapshot_dir = Path(
                    cached_hf_snapshot(
                        HF_REPO_ID,
                        HF_REPO_REVISION,
                    )
                )

            db_path, dai_paths, catalog_path, samples_path = (
                locate_hf_sources(snapshot_dir)
            )

            # Se os CSVs pré-extraídos existirem, eles têm preferência:
            # são mais rápidos e evitam desserializar pickle no servidor.
            has_extracted_catalog = (
                catalog_path is not None
                and samples_path is not None
            )

            dai_paths_to_load = (
                ()
                if has_extracted_catalog
                else tuple(str(path) for path in dai_paths)
            )

            if dai_paths_to_load and not TRUST_HF_DAI:
                st.error(
                    "Há arquivos .DAI no Hugging Face, mas sua abertura não "
                    "foi autorizada. Defina TRUST_HF_DAI=true nos Secrets."
                )
                st.stop()

            missing_csv = snapshot_dir / "__arquivo_ausente__.csv"

            works, catalog, samples, errors = cached_local_data(
                str(db_path) if db_path else None,
                dai_paths_to_load,
                str(catalog_path or missing_csv),
                str(samples_path or missing_csv),
                EXPECTED_DAI_EXTRACTOR_VERSION,
            )

            name_repairs = pd.DataFrame()
            if has_extracted_catalog and dai_paths:
                catalog, samples, name_repairs = cached_restore_hf_model_names(
                    catalog,
                    samples,
                    tuple(str(path) for path in dai_paths),
                    EXPECTED_DAI_EXTRACTOR_VERSION,
                )

            opaque_names = (
                catalog["modelo_nome"].map(is_sha256_identifier)
                if "modelo_nome" in catalog
                else pd.Series(dtype=bool)
            )
            if opaque_names.any():
                st.sidebar.warning(
                    f"{int(opaque_names.sum())} modelo(s) ainda possuem identificador "
                    "opaco. O arquivo .DAI correspondente não foi localizado de forma "
                    "não ambígua pelo hash de conteúdo."
                )

            show_extraction_errors(errors)

            if works.empty and catalog.empty:
                st.error(
                    "O repositório foi acessado, mas nenhuma fonte compatível "
                    "foi encontrada."
                )
                st.stop()

            model_source = (
                "catálogo pré-extraído"
                if has_extracted_catalog
                else f"{len(dai_paths)} arquivo(s) .DAI"
            )

            return (
                works,
                catalog,
                samples,
                "Hugging Face · "
                f"SQLite: {'sim' if not works.empty else 'não'} · "
                f"modelos: {len(catalog)} · "
                f"origem: {model_source} · "
                f"nomes restaurados: {len(name_repairs)}",
            )

        except Exception as error:
            st.error(
                "Não foi possível carregar o repositório do Hugging Face: "
                f"{type(error).__name__}: {error}"
            )
            st.stop()

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
            EXPECTED_DAI_EXTRACTOR_VERSION,
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
        catalog, samples, errors = cached_uploaded_dais(
            sources, EXPECTED_DAI_EXTRACTOR_VERSION
        )
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


def global_filters(works: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    enriched = add_model_dimensions(works)
    st.sidebar.divider()
    st.sidebar.subheader("Filtros")
    st.sidebar.caption(
        "Afetam todas as análises, inclusive a fila de Prioridades e o arquivo CSV."
    )
    years = sorted(pd.to_numeric(enriched["ano"], errors="coerce").dropna().astype(int).unique())
    selected_years = st.sidebar.multiselect("Anos", years, default=years)
    types = sorted(enriched["tipo_label"].dropna().unique())
    selected_types = st.sidebar.multiselect("Tipos de trabalho", types, default=types)
    families = sorted(enriched["familia"].dropna().unique())
    selected_families = st.sidebar.multiselect("Famílias", families, default=families)
    filtered = enriched[
        enriched["ano"].isin(selected_years)
        & enriched["tipo_label"].isin(selected_types)
        & enriched["familia"].isin(selected_families)
    ].copy()
    return filtered, selected_families


def filter_model_sources_by_families(
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    selected_families: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica aos insumos dos modelos o mesmo recorte familiar da interface."""

    def filter_source(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        enriched = add_model_dimensions(frame)
        return enriched[enriched["familia"].isin(selected_families)].copy()

    return filter_source(catalog), filter_source(samples)


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
        st.warning(
            "Nenhum modelo corresponde às famílias selecionadas na barra lateral."
        )
        return
    st.caption(
        "O catálogo obedece às famílias selecionadas na barra lateral e exibe somente "
        "a revisão alfabética mais recente de cada linhagem."
    )
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
        "revisao_modelo",
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
            "revisao_modelo": "Revisão",
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


def page_coverage(
    works: pd.DataFrame, catalog: pd.DataFrame, samples: pd.DataFrame
) -> None:
    st.title("Cobertura e suporte espacial")
    st.caption(
        "Somente a revisão mais recente de cada linhagem é considerada; modelos vigentes "
        "e em alerta são selecionados por padrão."
    )
    if samples.empty:
        st.warning(
            "Os modelos processados não forneceram coordenadas `lat/lon` válidas para o mapa."
        )
        return
    catalog_dimensions = add_model_dimensions(catalog)
    active_families = sorted(catalog_dimensions["familia"].dropna().unique())
    family_catalog = catalog_dimensions.copy()
    candidates = sorted(
        set(family_catalog["modelo_nome"]) & set(samples["modelo_nome"])
    )
    if not candidates:
        st.warning(
            "Nenhum modelo das famílias selecionadas coincide entre o catálogo e as "
            "amostras. Ajuste o filtro “Famílias” na barra lateral."
        )
        return
    st.caption(
        "Famílias ativas no mapa: "
        + ", ".join(active_families)
        + ". Altere a seleção na barra lateral para atualizar modelos e envoltórias."
    )
    governed_catalog = add_temporal_governance(
        family_catalog, today=date.today()
    )
    default_models = sorted(
        set(
            governed_catalog.loc[
                governed_catalog["status_temporal"].isin(["Vigente", "Alerta"]),
                "modelo_nome",
            ]
        )
        & set(candidates)
    )
    selected_models = st.multiselect(
        "Modelos sobrepostos",
        candidates,
        default=default_models,
        key="coverage_models_" + "__".join(active_families),
        help=(
            "As opções obedecem ao filtro lateral de Famílias. Por padrão são "
            "selecionadas as revisões mais recentes em situação Vigente ou Alerta."
        ),
    )
    if not selected_models:
        st.info(
            "Nenhum modelo vigente ou em alerta está disponível nos filtros atuais. "
            "Selecione manualmente um modelo para montar a sobreposição."
        )
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
    temporal_status_by_model = governed_catalog.set_index("modelo_nome")[
        "status_temporal"
    ].to_dict()
    for model_name in selected_models:
        model_works = work_points[work_points["modelo_nome"] == model_name]
        model_samples = sample_points[sample_points["modelo_nome"] == model_name]
        model_classified = model_works["dentro_alcance"].dropna()
        model_distances = model_works["distancia_km"].dropna()
        summary_records.append(
            {
                "modelo_nome": model_name,
                "status_temporal": temporal_status_by_model.get(model_name, "Indeterminado"),
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
            "status_temporal": "Situação temporal",
            "pontos_amostra": "Pontos da amostra",
            "trabalhos": "Trabalhos",
            "dentro_envoltoria_pct": st.column_config.NumberColumn(
                "Dentro da envoltória", format="%.1f%%"
            ),
            "dist_p90_km": st.column_config.NumberColumn("Distância P90", format="%.2f km"),
        },
    )


def page_priority(works: pd.DataFrame, catalog: pd.DataFrame, samples: pd.DataFrame) -> None:
    st.title("Fila de intervenção e governança")
    st.caption(
        "A situação temporal determina a possibilidade de uso; P0–P3 organiza a ação. "
        "O escore é apenas um desempate operacional e não mede qualidade preditiva."
    )
    table = build_priority_table(works, catalog, samples, today=date.today())
    if table.empty:
        st.info("Não há dados suficientes para gerar a triagem.")
        return
    display_columns = [
        "modelo_nome",
        "familia",
        "prioridade_intervencao",
        "acao_recomendada",
        "demanda_recente",
        "demanda_total",
        "data_final",
        "idade_dado_meses",
        "status_temporal",
        "status_completude",
        "motivo_completude",
        "dist_p90_km",
        "amostra_nn_p90_km",
        "dist_relativa_p90",
        "fora_envoltoria_pct",
        "score_suporte",
        "confianca_suporte",
        "score_auxiliar",
        "cobertura_evidencias_pct",
        "status_escore",
    ]
    required_columns = set(display_columns) | {
        "metodo_demanda",
        "regra_triagem_versao",
        "referencia_demanda_trabalhos",
        "score_triagem",
        "nivel_triagem",
    }
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        st.error(
            "A plataforma manteve módulos de versões diferentes durante a atualização. "
            "Reinicie o aplicativo em “Manage app” → “Reboot app” para concluir a "
            "implantação."
        )
        st.caption(
            "Campos da regra atual ainda indisponíveis: " + ", ".join(missing_columns)
        )
        return
    st.info(
        "Esta fila corresponde aos filtros atuais da barra lateral. Uma seleção parcial "
        "pode alterar a demanda e as classes P0–P3; para auditar o portfólio completo, "
        f"selecione todas as opções. Demanda: {table.iloc[0]['metodo_demanda']}; regra "
        f"{table.iloc[0]['regra_triagem_versao']}; referência de impacto: "
        f"{int(table.iloc[0]['referencia_demanda_trabalhos'])} trabalhos na janela."
    )
    st.warning(
        "Não há base de teste identificada para COD, PRD, mediana das razões e "
        "regressividade. Por isso, todos os escores numéricos permanecem provisórios."
    )
    st.download_button(
        "Baixar fila filtrada (.csv)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fila_modelos_filtrada_{date.today().isoformat()}.csv",
        mime="text/csv",
        help="Snapshot agregado, sem endereços ou identificadores de imóveis.",
    )
    priority_counts = table["prioridade_intervencao"].value_counts()
    priority_metrics = st.columns(4)
    priority_metrics[0].metric("P0 · ação imediata", int(priority_counts.get("P0", 0)))
    priority_metrics[1].metric("P1 · prioritária", int(priority_counts.get("P1", 0)))
    priority_metrics[2].metric("P2 · programar", int(priority_counts.get("P2", 0)))
    priority_metrics[3].metric("P3 · monitorar", int(priority_counts.get("P3", 0)))

    status_counts = table["status_temporal"].value_counts()
    st.caption(
        f"Situação temporal: {int(status_counts.get('Não utilizar', 0))} não utilizar · "
        f"{int(status_counts.get('Alerta', 0))} em alerta · "
        f"{int(status_counts.get('Vigente', 0))} vigentes."
    )
    top = table.head(15).copy()
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Ordenação dentro das classes")
        st.plotly_chart(
            horizontal_bar(
                top,
                "modelo_nome",
                "score_auxiliar",
                {"modelo_nome": "Modelo", "score_auxiliar": "Escore auxiliar"},
                color="prioridade_intervencao",
                height=550,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with right:
        st.subheader("Onde ocorreram os usos prioritários")
        mapped = works.merge(
            table[
                [
                    "modelo_nome",
                    "score_triagem",
                    "prioridade_intervencao",
                    "nivel_triagem",
                    "status_temporal",
                ]
            ],
            on="modelo_nome",
            how="left",
        )
        st.plotly_chart(
            priority_map(mapped),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    st.subheader("Fundamentação da fila")
    display = table[display_columns].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "modelo_nome": "Modelo",
            "familia": "Família",
            "prioridade_intervencao": "Prioridade",
            "acao_recomendada": "Próxima ação",
            "demanda_recente": "Trabalhos na janela",
            "demanda_total": "Trabalhos totais",
            "data_final": st.column_config.DateColumn("Dado mais contemporâneo"),
            "idade_dado_meses": st.column_config.NumberColumn(
                "Meses completos", format="%d"
            ),
            "status_temporal": "Situação temporal",
            "status_completude": "Completude",
            "motivo_completude": "Pendências de evidência",
            "dist_p90_km": st.column_config.NumberColumn("Distância P90 (km)", format="%.2f"),
            "amostra_nn_p90_km": st.column_config.NumberColumn(
                "P90 entre dados da amostra (km)", format="%.2f"
            ),
            "dist_relativa_p90": st.column_config.NumberColumn(
                "Distância relativa", format="%.2f×"
            ),
            "fora_envoltoria_pct": st.column_config.NumberColumn(
                "Fora da envoltória", format="%.1f%%"
            ),
            "score_suporte": st.column_config.ProgressColumn(
                "Risco espacial", min_value=0, max_value=1, format="%.2f"
            ),
            "confianca_suporte": "Confiança espacial",
            "score_auxiliar": st.column_config.ProgressColumn(
                "Escore auxiliar", min_value=0, max_value=100, format="%.1f"
            ),
            "cobertura_evidencias_pct": st.column_config.ProgressColumn(
                "Cobertura das evidências", min_value=0, max_value=100, format="%.0f%%"
            ),
            "status_escore": "Validade do escore",
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
12 meses permanece em alerta. A regra temporal é obrigatória e não faz parte do escore auxiliar.

### Fila de intervenção

- **P0:** modelo que não deve ser utilizado, mas teve demanda na janela recente;
- **P1:** atualização prioritária ou decisão entre atualização e aposentadoria;
- **P2:** revisão programada, alerta sem demanda ou evidências incompletas;
- **P3:** monitoramento periódico.

A fila, seus gráficos, mapa, tabela e CSV obedecem aos filtros de anos, tipos de trabalho e famílias.
Uma seleção parcial produz uma visão analítica parcial; para a auditoria institucional do portfólio,
é necessário selecionar todas as opções. Quando existe data completa do trabalho, a demanda usa
janela móvel de 12 meses. Se o SQLite fornece apenas o ano, a aplicação usa o ano civil atual e o
anterior e identifica explicitamente essa aproximação.

O escore numérico serve somente para ordenar modelos dentro de uma mesma classe. Ele separa impacto
operacional, risco de desempenho, suporte espacial e risco operacional, informando também a cobertura
das evidências. Como os `.DAI` atuais não identificam uma base de teste, o risco de desempenho fica
ausente e o escore é marcado como provisório. R² de ajuste não é usado como substituto.

Ausência de catálogo, data ou geometria é registrada em **Completude**, sem ser interpretada como
prova de baixo desempenho. O suporte espacial compara a distância P90 dos trabalhos recentes com a
distância P90 entre vizinhos da própria amostra e com a proporção fora da envoltória convexa; as
distâncias usam WGS84 e haversine em quilômetros. Com menos de 10 trabalhos georreferenciados,
o diagnóstico é exploratório e não promove automaticamente a prioridade.

### Consolidação de revisões

- nomes com a mesma base numérica e sufixos finais alfabéticos são tratados como uma linhagem;
- a ordem é alfabética, portanto `MOD_V_TER_Z1_006J` substitui `MOD_V_TER_Z1_006I`;
- usos históricos das revisões anteriores são atribuídos à revisão mais recente para medir demanda;
- catálogo e amostra espacial antigos são descartados, sem misturar metadados ou coordenadas de
  treinamento entre revisões;
- o mapa seleciona inicialmente todas as revisões mais recentes **Vigentes** ou em **Alerta**.

A revisão vencedora é determinada pela união dos nomes encontrados nas fontes. Se uma fonte indicar
uma revisão nova mas seu `.DAI` não estiver disponível, o aplicativo não herda silenciosamente os
metadados ou a geometria da revisão antiga.

### O que o MVP ainda não conclui

- distância espacial não substitui análise de extrapolação multivariada;
- a envoltória convexa pode preencher lacunas sem dados e não equivale à área autorizada do modelo;
- R² não determina sozinho a qualidade ou a validade do modelo;
- o aplicativo não calcula COD, PRD, mediana das razões ou regressividade sem valores observados
  e estimados em uma base de teste identificada;
- o escore auxiliar não substitui decisão técnica e não é conclusivo sem validação de desempenho;
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
    works, catalog, samples, revision_audit = consolidate_latest_model_revisions(
        works, catalog, samples
    )
    if works.empty:
        filtered_works = works
        selected_families = (
            sorted(add_model_dimensions(catalog)["familia"].dropna().unique())
            if not catalog.empty
            else []
        )
    else:
        filtered_works, selected_families = global_filters(works)
    filtered_catalog, filtered_samples = filter_model_sources_by_families(
        catalog, samples, selected_families
    )
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
    if not catalog.empty:
        valid_model_dates = (
            int(pd.to_datetime(catalog["data_final"], errors="coerce").notna().sum())
            if "data_final" in catalog
            else 0
        )
        st.sidebar.caption(
            "Datas dos modelos: "
            f"{valid_model_dates}/{len(catalog)} com data final verificável."
        )
    if not works.empty and not catalog.empty:
        work_names = set(works["modelo_nome"].dropna().astype(str).str.casefold())
        catalog_names = set(catalog["modelo_nome"].dropna().astype(str).str.casefold())
        matched_names = work_names & catalog_names
        st.sidebar.caption(
            "Compatibilidade SQLite ↔ catálogo: "
            f"{len(matched_names)}/{len(catalog_names)} modelo(s) do catálogo "
            "possuem uso histórico correspondente."
        )
        if not matched_names:
            st.sidebar.warning(
                "Nenhum nome de modelo do catálogo coincide com o SQLite. "
                "Cobertura e prioridades não devem ser interpretadas até revisar a origem."
            )
    consolidated = revision_audit[revision_audit["n_versoes"] > 1]
    if not consolidated.empty:
        st.sidebar.caption(
            f"{len(consolidated)} linhagem(ns) consolidada(s) na revisão mais recente."
        )
        with st.sidebar.expander("Ver consolidação de versões"):
            st.dataframe(
                consolidated[
                    ["modelo_linhagem", "modelo_mais_recente", "versoes_encontradas"]
                ],
                hide_index=True,
                width="stretch",
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
        page_models(filtered_works, filtered_catalog)
    elif page == "Cobertura":
        page_coverage(filtered_works, filtered_catalog, filtered_samples)
    elif page == "Prioridades":
        page_priority(filtered_works, filtered_catalog, filtered_samples)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
