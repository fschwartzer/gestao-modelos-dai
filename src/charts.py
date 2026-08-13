from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _center(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return {"lat": -30.05, "lon": -51.18}
    return {"lat": float(valid["latitude"].median()), "lon": float(valid["longitude"].median())}


def demand_map(points: pd.DataFrame) -> go.Figure:
    valid = points.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return go.Figure()
    center = _center(valid)
    figure = px.density_map(
        valid,
        lat="latitude",
        lon="longitude",
        radius=18,
        center=center,
        zoom=9.6,
        hover_name="nome",
        hover_data={"ano": True, "tipo_label": True, "latitude": False, "longitude": False},
        color_continuous_scale="YlOrRd",
        opacity=0.72,
        map_style="carto-positron",
    )
    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=530,
        coloraxis_colorbar_title="Densidade",
    )
    return figure


def coverage_map(work_points: pd.DataFrame, sample_points: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    center_source = pd.concat([work_points, sample_points], ignore_index=True)
    center = _center(center_source)
    if not sample_points.empty:
        figure.add_trace(
            go.Scattermap(
                lat=sample_points["latitude"],
                lon=sample_points["longitude"],
                mode="markers",
                name="Amostra do modelo",
                marker={"size": 8, "color": "#2563eb", "opacity": 0.55},
                hovertemplate="Dado de treinamento<extra></extra>",
            )
        )
    if not work_points.empty:
        custom = work_points[["nome", "ano", "distancia_km"]].to_numpy()
        figure.add_trace(
            go.Scattermap(
                lat=work_points["latitude"],
                lon=work_points["longitude"],
                mode="markers",
                name="Trabalho técnico",
                marker={"size": 12, "color": "#dc2626", "opacity": 0.9, "symbol": "diamond"},
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Ano: %{customdata[1]}"
                    "<br>Distância: %{customdata[2]:.2f} km<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        map={"style": "carto-positron", "center": center, "zoom": 10},
        margin=dict(l=0, r=0, t=0, b=0),
        height=545,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
    )
    return figure


def priority_map(points: pd.DataFrame) -> go.Figure:
    valid = points.dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        return go.Figure()
    color_map = {"Alta": "#dc2626", "Média": "#f59e0b", "Baixa": "#16a34a"}
    figure = px.scatter_map(
        valid,
        lat="latitude",
        lon="longitude",
        color="nivel_triagem",
        color_discrete_map=color_map,
        category_orders={"nivel_triagem": ["Alta", "Média", "Baixa"]},
        hover_name="nome",
        hover_data={
            "modelo_nome": True,
            "score_triagem": ":.1f",
            "ano": True,
            "latitude": False,
            "longitude": False,
        },
        center=_center(valid),
        zoom=9.6,
        opacity=0.72,
        map_style="carto-positron",
    )
    figure.update_traces(marker={"size": 9})
    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        legend_title="Triagem",
    )
    return figure


def horizontal_bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    labels: dict[str, str],
    color: str | None = None,
    height: int = 430,
) -> go.Figure:
    ordered = frame.sort_values(value, ascending=True)
    figure = px.bar(
        ordered,
        x=value,
        y=category,
        orientation="h",
        color=color,
        text=value,
        labels=labels,
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=30, t=10, b=0),
        legend_title_text="",
        yaxis_title="",
    )
    return figure


def annual_stacked_bar(frame: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        frame,
        x="ano",
        y="usos",
        color="familia",
        labels={"ano": "Ano", "usos": "Usos de modelos", "familia": "Família"},
    )
    figure.update_layout(
        barmode="stack",
        height=410,
        margin=dict(l=0, r=0, t=10, b=0),
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
    )
    return figure


def distance_bar(frame: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        frame,
        x="faixa",
        y="trabalhos",
        text="trabalhos",
        labels={"faixa": "Distância ao dado mais próximo", "trabalhos": "Trabalhos"},
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    return figure
