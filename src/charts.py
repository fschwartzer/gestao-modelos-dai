from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.spatial import empirical_reach_polygon


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

MODEL_COLORS = [
    "#2563eb",
    "#7c3aed",
    "#0891b2",
    "#c2410c",
    "#4d7c0f",
    "#be185d",
    "#0f766e",
    "#a16207",
]


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


def _transparent(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def coverage_map(
    work_points: pd.DataFrame,
    sample_points: pd.DataFrame,
    *,
    show_samples: bool = False,
) -> go.Figure:
    figure = go.Figure()
    center_source = pd.concat([work_points, sample_points], ignore_index=True)
    center = _center(center_source)

    palette = MODEL_COLORS
    for sequence, (model_name, model_samples) in enumerate(
        sample_points.groupby("modelo_nome", sort=True)
    ):
        color = palette[sequence % len(palette)]
        polygon = empirical_reach_polygon(model_samples)
        if polygon is not None:
            longitude, latitude = polygon.exterior.xy
            figure.add_trace(
                go.Scattermap(
                    lat=list(latitude),
                    lon=list(longitude),
                    mode="lines",
                    fill="toself",
                    fillcolor=_transparent(color, 0.16),
                    line={"color": color, "width": 2},
                    name=f"Envoltória · {model_name}",
                    hovertemplate=f"<b>{model_name}</b><br>Envoltória convexa da amostra<extra></extra>",
                )
            )
        if show_samples:
            figure.add_trace(
                go.Scattermap(
                    lat=model_samples["latitude"],
                    lon=model_samples["longitude"],
                    mode="markers",
                    name=f"Amostra · {model_name}",
                    marker={"size": 6, "color": color, "opacity": 0.45},
                    hovertemplate=f"Dado da amostra<br>{model_name}<extra></extra>",
                )
            )
    if not work_points.empty:
        styles = {
            "Dentro da envoltória": ("#15803d", "circle"),
            "Fora da envoltória": ("#dc2626", "diamond"),
            "indeterminado": ("#64748b", "circle"),
        }
        for status, (color, symbol) in styles.items():
            subset = work_points[work_points["status_alcance"] == status]
            if subset.empty:
                continue
            custom = subset[["nome", "ano", "modelo_nome", "distancia_km"]].to_numpy()
            figure.add_trace(
                go.Scattermap(
                    lat=subset["latitude"],
                    lon=subset["longitude"],
                    mode="markers",
                    name=status,
                    marker={"size": 11, "color": color, "opacity": 0.9, "symbol": symbol},
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Ano: %{customdata[1]}"
                        "<br>Modelo: %{customdata[2]}"
                        "<br>Distância à amostra: %{customdata[3]:.2f} km<extra></extra>"
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
