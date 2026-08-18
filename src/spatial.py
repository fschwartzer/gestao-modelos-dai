from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.prepared import prep


SPATIAL_CRS = "EPSG:4326"
MIN_REACH_POINTS = 3


def empirical_reach_polygon(sample_points: pd.DataFrame) -> Polygon | None:
    """Calcula a envoltória convexa da amostra em WGS84, sem interpretar distância."""

    if sample_points.empty:
        return None
    valid = (
        sample_points[["latitude", "longitude"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .drop_duplicates()
    )
    if len(valid) < MIN_REACH_POINTS:
        return None
    geometry = MultiPoint(valid[["longitude", "latitude"]].to_numpy()).convex_hull
    return geometry if isinstance(geometry, Polygon) and geometry.is_valid else None


def add_reach_status(work_points: pd.DataFrame, sample_points: pd.DataFrame) -> pd.DataFrame:
    """Classifica pares trabalho-modelo dentro/fora da envoltória da própria amostra."""

    result = work_points.copy()
    result["status_alcance"] = "indeterminado"
    result["dentro_alcance"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if result.empty or sample_points.empty or "modelo_nome" not in result:
        return result

    for model_name, indices in result.groupby("modelo_nome", dropna=False).groups.items():
        model_samples = sample_points[sample_points["modelo_nome"] == model_name]
        polygon = empirical_reach_polygon(model_samples)
        if polygon is None:
            continue
        prepared = prep(polygon)
        valid_indices = result.loc[indices].dropna(subset=["latitude", "longitude"]).index
        inside = [
            prepared.covers(Point(float(result.at[index, "longitude"]), float(result.at[index, "latitude"])))
            for index in valid_indices
        ]
        result.loc[valid_indices, "dentro_alcance"] = inside
        result.loc[valid_indices, "status_alcance"] = np.where(
            inside, "Dentro da envoltória", "Fora da envoltória"
        )
    return result
