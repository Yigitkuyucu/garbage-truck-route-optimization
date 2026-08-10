"""Projeksiyon ve geometri yardimcilari (KABUK).

Tum metrik hesaplar (alan, mesafe, en yakin) WGS84'te degil, bolgenin UTM
projeksiyonunda yapilir. Boylece 'metre' gercekten metre olur.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, Point

Coord = tuple[float, float]  # (lat, lon)


def project_graph(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, str]:
    """Yol agini UTM'e projekte et. (projeli_graf, crs) dondur."""
    graph_proj = ox.projection.project_graph(graph)
    crs = str(graph_proj.graph["crs"])
    return graph_proj, crs


def latlon_to_xy(coords: list[Coord], crs: str) -> np.ndarray:
    """(lat, lon) listesini verilen CRS'te (x, y) metre dizisine cevir.

    Donen dizi float64, sekil (N, 2): [[x, y], ...].
    """
    if not coords:
        return np.empty((0, 2), dtype=np.float64)
    geoms = [Point(lon, lat) for lat, lon in coords]
    pts = gpd.GeoSeries(geoms, crs="EPSG:4326")
    proj = pts.to_crs(crs)
    return np.column_stack([proj.x.to_numpy(), proj.y.to_numpy()]).astype(np.float64)


def project_buildings(buildings: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Binalari CRS'e projekte et; taban alani (m2) ve merkez (x, y) ekle."""
    gdf = buildings.to_crs(crs).copy()
    gdf["area_m2"] = gdf.geometry.area
    cent = gdf.geometry.centroid
    gdf["cx"] = cent.x
    gdf["cy"] = cent.y
    return gdf


def segments_to_xy(segments: list[tuple[Coord, Coord]], crs: str) -> list[LineString]:
    """Pazar koridor segmentlerini (lat,lon çiftleri) projeli LineString'e cevir."""
    lines: list[LineString] = []
    for a, b in segments:
        xy = latlon_to_xy([a, b], crs)
        lines.append(LineString([(xy[0, 0], xy[0, 1]), (xy[1, 0], xy[1, 1])]))
    return lines


def nearest_index(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Her point icin en yakin target'in indeksini dondur (Oklid, metre).

    points: (N, 2), targets: (M, 2) -> (N,) int64. Kucuk olceklerde (M<~500)
    tam matris yeterince hizli.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    diff = points[:, None, :] - targets[None, :, :]
    d2 = np.einsum("nmk,nmk->nm", diff, diff)
    return np.argmin(d2, axis=1).astype(np.int64)
