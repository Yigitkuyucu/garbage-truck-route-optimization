"""OSM yol agi + bina poligonlari indirme ve cache.

KABUK modulu (data/*): sinif/dataframe serbest.

Yol agi ve binalar OSMnx ile BIR KEZ indirilir, diske yazilir. Sonraki kosular
aga hic dokunmaz - cache bolge kimligine (region hash) gore anahtarlanir.

Veri kaynagi: (c) OpenStreetMap contributors (ODbL).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

from config import Region

CACHE_DIR = Path("data/cache")

# Bina poligonlarini getirirken kullanilan OSM etiketi
_BUILDING_TAGS = {"building": True}


@dataclass(frozen=True)
class OSMData:
    """Indirilen ham OSM verisi (yol agi + binalar)."""

    graph: nx.MultiDiGraph          # yonlu yol agi (tek yon sokaklar dahil)
    buildings: gpd.GeoDataFrame     # bina poligonlari (WGS84, EPSG:4326)
    region_key: str                 # cache anahtari


def region_key(region: Region) -> str:
    """Bolgeyi tekillestiren kisa anahtar (merkez + yaricap + ag tipi)."""
    lat, lon = region.center
    raw = f"{region.name}|{lat:.6f}|{lon:.6f}|{region.radius_m}|{region.network_type}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{region.name}_{digest}"


def _graph_path(key: str) -> Path:
    return CACHE_DIR / f"{key}_graph.graphml"


def _buildings_path(key: str) -> Path:
    return CACHE_DIR / f"{key}_buildings.parquet"


def load_osm(region: Region, *, force_refresh: bool = False) -> OSMData:
    """Yol agi + binalari getir. Cache varsa diskten, yoksa OSMnx ile indir.

    Cache diske yazilir; ikinci kosu aga dokunmaz.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = region_key(region)
    gpath = _graph_path(key)
    bpath = _buildings_path(key)

    if not force_refresh and gpath.exists() and bpath.exists():
        graph = ox.io.load_graphml(gpath)
        buildings = gpd.read_parquet(bpath)
        return OSMData(graph=graph, buildings=buildings, region_key=key)

    center = region.center  # (lat, lon)
    graph = ox.graph_from_point(
        center,
        dist=region.radius_m,
        network_type=region.network_type,
        simplify=True,
    )
    buildings = ox.features_from_point(
        center,
        tags=_BUILDING_TAGS,
        dist=region.radius_m,
    )
    # Sadece poligon/multipoligon geometrileri tut (nokta etiketleri elenir)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings = buildings.reset_index()

    ox.io.save_graphml(graph, gpath)
    _write_buildings(buildings, bpath)

    return OSMData(graph=graph, buildings=buildings, region_key=key)


def _write_buildings(buildings: gpd.GeoDataFrame, path: Path) -> None:
    """Binalari parquet olarak yaz. OSM sutunlari heterojen olabilir; sadece
    ihtiyac duyulan sutunlar + geometri korunur, gerisi stringlestirilir."""
    keep = [c for c in ("building", "building:levels", "amenity", "shop") if c in buildings.columns]
    gdf = buildings[[*keep, "geometry"]].copy()
    for col in keep:
        gdf[col] = gdf[col].astype("string")
    gdf.to_parquet(path)
