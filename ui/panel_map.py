"""Folium harita olusturucu (KABUK) - bir gunun rotasi + atlanan konteynerler.

Hikaye (kullanici karari): B0 hepsini ziyaret eder; akilli cozucu bazilarini
ATLAR. Atlananlar SOLUK gri; ziyaret edilenler dolulukla renklenir; her arac
ayri renk rota. Garaj/dokum ayri isaretler; deadhead (sabit 16 km) KESIKLI -
optimizasyonun dokunamadigi pay gorunur olsun.

Cizim duz cizgi (dugumden dugume) - ziyaret SIRASI ve ATLAMA net; gercek yol
geometrisi bir sonraki cilalama. Cekirdek buraya girmez.
"""

from __future__ import annotations

import folium

from data.dataset import BuiltDataset
from sim.engine import DayState

# Arac rota renkleri (ayirt edilebilir, koyu paletten).
VEHICLE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _fill_color(ratio: float) -> str:
    """Doluluk orani -> yesil(bos) - sari - kirmizi(dolu) gradyani (hex)."""
    r = max(0.0, min(1.0, ratio))
    if r < 0.5:  # yesil -> sari
        t = r / 0.5
        red, green = int(60 + t * 195), 180
    else:  # sari -> kirmizi
        t = (r - 0.5) / 0.5
        red, green = 255, int(180 - t * 160)
    return f"#{red:02x}{green:02x}30"


def _bounds(coords: list[tuple[float, float]]) -> list[list[float]]:
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def build_map(
    ds: BuiltDataset,
    state: DayState,
    *,
    title: str,
    include_depot_dump: bool = False,
    height: int = 520,
) -> folium.Map:
    """Bir cozucunun bir gunku cozumunu haritala."""
    coords = ds.container_coords
    depot, dump = ds.depot_coord, ds.dump_coord
    fill = state.fill_before
    volume = state.volume
    visited = state.visited

    ctr_lat = sum(c[0] for c in coords) / len(coords)
    ctr_lon = sum(c[1] for c in coords) / len(coords)
    m = folium.Map(location=[ctr_lat, ctr_lon], zoom_start=15, tiles="cartodbpositron")

    # --- Rotalar (arac basi renk) ---
    for v, route in enumerate(state.solution.routes):
        if not route:
            continue
        color = VEHICLE_COLORS[v % len(VEHICLE_COLORS)]
        pts = [coords[c] for c in route]
        # bolge-ici bacaklar (optimize edilebilir) - dolu renk
        if len(pts) >= 2:
            folium.PolyLine(
                pts, color=color, weight=3, opacity=0.9,
                tooltip=f"Arac {v + 1} - {len(route)} durak",
            ).add_to(m)
        # deadhead (sabit) - kesikli, soluk: garaj->ilk, son->dokum
        for a, b in ((depot, pts[0]), (pts[-1], dump)):
            folium.PolyLine(
                [a, b], color=color, weight=1.5, opacity=0.35, dash_array="6,8",
            ).add_to(m)

    # --- Konteynerler ---
    for i, (lat, lon) in enumerate(coords):
        ratio = float(fill[i] / volume[i]) if volume[i] else 0.0
        was_visited = bool(visited[i])
        must = bool(state.must_visit[i])
        tip = (
            f"Konteyner {i} - {int(ds.n_bins[i])} bin<br>"
            f"doluluk: {fill[i]:.0f} / {volume[i]} L (%{ratio * 100:.0f})<br>"
            f"{'ZIYARET' if was_visited else 'ATLANDI'}"
            f"{' - must_visit' if must else ''}"
        )
        if was_visited:
            folium.CircleMarker(
                [lat, lon], radius=5, color="#333", weight=1,
                fill=True, fill_color=_fill_color(ratio), fill_opacity=0.95,
                tooltip=tip,
            ).add_to(m)
        else:  # ATLANDI - soluk gri, hikaye burada
            folium.CircleMarker(
                [lat, lon], radius=4, color="#999", weight=1,
                fill=True, fill_color="#cccccc", fill_opacity=0.5,
                tooltip=tip,
            ).add_to(m)

    # --- Garaj + dokum ---
    folium.Marker(
        depot, tooltip="Garaj (kamyon cikisi)",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        dump, tooltip="Dokum sahasi",
        icon=folium.Icon(color="darkred", icon="trash", prefix="fa"),
    ).add_to(m)

    # --- Baslik + gorunum ---
    folium.map.Marker(
        [ctr_lat, ctr_lon],
        icon=folium.DivIcon(html=(
            f'<div style="font-weight:600;font-size:13px;color:#222;'
            f'background:rgba(255,255,255,.75);padding:2px 6px;border-radius:4px;'
            f'display:inline-block;transform:translate(-50%,-260px)">{title}</div>'
        )),
    ).add_to(m)

    if include_depot_dump:
        m.fit_bounds(_bounds([*coords, depot, dump]))
    else:
        m.fit_bounds(_bounds(coords))
    m.get_root().width = "100%"
    m.get_root().height = f"{height}px"
    return m


def map_html(m: folium.Map) -> str:
    """Streamlit components.html icin gomulu HTML (st_folium bagimliligi yok)."""
    return m.get_root().render()
