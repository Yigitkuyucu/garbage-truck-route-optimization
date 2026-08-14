"""Talep-dengeli toplama noktalari (fix 2 + fix 3).

Her toplama noktasi TEK dugumdur; kapasitesi talebine gore boyutlanir:
    n_bin  = max(1, ceil(nokta_talebi * target_days / bin_hacmi))
    hacim  = n_bin * bin_hacmi
Boylece carsi arkasi "cok-bin kume" tek dugum + buyuk hacim olur; tek noktanin
tasmasi YERLESTIRMEYLE cozulur (olcek kucultmeden). gun/dolus ~= target_days.

Noktalar bina yogunlugu izgarasindan (point_spacing_m) turer; her hucre bir
nokta, agirlikli merkezinde.

KABUK modulu.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd

from config import Config

# Birim donusumu (m3 -> L) - sihirli sayi yasaginin istisnasi
LITERS_PER_M3 = 1000


def build_collection_points(bdem: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Bina talep tablosu -> toplama noktasi (dugum) tablosu.

    Donen sutunlar: cx, cy, base_l, residents, commercial_type, market_day,
    n_bins, volume_l.
    """
    cx = bdem["cx"].to_numpy(dtype=np.float64)
    cy = bdem["cy"].to_numpy(dtype=np.float64)
    w = bdem["base_liters"].to_numpy(dtype=np.float64)
    res = bdem["residents"].to_numpy(dtype=np.float64)
    ctype = bdem["commercial_type"].to_numpy()
    mday = bdem["market_day"].to_numpy()

    spacing = float(cfg.containers.point_spacing_m)
    ix = np.floor((cx - cx.min()) / spacing).astype(np.int64)
    iy = np.floor((cy - cy.min()) / spacing).astype(np.int64)
    cell = ix * (iy.max() + 1) + iy

    vol_bin = cfg.containers.volume_l
    # Bin basi hedef gunluk yuk = referans yogunluk x kisi basi litre.
    bm = cfg.building_model
    per_person_l = bm.kg_per_person_day / bm.waste_density_kg_m3 * LITERS_PER_M3
    per_bin_load = cfg.containers.target_people_per_bin * per_person_l
    # Gurultu guvenlik marji: bin hacmi, yuksek-gurultu gununu karsilasin (B0 tasma=0)
    noise_factor = 1.0 + cfg.containers.noise_sigma_margin * cfg.simulation.daily_noise_sigma

    rows = []
    for cid in np.unique(cell):
        m = cell == cid
        demand = float(w[m].sum())
        wm = w[m]
        if demand > 0:
            px = float(np.average(cx[m], weights=wm))
            py = float(np.average(cy[m], weights=wm))
        else:
            px, py = float(cx[m].mean()), float(cy[m].mean())
        residents = float(res[m].sum())
        dom = _dominant_type(ctype[m], wm)
        mk = _weighted_market(mday[m], wm)
        # Yogunluk bin'i (referans kisi/bin) + gurultu guvenlik bin'i, en yuksegi.
        density_bins = round(demand / per_bin_load)
        safety_bins = math.ceil(demand * noise_factor / vol_bin)
        n_bins = max(1, density_bins, safety_bins)
        rows.append((px, py, demand, residents, dom, mk, n_bins, n_bins * vol_bin))

    cols = ["cx", "cy", "base_l", "residents",
            "commercial_type", "market_day", "n_bins", "volume_l"]
    df = pd.DataFrame(rows, columns=cols)
    # object dtype zorla: pandas string-inference None'u 'nan'a cevirmesin
    df["commercial_type"] = pd.Series([r[4] for r in rows], dtype=object)
    df["market_day"] = pd.Series([r[5] for r in rows], dtype=object)
    return df


def _weighted_argmax(keys: np.ndarray, weights: np.ndarray) -> str | None:
    """Agirlik toplami en yuksek anahtari dondur. Bos -> None."""
    acc: dict[str, float] = {}
    for k, w in zip(keys, weights, strict=True):
        acc[k] = acc.get(k, 0.0) + float(w)
    if not acc:
        return None
    return max(acc, key=lambda x: acc[x])


def _dominant_type(types: np.ndarray, weights: np.ndarray) -> str:
    return _weighted_argmax(types, weights) or "residential"


def _weighted_market(days: np.ndarray, weights: np.ndarray) -> str | None:
    mask = np.array([d is not None for d in days], dtype=bool)
    if not mask.any():
        return None
    return _weighted_argmax(days[mask], weights[mask])


def xy_to_latlon(xy: np.ndarray, crs: str) -> list[tuple[float, float]]:
    """Projeli (x, y) -> (lat, lon) listesi (harita/rapor icin)."""
    if len(xy) == 0:
        return []
    gs = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs).to_crs("EPSG:4326")
    return [(float(p.y), float(p.x)) for p in gs]
