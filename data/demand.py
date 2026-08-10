"""Talep turetme: bina -> sakin -> gunluk litre.

KABUK modulu.

Zincir:
    taban alani x kat sayisi           -> kat alani (m2)
    kat alani / m2_per_person          -> sakin sayisi
    sakin x kg_per_person_day / yogunluk x 1000 -> gunluk litre

Ticari/pazar siniflama: OSM etiketi tum binalarda 'building=yes'
oldugu icin konut/ticari ayrimi commercial_points yaricapindan yapilir.

>>> MODELLEME KARARI (kullaniciya dogrulatilacak) <<<
Ticari katsayi, binanin taban-litresine CARPAN olarak uygulanir (residential=1.0).
Gerekce: kucuk dukkan fiziksel olarak kucuk -> taban litresi zaten dusuk; buyuk
apartman -> taban litresi yuksek. Boylece "apartmanlar baskin" (B5c niyeti) korunur
ve katsayi "birim alan basina konuta gore ne kadar cop" anlamina gelir.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from config import Config, DemandCoefficients
from data.geo import latlon_to_xy, segments_to_xy

# Birim donusumu - sihirli sayi yasaginin istisnasi
KG_M3_TO_LITERS = 1000

# Ticari tur siralamasi: ortusme halinde en yuksek gecerli
_TYPE_RANK = {"residential": 0, "low": 1, "mid": 2, "high": 3}


def parse_osm_levels(raw: pd.Series) -> np.ndarray:
    """OSM 'building:levels' -> float dizi; bos/gecersiz -> NaN."""
    out = np.full(len(raw), np.nan, dtype=np.float64)
    for i, val in enumerate(raw.to_numpy()):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        s = str(val).split(";")[0].split(",")[0].strip()
        try:
            lv = int(float(s))
        except (ValueError, TypeError):
            continue
        if lv > 0:
            out[i] = float(lv)
    return out


def assign_levels(
    raw: pd.Series, ctype: np.ndarray, cfg: Config, rng: np.random.Generator
) -> np.ndarray:
    """Kat sayisi dizisi (fix 1). OSM degeri varsa o; yoksa TIPE gore dagilimdan.

    Konut: uniform[res.min, res.max] (ort 4). Ticari (low/mid/high):
    uniform[com.min, com.max] (ort 2). Seed'den turetilen deterministik cekim.
    """
    lm = cfg.building_model.levels
    n = len(raw)
    osm = parse_osm_levels(raw)

    res = rng.integers(lm.residential.min, lm.residential.max + 1, size=n)
    com = rng.integers(lm.commercial.min, lm.commercial.max + 1, size=n)
    is_res = ctype == "residential"
    drawn = np.where(is_res, res, com).astype(np.int64)

    has_osm = ~np.isnan(osm)
    out = np.where(has_osm, np.nan_to_num(osm).astype(np.int64), drawn)
    return out.astype(np.int64)


def _coeff_for(coeffs: DemandCoefficients, ctype: str) -> float:
    return {
        "residential": coeffs.residential,
        "low": coeffs.commercial_low,
        "mid": coeffs.commercial_mid,
        "high": coeffs.commercial_high,
    }[ctype]


def classify_commercial(
    centroids: np.ndarray, cfg: Config, crs: str
) -> np.ndarray:
    """Her bina icin ticari tur ('residential'|'low'|'mid'|'high').

    Bir commercial_point yaricapi icindeki binalar o turu kazanir; ortusmede en
    yuksek rank gecerli.
    """
    types = np.array(["residential"] * len(centroids), dtype=object)
    if len(centroids) == 0:
        return types
    ranks = np.zeros(len(centroids), dtype=np.int64)
    for cp in cfg.commercial_points:
        cx = latlon_to_xy([cp.coord], crs)[0]
        d2 = np.sum((centroids - cx) ** 2, axis=1)
        inside = d2 <= float(cp.radius_m) ** 2
        r = _TYPE_RANK[cp.type]
        take = inside & (r > ranks)
        types[take] = cp.type
        ranks[take] = r
    return types


def flag_market(centroids: np.ndarray, cfg: Config, crs: str) -> np.ndarray:
    """Her bina icin pazar gunu (str) ya da None. Pazar koridoru yaricapindaysa."""
    days = np.array([None] * len(centroids), dtype=object)
    if len(centroids) == 0:
        return days
    pts = gpd.GeoSeries([Point(x, y) for x, y in centroids], crs=crs)
    for mz in cfg.market_zones:
        lines = segments_to_xy(mz.segments, crs)
        within = np.zeros(len(centroids), dtype=bool)
        for line in lines:
            within |= pts.distance(line).to_numpy() <= float(mz.radius_m)
        # Ilk atanan pazar gunu kalir (koridorlar cakismaz varsayimi)
        assign = within & (days == None)  # noqa: E711
        days[assign] = mz.day
    return days


def compute_building_demand(
    buildings_proj: gpd.GeoDataFrame,
    cfg: Config,
    crs: str,
    rng: np.random.Generator,
    coeffs: DemandCoefficients | None = None,
) -> pd.DataFrame:
    """Bina basi talep tablosu. Sutunlar:
    cx, cy, floor_area, residents, commercial_type, market_day, base_liters.

    base_liters: gunluk litre (ticari carpan uygulanmis, pazar HARIC - pazar
    surge simulasyonda uygulanir).
    """
    bm = cfg.building_model
    coeffs = coeffs or cfg.demand_coefficients

    centroids = np.column_stack(
        [buildings_proj["cx"].to_numpy(), buildings_proj["cy"].to_numpy()]
    ).astype(np.float64)
    ctype = classify_commercial(centroids, cfg, crs)
    mday = flag_market(centroids, cfg, crs)

    area = buildings_proj["area_m2"].to_numpy(dtype=np.float64)
    levels = assign_levels(buildings_proj["building:levels"], ctype, cfg, rng)
    floor_area = area * levels
    residents = floor_area / bm.m2_per_person
    resident_liters = (
        residents * bm.kg_per_person_day / bm.waste_density_kg_m3 * KG_M3_TO_LITERS
    )

    mult = np.array([_coeff_for(coeffs, t) for t in ctype], dtype=np.float64)
    base_liters = resident_liters * mult

    df = pd.DataFrame(
        {
            "cx": centroids[:, 0],
            "cy": centroids[:, 1],
            "floor_area": floor_area,
            "residents": residents,
            "commercial_type": pd.Series(ctype, dtype=object),
            "base_liters": base_liters,
        }
    )
    # market_day: object dtype zorla; pandas string-inference None'u 'nan'a cevirmesin
    df["market_day"] = pd.Series(mday, dtype=object)
    return df
