"""Veri altyapisi orkestratoru.

Boru hatti:
    load_osm -> projeksiyon -> bina talebi -> konteyner yerlestirme ->
    bina->konteyner atama -> mesafe/sure matrisi -> BuiltDataset (cache)

Sonunda saglik capalari ve okunur ozet.

CLI:  uv run python -m data.build --config config.yaml [--force]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from config import Config, load_config
from data.dataset import BuiltDataset
from data.demand import compute_building_demand
from data.geo import latlon_to_xy, project_buildings, project_graph
from data.matrix import build_matrices
from data.osm_loader import load_osm
from data.placement import build_collection_points, xy_to_latlon

# Kat sayisi cekimi icin alt-seed etiketi (tekrarlanabilirlik)
_LEVELS_SEED_TAG = 101


def _clip_to_study(buildings_proj, cfg: Config, crs: str, urban=None):
    """Binalari calisma alanina kirp.

    clip_mode='urban_polygon' (varsayilan): OSM'in kentsel doku poligonu
    kullanilir. Gerekce olculdu - "merkez" bir daire degildir: 800 m dairesi
    Gelibolu merkezin kentsel dokusunun yalnizca %70'ini, 1500 m %99'unu
    kapsiyordu. Poligon, yapay yaricap secimini tumden ortadan kaldirir.

    clip_mode='circle': study_radius_m yaricapli daire (yedek/karsilastirma).
    Kentsel poligon bulunamazsa da buna dusulur.
    """
    ctr = latlon_to_xy([cfg.region.center], crs)[0]
    if cfg.region.clip_mode == "urban_polygon" and urban is not None and len(urban):
        poly = _main_urban_polygon(urban, crs, ctr)
        if poly is not None:
            import geopandas as gpd

            pts = gpd.GeoSeries(
                gpd.points_from_xy(buildings_proj["cx"], buildings_proj["cy"]), crs=crs
            )
            return buildings_proj[pts.within(poly).to_numpy()].copy()
    d = np.sqrt(
        (buildings_proj["cx"] - ctr[0]) ** 2 + (buildings_proj["cy"] - ctr[1]) ** 2
    )
    return buildings_proj[d <= cfg.region.study_radius_m].copy()


def _main_urban_polygon(urban, crs: str, ctr):
    """Merkezi iceren (yoksa merkeze en yakin) EN BUYUK kentsel poligon.

    Gelibolu'da bu 5,07 km2'lik ana yerlesim dokusudur; geri kalanlar 2-8 km
    uzaklikta koy/site poligonlaridir.
    """
    from shapely.geometry import Point

    g = urban.to_crs(crs)
    g = g[g.geometry.notna() & g.geometry.is_valid]
    if g.empty:
        return None
    c = Point(ctr[0], ctr[1])
    hit = g[g.geometry.contains(c)]
    pool = hit if len(hit) else g
    return pool.loc[pool.geometry.area.idxmax()].geometry


def build_dataset(cfg: Config, *, force: bool = False, save: bool = True) -> BuiltDataset:
    """Veri kumesini insa et (ya da cache'ten yukle).

    save=False: cache'e YAZMA (duyarlilik varyantlari icin - config_hash ham
    metinden geldigi icin model_copy override'lari ana cache'i ezmesin).
    """
    if not force and BuiltDataset.cache_exists(cfg.config_hash):
        return BuiltDataset.load(cfg.config_hash)

    osm = load_osm(cfg.region)
    graph_proj, crs = project_graph(osm.graph)
    buildings_proj = project_buildings(osm.buildings, crs)
    buildings_proj = _clip_to_study(buildings_proj, cfg, crs, osm.urban)

    # Bina basi talep + siniflama (kat cekimi seed'den turetilir)
    rng = np.random.default_rng([cfg.seed, _LEVELS_SEED_TAG])
    bdem = compute_building_demand(buildings_proj, cfg, crs, rng)

    # Talep-dengeli toplama noktalari (her nokta = tek dugum)
    pts = build_collection_points(bdem, cfg)
    cont_xy = pts[["cx", "cy"]].to_numpy(dtype=np.float64)

    # Matris icin nokta sirasi: garaj, konteynerler, dokum
    depot_xy = latlon_to_xy([cfg.depot.coord], crs)
    dump_xy = latlon_to_xy([cfg.dump_site.coord], crs)
    points_xy = np.vstack([depot_xy, cont_xy, dump_xy])

    mats = build_matrices(graph_proj, points_xy)

    ds = BuiltDataset(
        config_hash=cfg.config_hash,
        region_key=osm.region_key,
        container_source="collection_points",
        depot_coord=cfg.depot.coord,
        dump_coord=cfg.dump_site.coord,
        container_coords=xy_to_latlon(cont_xy, crs),
        base_rate_l=pts["base_l"].to_numpy(dtype=np.float64),
        volume_l=pts["volume_l"].to_numpy(dtype=np.int64),
        n_bins=pts["n_bins"].to_numpy(dtype=np.int64),
        residents=pts["residents"].to_numpy(dtype=np.float64),
        commercial_type=pts["commercial_type"].tolist(),
        market_day=[None if v is None else str(v) for v in pts["market_day"].tolist()],
        dist_m=mats.dist_m,
        time_s=mats.time_s,
        node_ids=mats.node_ids,
    )
    if save:
        ds.save()
    return ds


def anchors(cfg: Config, ds: BuiltDataset) -> dict[str, float]:
    """Uc capa + nufus. Sayisal degerleri dondur."""
    total_daily = float(ds.base_rate_l.sum())
    cap = cfg.vehicle.effective_capacity_l
    per_bin_fill = ds.base_rate_l / np.maximum(ds.n_bins, 1)
    # gun/dolus = nokta hacmi / gunluk talep (talebi 0 olmayan noktalar)
    nz = ds.base_rate_l > 0
    days_to_fill = ds.volume_l[nz] / ds.base_rate_l[nz]
    pop = float(ds.residents.sum())
    total_bins = int(ds.n_bins.sum())
    val = cfg.validation
    return {
        "trucks": float(math.ceil(total_daily / cap)),
        "trucks_expected": val.expected_trucks(pop),
        "total_daily": total_daily,
        "mean_bin_fill": float(per_bin_fill.mean()),
        "max_bin_fill": float(per_bin_fill.max()),
        "mean_days_to_fill": float(days_to_fill.mean()),
        "min_days_to_fill": float(days_to_fill.min()),
        "population": pop,
        "total_bins": float(total_bins),
        "bins_expected": val.expected_containers(pop),
        "people_per_bin": pop / total_bins if total_bins else math.inf,
    }


def days_to_fill_per_bin(ds: BuiltDataset) -> np.ndarray:
    """Bin basi gun/dolus dizisi (nokta gun/dolusu n_bin kez tekrar). Talep>0."""
    nz = ds.base_rate_l > 0
    point_days = ds.volume_l[nz] / ds.base_rate_l[nz]
    counts = ds.n_bins[nz]
    return np.repeat(point_days, counts)


def days_to_fill_percentiles(ds: BuiltDataset) -> dict[str, float]:
    """Bin basi gun/dolus dagilimi: min, p10, p25, medyan, p75, p90, maks."""
    d = days_to_fill_per_bin(ds)
    qs = np.percentile(d, [0, 10, 25, 50, 75, 90, 100])
    keys = ["min", "p10", "p25", "median", "p75", "p90", "max"]
    return dict(zip(keys, (float(x) for x in qs), strict=True))


def deadhead_legs(ds: BuiltDataset) -> dict[str, int]:
    """Sabit maliyet (deadhead) bacaklari (m) - garaj<->bolge<->dokum.

    Bolge ici rotalamayla optimize EDILEMEZ (KPI ayrimi; kullanici karari).
    """
    depot, dump = ds.depot_index, ds.dump_index
    cont = [ds.container_index(i) for i in range(ds.num_containers)]
    d = ds.dist_m
    approach = int(min(d[depot, c] for c in cont))      # garaj -> en yakin nokta
    egress = int(min(d[c, dump] for c in cont))         # en yakin nokta -> dokum
    ret = int(d[dump, depot])                           # dokum -> garaj
    return {
        "approach": approach,
        "egress": egress,
        "return": ret,
        "fixed_round_trip": approach + egress + ret,
        "depot_dump_direct": int(d[depot, dump]),
    }


def health_report(cfg: Config, ds: BuiltDataset) -> list[str]:
    """Uc capa (kullanicinin tanimi) + nufus. Uyarilari dondur."""
    a = anchors(cfg, ds)
    tol = cfg.validation.anchor_tolerance
    warnings: list[str] = []

    # 1) Kamyon capasi (referans oran uzerinden olceklenir)
    exp_t = a["trucks_expected"]
    if exp_t > 0 and not (1.0 / tol <= a["trucks"] / exp_t <= tol):
        warnings.append(
            f"KAMYON CAPASI: {a['trucks']:.0f} vs beklenen {exp_t:.1f} "
            f"({cfg.validation.reference_trucks} x nufus/"
            f"{cfg.validation.reference_population:,}). calisma alanini ayarla."
        )

    # 2) Bin dolus
    if a["mean_bin_fill"] > cfg.containers.volume_l:
        warnings.append(
            f"BIN DOLUS: ort {a['mean_bin_fill']:,.0f} L > bin hacmi "
            f"{cfg.containers.volume_l} L. Yerlestirme yetersiz."
        )

    # 3) Konteyner sayisi capasi (referans kisi/konteyner orani)
    exp_c = a["bins_expected"]
    if exp_c > 0 and not (1.0 / tol <= a["total_bins"] / exp_c <= tol):
        warnings.append(
            f"KONTEYNER CAPASI: {a['total_bins']:.0f} bin vs beklenen {exp_c:.0f} "
            f"({cfg.validation.people_per_container:.0f} kisi/konteyner orani). "
            f"Model {a['people_per_bin']:.0f} kisi/bin uretti."
        )

    # 4) gun/dolus dagilimi - per-container must_visit'i (C4a) besler
    p = days_to_fill_percentiles(ds)
    cap = cfg.constraints.hygiene_cap_days
    warnings.append(
        f"GUN/DOLUS DAGILIMI: medyan={p['median']:.1f} p90={p['p90']:.1f} maks={p['max']:.1f}. "
        f"must_visit DOLULUK-FARKINDALIKLI (C4b): yarin tasacaksa VEYA "
        f"gun_sayisi>={cap} ise topla; yalnizca riskli konteyner zorlanir."
    )

    tuik = cfg.validation.tuik_population
    warnings.append(
        f"NUFUS (bilgi): calisma alani {a['population']:,.0f} kisi vs TUIK merkez "
        f"{tuik:,} (%{100 * a['population'] / tuik:.0f})."
    )
    return warnings


def _print_histogram(data: np.ndarray, max_wait: int) -> None:
    """ASCII histogram: gun/dolus kovalari. must_visit'i asan kovalari isaretler."""
    edges = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 7, np.inf]
    labels = ["<1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-4", "4-5", "5-7", "7+"]
    counts, _ = np.histogram(data, bins=edges)
    peak = max(int(counts.max()), 1)
    total = len(data)
    los = edges[:-1]
    for lbl, c, lo in zip(labels, counts, los, strict=True):
        bar = "#" * round(40 * c / peak)
        # lo >= must_visit: must_visit'ten yavas dolar -> guvenle atlanabilir (skip firsati)
        flag = "  <- skip firsati (yavas)" if lo >= max_wait and c > 0 else ""
        print(f"    {lbl:>6} gun | {bar:<40} {c:>4} ({100 * c / total:4.1f}%){flag}")


def print_summary(cfg: Config, ds: BuiltDataset) -> None:
    import pandas as pd

    n = ds.num_containers
    a = anchors(cfg, ds)
    cap = cfg.vehicle.effective_capacity_l
    types = pd.Series(ds.commercial_type).value_counts()
    n_market = sum(1 for m in ds.market_day if m is not None)

    print("=" * 64)
    print(f"  VERI ALTYAPISI OZETI - {cfg.region.name} (study r={cfg.region.study_radius_m}m)")
    print("=" * 64)
    print(f"  config_hash        : {ds.config_hash}")
    print(f"  toplama noktasi    : {n}   (toplam bin: {int(ds.n_bins.sum())})")
    k = ds.dist_m.shape[0]
    print(f"  matris boyutu      : {k} x {k} (garaj + {n} + dokum)")
    print("-" * 64)
    tuik = cfg.validation.tuik_population
    vol = cfg.containers.volume_l
    print(f"  calisma alani nufus: {a['population']:,.0f}  (TUIK merkez: {tuik:,})")
    print(f"  gunluk toplam talep: {a['total_daily']:,.0f} L")
    print(f"  etkin kapasite     : {cap:,} L / kamyon")
    print("-" * 64)
    val = cfg.validation
    print(f"  CAPALAR (referans olcek, MERKEZ: {val.reference_containers} kont / "
          f"{val.reference_trucks} kamyon / {val.reference_population:,} kisi):")
    print(f"   1) kamyon          : {a['trucks']:.0f}     "
          f"(beklenen {a['trucks_expected']:.1f} = {val.reference_trucks} x "
          f"nufus/{val.reference_population:,})")
    print(f"   2) konteyner (bin) : {a['total_bins']:.0f}   "
          f"(beklenen {a['bins_expected']:.0f} @ {val.people_per_container:.0f} kisi/kont) "
          f"-> {a['people_per_bin']:.0f} kisi/bin")
    print(f"   3) bin dolus ort   : {a['mean_bin_fill']:,.0f} L  "
          f"maks {a['max_bin_fill']:,.0f} L  (hacim {vol})")
    print(f"   4) gun/dolus ort   : {a['mean_days_to_fill']:.2f}")
    print("-" * 64)
    p = days_to_fill_percentiles(ds)
    print("  GUN/DOLUS DAGILIMI (bin basi - skip firsati YAVAS kuyruktan):")
    print(f"    min={p['min']:.2f}  p10={p['p10']:.2f}  p25={p['p25']:.2f}  "
          f"medyan={p['median']:.2f}  p75={p['p75']:.2f}  p90={p['p90']:.2f}  "
          f"maks={p['max']:.2f}")
    _print_histogram(days_to_fill_per_bin(ds), cfg.constraints.hygiene_cap_days)
    print("-" * 64)
    legs = deadhead_legs(ds)
    print("  SABIT MALIYET (deadhead, optimize EDILEMEZ; KPI ayrimi):")
    print(f"    garaj->bolge   : {legs['approach'] / 1000:.2f} km")
    print(f"    bolge->dokum   : {legs['egress'] / 1000:.2f} km")
    print(f"    dokum->garaj   : {legs['return'] / 1000:.2f} km")
    print(f"    => sabit tur   : {legs['fixed_round_trip'] / 1000:.2f} km / kamyon-seferi")
    print("-" * 64)
    print(f"  bin/nokta          : ort {ds.n_bins.mean():.1f}  maks {int(ds.n_bins.max())}")
    intra = ds.dist_m[1:n + 1, 1:n + 1]
    print(f"  bolge ici ort mesafe: {intra[intra > 0].mean():,.0f} m (optimize edilebilir)")
    print("  nokta turleri      :")
    for t, c in types.items():
        print(f"      {t:<14}: {c}")
    print(f"  pazar noktasi      : {n_market}")
    print("=" * 64)

    for w in health_report(cfg, ds):
        print(f"   - {w}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Veri altyapisi insa (Adim 1)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="cache'i yok say, yeniden insa et")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = build_dataset(cfg, force=args.force)
    print_summary(cfg, ds)


if __name__ == "__main__":
    main()
