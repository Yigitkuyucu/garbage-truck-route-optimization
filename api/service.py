"""Servis katmani: cekirdek nesneleri API semalarina cevirir (FAZ 3, M.9).

Bu katman **ince**dir ve is mantigi barindirmaz:
  - zorunlu ziyaret karari ve gun cozumu  -> Simulator.solve_day
  - maliyet, yakit, fizibilite            -> Evaluator (DayState.result)
  - durum tasima, CSV, durak listesi      -> sim.operations

Burada yalnizca **bicimleme** yapilir. Ozellikle hicbir maliyet yeniden
hesaplanmaz (Kural B); CO2 yakittan turetilen bir metrik oldugu icin burada
carpilir, amac fonksiyonunun parcasi degildir.
"""

from __future__ import annotations

import numpy as np

from api.schemas import (
    Bootstrap,
    ContainerInfo,
    FillSummary,
    Plan,
    PlanKpi,
    RouteLeg,
    StateSummary,
    StopRow,
)
from api.session import get_config, get_dataset
from sim.engine import DayState
from sim.operations import OperationalState, stop_list

ML_TO_L = 1000

SOLVERS = [
    # Notlar OLCUME dayanir (10 seed x 90 gun; sabit rotaya karsi yakit):
    #   60 sn -> B2 -%24, B1 -%18, X1 -%14, X2 -%11
    #   30 sn -> B1 -%19, X1 -%14, B2 -%11 (B2 sure aclıgı ceker, 4 gun cozemedi)
    # B2 sureye COK duyarli, B1 hic duyarli degil. Varsayilan limit bu yuzden 60 sn.
    {"code": "B2", "name": "OR-Tools", "note": "en iyi sonuc - 60 sn verin"},
    {"code": "B1", "name": "Esik + greedy", "note": "aninda, sureden etkilenmez"},
    {"code": "X1", "name": "ABC (temel)", "note": "projenin odak algoritmasi"},
    {"code": "X2", "name": "ABC + yerel arama", "note": "ablasyon varyanti"},
    {"code": "B0", "name": "Sabit rota", "note": "mevcut durum - hepsini toplar"},
]
SOLVER_NAMES = {s["code"]: s["name"] for s in SOLVERS}


def build_bootstrap(default_lambda: float) -> Bootstrap:
    cfg, ds = get_config(), get_dataset()
    return Bootstrap(
        config_hash=cfg.config_hash,
        region=cfg.region.name,
        n_containers=ds.num_containers,
        n_vehicles=cfg.fleet.num_vehicles,
        capacity_l=cfg.vehicle.effective_capacity_l,
        hygiene_cap_days=cfg.constraints.hygiene_cap_days,
        co2_kg_per_l=cfg.fuel.co2_kg_per_l,
        default_lambda=default_lambda,
        solvers=SOLVERS,
        containers=[
            ContainerInfo(id=i, lat=c[0], lon=c[1], volume_l=int(ds.volume_l[i]))
            for i, c in enumerate(ds.container_coords)
        ],
        depot=[ds.depot_coord[0], ds.depot_coord[1]],
        dump=[ds.dump_coord[0], ds.dump_coord[1]],
    )


def build_state_summary(ops: OperationalState) -> StateSummary:
    ds = get_dataset()
    cfg = get_config()
    pct = ops.fill_l / ds.volume_l * 100
    cap = cfg.constraints.hygiene_cap_days
    return StateSummary(
        last_date=ops.last_date,
        mean_fill_pct=round(float(pct.mean()), 1),
        total_fill_l=int(ops.fill_l.sum()),
        full_count=int((pct > 80).sum()),
        near_hygiene=int((ops.days_since >= cap - 1).sum()),
        max_days_waiting=int(ops.days_since.max()) if ops.days_since.size else 0,
        history=list(reversed(ops.history))[:30],
    )


def build_fill_summary(
    fill: np.ndarray, source: str, warnings: list[str]
) -> FillSummary:
    ds = get_dataset()
    pct = fill / ds.volume_l * 100
    return FillSummary(
        source=source,
        total_l=int(fill.sum()),
        mean_pct=round(float(pct.mean()), 1),
        over_capacity=int((pct > 100).sum()),
        warnings=warnings,
    )


def _route_coords(state: DayState) -> list[RouteLeg]:
    """Her arac icin garaj -> konteynerler -> dokum -> garaj koordinat zinciri.

    Duz cizgi (gercek yol geometrisi degil) - bilinen sinirlilik.
    """
    ds = get_dataset()
    depot = [ds.depot_coord[0], ds.depot_coord[1]]
    dump = [ds.dump_coord[0], ds.dump_coord[1]]
    legs: list[RouteLeg] = []
    for v, route in enumerate(state.solution.routes):
        if not route:
            continue
        pts = [depot]
        pts += [[ds.container_coords[c][0], ds.container_coords[c][1]] for c in route]
        pts += [dump, depot]
        legs.append(RouteLeg(vehicle=v + 1, coords=pts))
    return legs


def build_plan(state: DayState, solver_code: str) -> Plan:
    cfg = get_config()
    r = state.result
    fuel_l = r.fuel_ml / ML_TO_L

    # to_dict: 'doluluk_%' gibi tanimlayici olmayan sutun adlari korunur
    # (itertuples onlari _N pozisyonel adlara cevirirdi).
    df = stop_list(state, get_dataset())
    stops = [
        StopRow(
            vehicle=int(row["arac"]),
            order=int(row["sira"]),
            container_id=int(row["konteyner_id"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            fill_l=int(row["doluluk_l"]),
            fill_pct=float(row["doluluk_%"]),
            must_visit=bool(row["zorunlu"]),
            leg_m=int(row["bacak_m"]),
            cumulative_m=int(row["kumulatif_m"]),
            truck_load_l=int(row["kamyon_yuku_l"]),
        )
        for row in df.to_dict("records")
    ]

    visited = np.where(state.visited)[0].tolist()
    skipped = np.where(~state.visited)[0].tolist()
    must = np.where(state.must_visit)[0].tolist()

    return Plan(
        solver=solver_code,
        solver_name=SOLVER_NAMES.get(solver_code, solver_code),
        feasible=r.feasible,
        violations=list(r.violations),
        kpi=PlanKpi(
            fuel_l=round(fuel_l, 2),
            co2_kg=round(fuel_l * cfg.fuel.co2_kg_per_l, 1),
            fuel_travel_l=round(r.fuel_travel_ml / ML_TO_L, 2),
            fuel_stop_l=round(r.fuel_stop_ml / ML_TO_L, 2),
            fuel_compaction_l=round(r.fuel_compaction_ml / ML_TO_L, 2),
            load_term_l=round(r.fuel_load_term_ml / ML_TO_L, 3),
            stops=int(state.visited.sum()),
            skipped=int(r.n_skipped),
            distance_km=round(r.total_distance / 1000, 2),
            intra_km=round(r.intra_distance / 1000, 2),
            collected_l=int(state.collected_l),
            mean_fill_pct=round(state.mean_fill_pct, 1),
            shift_util_pct=round(state.shift_util_pct, 1),
            overflow_events=int(state.overflow_events),
            overflow_l=int(state.overflow_l),
        ),
        routes=_route_coords(state),
        stops=stops,
        visited_ids=visited,
        skipped_ids=skipped,
        must_visit_ids=must,
    )
