"""API istek/yanit semalari (FAZ 3).

Yalnizca tasima bicimi tanimlar; is mantigi ICERMEZ. Maliyet ve fizibilite
Evaluator'dan, zorunlu ziyaret karari Simulator'dan gelir (Kural B).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# ------------------------------------------------------------------- istekler


class SimulateRequest(BaseModel):
    seed: int = Field(0, ge=0, le=9999)


class SolveRequest(BaseModel):
    solver: str = Field("B2")
    skip_lambda: float = Field(0.1, gt=0, le=100)
    time_limit_sec: int = Field(10, ge=1, le=120)


class ApplyRequest(BaseModel):
    collection_date: date


# -------------------------------------------------------------------- yanitlar


class ContainerInfo(BaseModel):
    id: int
    lat: float
    lon: float
    volume_l: int


class Bootstrap(BaseModel):
    """Arayuz ilk acilista bir kez cagirir: degismeyen her sey."""

    config_hash: str
    region: str
    n_containers: int
    n_vehicles: int
    capacity_l: int
    hygiene_cap_days: int
    co2_kg_per_l: float
    default_lambda: float
    solvers: list[dict[str, str]]
    containers: list[ContainerInfo]
    depot: list[float]
    dump: list[float]


class StateSummary(BaseModel):
    """Kalici operasyonel durumun ozeti."""

    last_date: str | None
    mean_fill_pct: float
    total_fill_l: int
    full_count: int          # doluluk > %80
    near_hygiene: int        # hijyen tavanina 1 gun kalan ya da gecmis
    max_days_waiting: int
    history: list[dict]


class FillSummary(BaseModel):
    """Girilen (henuz cozulmemis) dolulugun ozeti."""

    source: str
    total_l: int
    mean_pct: float
    over_capacity: int       # hacmini asan konteyner sayisi
    warnings: list[str]


class RouteLeg(BaseModel):
    vehicle: int
    coords: list[list[float]]   # [[lat, lon], ...] garaj -> ... -> dokum -> garaj


class StopRow(BaseModel):
    vehicle: int
    order: int
    container_id: int          # -1 = dokum sahasi
    lat: float
    lon: float
    fill_l: int
    fill_pct: float
    must_visit: bool
    leg_m: int
    cumulative_m: int
    truck_load_l: int


class PlanKpi(BaseModel):
    fuel_l: float
    co2_kg: float
    fuel_travel_l: float
    fuel_stop_l: float
    fuel_compaction_l: float
    load_term_l: float
    stops: int
    skipped: int
    distance_km: float
    intra_km: float
    collected_l: int
    mean_fill_pct: float
    shift_util_pct: float
    overflow_events: int
    overflow_l: int


class Plan(BaseModel):
    solver: str
    solver_name: str
    feasible: bool
    violations: list[str]
    kpi: PlanKpi
    routes: list[RouteLeg]
    stops: list[StopRow]
    visited_ids: list[int]
    skipped_ids: list[int]
    must_visit_ids: list[int]
