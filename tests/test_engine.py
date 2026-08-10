"""Simulasyon motoru testleri - kutle korunumu, tasma, must_visit, warm-up."""

from __future__ import annotations

import numpy as np
import pytest

from config import Config, load_config
from data.dataset import BuiltDataset
from sim.engine import Simulator
from solvers.greedy import GreedySolver
from solvers.threshold_greedy import ThresholdGreedySolver

POS = [0, 1, 2, 3, 10]
DIST = np.array([[abs(a - b) for b in POS] for a in POS], dtype=np.int64)


def make_dataset(base_rate: list[float], volume: list[int]) -> BuiltDataset:
    n = len(base_rate)
    return BuiltDataset(
        config_hash="test",
        region_key="test",
        container_source="test",
        depot_coord=(0.0, 0.0),
        dump_coord=(0.0, 0.0),
        container_coords=[(0.0, 0.0)] * n,
        base_rate_l=np.array(base_rate, dtype=np.float64),
        volume_l=np.array(volume, dtype=np.int64),
        n_bins=np.ones(n, dtype=np.int64),
        residents=np.zeros(n, dtype=np.float64),
        commercial_type=["residential"] * n,
        market_day=[None] * n,
        dist_m=DIST,
        time_s=DIST.copy(),
        node_ids=np.arange(n + 2, dtype=np.int64),
    )


def small_cfg(**over: int) -> Config:
    cfg = load_config("config.yaml")
    c = cfg.model_copy(deep=True)
    object.__setattr__(c.simulation, "warmup_days", over.get("warmup", 2))
    object.__setattr__(c.simulation, "report_days", over.get("report", 5))
    object.__setattr__(c.simulation, "num_seeds", 1)
    object.__setattr__(c.simulation, "daily_noise_sigma", 0.0)
    if "vehicles" in over:
        object.__setattr__(c.fleet, "num_vehicles", over["vehicles"])
    if "cap" in over:
        object.__setattr__(c.constraints, "hygiene_cap_days", over["cap"])
    return c


def test_mass_conservation_b0() -> None:
    ds = make_dataset([100, 100, 100], [1100, 1100, 1100])
    cfg = small_cfg()
    sim = Simulator(ds, cfg)
    res = sim.run(GreedySolver(cfg.fleet.num_vehicles), np.random.default_rng(0), 1.0)
    assert res.mass_ok
    # uretilen == toplanan + kalan
    assert res.total_generated_l == res.total_collected_l + res.remaining_l


def test_b0_no_overflow() -> None:
    # B0 her gun hepsini toplar -> fill birikmez -> tasma yok
    ds = make_dataset([100, 100, 100], [1100, 1100, 1100])
    cfg = small_cfg()
    sim = Simulator(ds, cfg)
    res = sim.run(GreedySolver(cfg.fleet.num_vehicles), np.random.default_rng(0), 1.0)
    assert res.total_overflow_events == 0
    assert all(r.overflow_events == 0 for r in res.records)


def test_warmup_discarded() -> None:
    ds = make_dataset([100, 100, 100], [1100, 1100, 1100])
    cfg = small_cfg(warmup=3, report=7)
    sim = Simulator(ds, cfg)
    res = sim.run(GreedySolver(cfg.fleet.num_vehicles), np.random.default_rng(0), 1.0)
    assert len(res.records) == 7
    assert res.records[0].day == 0


def test_must_visit_triggers_under_skipping_b1() -> None:
    # B1 esik 0.99: hicbir konteyner gunde %99'a ulasmaz -> per-container must_visit tetikler
    ds = make_dataset([100, 100, 100], [1100, 1100, 1100])  # gun_dolus=11 -> must_visit=cap
    cfg = small_cfg(report=10, cap=3)
    sim = Simulator(ds, cfg)
    res = sim.run(
        ThresholdGreedySolver(cfg.fleet.num_vehicles, 0.99), np.random.default_rng(0), 1.0
    )
    # per-container must_visit (=3) sayesinde birikim sinirli -> tasma yok, kutle korunur
    assert res.mass_ok
    assert res.total_overflow_events == 0
    assert any(r.n_visited > 0 for r in res.records)


def test_per_container_must_visit_fast_container_daily() -> None:
    # Hizli konteyner (gun_dolus kucuk) -> must_visit=1 (gunluk), tasma-guvenli.
    ds = make_dataset([600, 100, 100], [1100, 1100, 1100])  # k0 gun_dolus=1.83 -> floor(/1.2)=1
    cfg = small_cfg(report=8, cap=3)
    sim = Simulator(ds, cfg)
    # k0'i atlayan bir cozucu bile onu her gun toplamak zorunda (must_visit=1)
    res = sim.run(
        ThresholdGreedySolver(cfg.fleet.num_vehicles, 0.99), np.random.default_rng(0), 1.0
    )
    assert res.total_overflow_events == 0  # hizli konteyner gunluk -> tasmaz


def test_feasibility_check_raises_when_fleet_too_small() -> None:
    # Cok yuksek base_rate + tek arac -> en kotu gun kapasiteyi asar
    ds = make_dataset([60000, 60000, 60000], [100000, 100000, 100000])
    cfg = small_cfg(vehicles=1)
    sim = Simulator(ds, cfg)
    with pytest.raises(ValueError, match="FILO YETERSIZ"):
        sim.feasibility_check()
