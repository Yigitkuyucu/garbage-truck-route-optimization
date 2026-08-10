"""Doluluk uretim modeli (sim/containers.py) testleri."""

from __future__ import annotations

import numpy as np

from config import load_config
from sim.containers import derive_seeds, generate_daily_fills, peak_daily_total, weekday_of


def _cfg_no_noise():
    cfg = load_config("config.yaml")
    c = cfg.model_copy(deep=True)
    object.__setattr__(c.simulation, "daily_noise_sigma", 0.0)
    return c


def test_weekday_of() -> None:
    assert weekday_of(0) == "monday"
    assert weekday_of(1) == "tuesday"
    assert weekday_of(7) == "monday"
    assert weekday_of(13) == "sunday"


def test_generate_shape_and_nonneg() -> None:
    cfg = load_config("config.yaml")
    base = np.array([100.0, 200.0, 300.0])
    fills = generate_daily_fills(base, [None, None, None], cfg, np.random.default_rng(0), 30)
    assert fills.shape == (30, 3)
    assert (fills >= 0).all()


def test_determinism() -> None:
    cfg = load_config("config.yaml")
    base = np.array([100.0, 200.0, 300.0])
    a = generate_daily_fills(base, [None, None, None], cfg, np.random.default_rng(7), 20)
    b = generate_daily_fills(base, [None, None, None], cfg, np.random.default_rng(7), 20)
    assert np.array_equal(a, b)


def test_market_surge_not_applied() -> None:
    # Karar: pazar surge fill modeline GIRMEZ (pazar ayri toplanir).
    cfg = _cfg_no_noise()  # gurultu yok -> deterministik
    base = np.array([100.0, 100.0, 100.0])
    fills = generate_daily_fills(base, [None, "tuesday", None], cfg, np.random.default_rng(0), 3)
    # pazar konteyneri (k1) sali gunu de normal base uretir (surge yok)
    assert fills[0].tolist() == [100.0, 100.0, 100.0]
    assert fills[1].tolist() == [100.0, 100.0, 100.0]


def test_peak_ge_mean() -> None:
    cfg = load_config("config.yaml")
    base = np.array([100.0, 200.0, 300.0])
    peak, mean = peak_daily_total(base, [None, None, "saturday"], cfg)
    assert peak >= mean > 0


def test_derive_seeds_reproducible() -> None:
    a = derive_seeds(123, 3)[0].normal(size=5)
    b = derive_seeds(123, 3)[0].normal(size=5)
    assert np.array_equal(a, b)
