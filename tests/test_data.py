"""Veri katmani saf fonksiyon testleri (ag gerektirmez)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import load_config
from data.demand import assign_levels, parse_osm_levels
from data.geo import nearest_index


def test_parse_osm_levels() -> None:
    s = pd.Series(["2", "5", None, "abc", "3;4", np.nan, "0", "-1"])
    out = parse_osm_levels(s)
    # gecerli -> deger; None/gecersiz/0/negatif -> NaN
    assert out[0] == 2 and out[1] == 5 and out[4] == 3
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[6]) and np.isnan(out[7])


def test_assign_levels_distribution() -> None:
    cfg = load_config("config.yaml")
    rng = np.random.default_rng(0)
    # OSM'siz binalar: tipe gore araliktan cekilir
    raw = pd.Series([None, None, None, None])
    ctype = np.array(["residential", "residential", "high", "low"])
    out = assign_levels(raw, ctype, cfg, rng)
    lm = cfg.building_model.levels
    assert lm.residential.min <= out[0] <= lm.residential.max
    assert lm.residential.min <= out[1] <= lm.residential.max
    assert lm.commercial.min <= out[2] <= lm.commercial.max
    assert lm.commercial.min <= out[3] <= lm.commercial.max


def test_assign_levels_osm_wins() -> None:
    cfg = load_config("config.yaml")
    rng = np.random.default_rng(0)
    raw = pd.Series(["7", None])
    ctype = np.array(["residential", "residential"])
    out = assign_levels(raw, ctype, cfg, rng)
    assert out[0] == 7  # OSM degeri kullanilir


def test_assign_levels_deterministic() -> None:
    cfg = load_config("config.yaml")
    raw = pd.Series([None] * 50)
    ctype = np.array(["residential"] * 50)
    a = assign_levels(raw, ctype, cfg, np.random.default_rng(42))
    b = assign_levels(raw, ctype, cfg, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_nearest_index_simple() -> None:
    targets = np.array([[0.0, 0.0], [10.0, 0.0]])
    points = np.array([[1.0, 0.0], [9.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    out = nearest_index(points, targets)
    assert out.tolist() == [0, 1, 0, 1]
