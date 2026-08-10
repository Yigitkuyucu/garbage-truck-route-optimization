"""config.yaml dogrulama testleri."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Config, load_config


def test_load_config() -> None:
    cfg = load_config("config.yaml")
    assert cfg.region.name == "gelibolu_merkez"
    assert cfg.fleet.num_vehicles > 0


def test_effective_capacity() -> None:
    cfg = load_config("config.yaml")
    # 13 m3 * 5 sikistirma * 1000 L/m3 = 65000 L
    assert cfg.vehicle.effective_capacity_l == 65000


def test_config_hash_deterministic() -> None:
    a = load_config("config.yaml")
    b = load_config("config.yaml")
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 12


def test_weekday_multipliers_complete() -> None:
    cfg = load_config("config.yaml")
    assert len(cfg.simulation.weekday_multipliers) == 7


def test_extra_field_forbidden() -> None:
    cfg = load_config("config.yaml")
    data = cfg.model_dump()
    data["surprise_field"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(data)
