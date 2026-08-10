"""Tek kullanicilik oturum durumu (FAZ 3).

Iki istek arasinda tasinmasi gereken iki gecici sey var:

    pending_fill -> girilmis ama henuz cozulmemis doluluk
    plan         -> cozulmus ama henuz uygulanmamis gun

Bunlar **bellekte** tutulur. Prototip tek kullanicilidir (M.9); veritabani ve
oturum yonetimi kapsam disidir. Sunucu yeniden baslarsa bu gecici durum kaybolur -
KALICI durum (doluluk + bekleme sayaclari) diskteki OperationalState'tedir ve
etkilenmez.

Agir nesneler (dataset, config) surec omru boyunca bir kez kurulur; mesafe
matrisi cache'ten okunur, aga dokunulmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

from config import Config, load_config
from data.build import build_dataset
from data.dataset import BuiltDataset
from sim.engine import DayState, Simulator

FloatArr = npt.NDArray[np.float64]


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config("config.yaml")


@lru_cache(maxsize=1)
def get_dataset() -> BuiltDataset:
    """Cache'li BuiltDataset - aga dokunmaz."""
    return build_dataset(get_config())


@lru_cache(maxsize=1)
def get_simulator() -> Simulator:
    return Simulator(get_dataset(), get_config())


@dataclass
class Session:
    """Istekler arasi gecici durum. Kalici degildir."""

    pending_fill: FloatArr | None = None
    pending_source: str | None = None
    plan: DayState | None = None
    plan_solver: str | None = None
    plan_meta: dict[str, Any] | None = None

    def set_fill(self, fill: FloatArr, source: str) -> None:
        """Yeni doluluk girildi - varsa eski plan GECERSIZ olur."""
        self.pending_fill = fill
        self.pending_source = source
        self.clear_plan()

    def clear_plan(self) -> None:
        self.plan = None
        self.plan_solver = None
        self.plan_meta = None

    def clear_all(self) -> None:
        self.pending_fill = None
        self.pending_source = None
        self.clear_plan()


SESSION = Session()
