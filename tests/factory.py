"""Test yardimcisi: VRPProblem kurucusu icin FAZ 2 yakit varsayilanlari.

Varsayilanlar KASTEN notr secildi:

    base = 1.0 mL/m,  slope = 0,  dur-kalk = 0,  sikistirma = 0,  nominal = 1.0
    mass_kg = demand  (1 kg/L - gercekci degil, elle hesap icin yuvarlak)

Bu ayarda **yakit == mesafe** (mL cinsinden) olur. Boylece Faz 1'in elle
hesaplanmis mesafe/maliyet testleri aynen gecerli kalir ve yakit davranisi
AYRI testlerde acikca (sifir olmayan katsayilarla) kurulur.

Yakit modelinin kendisi tests/test_fuel.py'de sinanir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domain.problem import VRPProblem

NEUTRAL_FUEL: dict[str, Any] = {
    "fuel_base_ml_per_m": 1.0,
    "fuel_slope_ml_per_m_per_kg": 0.0,
    "stop_start_ml": 0,
    "compaction_ml_per_liter": 0.0,
    "nominal_ml_per_m": 1.0,
}


def vrp(**kwargs: Any) -> VRPProblem:
    """VRPProblem kur; verilmeyen yakit alanlarini notr varsayilanla doldur."""
    demand = kwargs["demand"]
    kwargs.setdefault("mass_kg", np.asarray(demand, dtype=np.float64))
    for key, value in NEUTRAL_FUEL.items():
        kwargs.setdefault(key, value)
    return VRPProblem(**kwargs)
