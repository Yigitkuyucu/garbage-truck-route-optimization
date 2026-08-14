"""Doluluk uretim modeli - simulatorun cop uretim cekirdegi.

    gunluk_dolus(konteyner) = temel_hiz x gun_carpani x (1 + N(0, sigma))
    pazar konteyneri, pazar gununde x market_surge_multiplier (D2)

KABUK modulu. Tekrarlanabilirlik: tek global seed -> alt seed'ler (proje kurali).

Gun 0 = Pazartesi varsayimi (haftalik ritim + pazar gunleri icin).
"""

from __future__ import annotations

import numpy as np

from config import Config

WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


def weekday_of(day: int) -> str:
    """Gun indeksi (0=Pazartesi) -> haftanin gunu adi."""
    return WEEKDAYS[day % 7]


def generate_daily_fills(
    base_rate: np.ndarray,
    market_days: list[str | None],
    cfg: Config,
    rng: np.random.Generator,
    n_days: int,
) -> np.ndarray:
    """(n_days, N) gunluk uretilen litre (>=0). Konteyner basi gurultu bagimsiz.

    NOT: Pazar surge'u UYGULANMAZ (kullanici karari). Gercekte pazar atigi
    kapaniste ozel ekiple toplanir; rutin konteyner optimizasyonunun konusu
    degildir (rapor sinirlilik). Pazar konteynerleri normal ticari talebiyle
    girer. market_days parametresi ileride kullanim/rapor icin tutulur.
    """
    n = base_rate.shape[0]
    sim = cfg.simulation
    day_mult = np.array(
        [sim.weekday_multipliers[weekday_of(d)] for d in range(n_days)], dtype=np.float64
    )
    noise = rng.normal(0.0, sim.daily_noise_sigma, size=(n_days, n))
    return base_rate[None, :] * day_mult[:, None] * np.maximum(0.0, 1.0 + noise)


def derive_seeds(global_seed: int, num_seeds: int) -> list[np.random.Generator]:
    """Global seed -> bagimsiz alt uretecler (tekrarlanabilir)."""
    seqs = np.random.SeedSequence(global_seed).spawn(num_seeds)
    return [np.random.default_rng(s) for s in seqs]


def peak_daily_total(
    base_rate: np.ndarray, market_days: list[str | None], cfg: Config
) -> tuple[float, float]:
    """Tum seed'ler + tum ufuk (warmup+report) uzerinde gunluk toplam uretimin
    (maksimum, ortalama) degerini dondur - filo boyutlandirma icin (C5+C8)."""
    sim = cfg.simulation
    n_days = sim.warmup_days + sim.report_days
    peak = 0.0
    totals: list[float] = []
    for rng in derive_seeds(cfg.seed, sim.num_seeds):
        fills = generate_daily_fills(base_rate, market_days, cfg, rng, n_days)
        day_totals = fills.sum(axis=1)
        totals.append(float(day_totals.mean()))
        peak = max(peak, float(day_totals.max()))
    return peak, float(np.mean(totals))
