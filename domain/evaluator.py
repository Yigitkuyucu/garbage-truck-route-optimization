"""Evaluator - TEK DOGRULUK KAYNAGI.

Maliyet formulu ve fizibilite kontrolu YALNIZCA burada yasar. Hicbir cozucu
kendi maliyetini hesaplamaz.

FAZ 2 AMAC FONKSIYONU - birim: TAM SAYI MILILITRE yakit:

    toplam_maliyet = Sigma yakit(yuk_i, mesafe_i)      <- YUK DUYARLI
                   + n_durak * dur_kalk
                   + Sigma sikistirma(toplanan_litre)
                   + Sigma skip_penalty(atlanan)

    oran(kutle) = base_ml_per_m + slope_ml_per_m_per_kg * kutle_kg

Kamyon garajdan BOS cikar, konteynerlerde YUKLENIR, dokumde BOSALIR:
    garaj -> ilk konteyner : yuk 0
    konteyner -> konteyner : o ana kadar BIRIKEN yuk   <- siralamaya duyarli
    son konteyner -> dokum : toplam yuk
    dokum -> garaj         : yuk 0

> M.7 UYARISI: siralama kaldiraci bu olcekte KUCUK (ulasilamaz tavan ~%0.5-1).
> Yuk terimi fiziksel dogruluk icin modelde; manset ona dayanmaz.

Yuvarlama: BACAK BASINA int (Bolum 5 "tam sayi, istisnasiz"). Bacak gruplamasi
OR-Tools'un ark gruplamasiyla BIREBIR ayni (capraz dogrulama kesin olsun diye -
dokum bacagi donus-arkina katlanir; bkz. ortools_solver.py).

Sert kisitlar (ihlal = infeasible):
    1. arac yuku <= kapasite
    2. rota suresi (seyahat + servis) <= vardiya limiti
    3. must_visit konteynerleri atlanamaz

Ek KPI ayrimi (kullanici F-KPI karari): mesafe = SABIT (deadhead: garaj->bolge->
dokum->garaj) + BOLGE-ICI (konteynerler arasi; optimize edilebilir). Faz 2'de
mesafe YAN metriktir; manset yakit/CO2 (CO2 kabukta yakittan turetilir).

Sicak yol duz dizilerle calisir (numba-hazir); Evaluator sinifi debug sarmalayicisidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.problem import VRPProblem
from domain.solution import Solution

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class EvalResult:
    """Bir cozumun degerlendirme sonucu. Yakit buyuklukleri mL (tam sayi)."""

    feasible: bool
    total_cost: int          # yakit + skip_penalty (AMAC FONKSIYONU, mL)
    fuel_ml: int             # seyahat + dur-kalk + sikistirma (GERCEK yakit)
    fuel_travel_ml: int      # yuk duyarli seyahat yakiti
    fuel_stop_ml: int        # dur-kalk (durak sayisiyla olcekli)
    fuel_compaction_ml: int  # sikistirma (toplanan litreyle olcekli)
    fuel_constload_ml: int   # SABIT-YUK karsiligi (B2 capraz dogrulamasi, M.7)
    total_distance: int      # Sigma rota mesafesi (yan metrik)
    fixed_distance: int      # deadhead (garaj<->bolge<->dokum)
    intra_distance: int      # bolge ici (optimize edilebilir)
    skip_cost: int           # Sigma skip_penalty(atlanan) - mL biriminde
    n_skipped: int
    n_visited: int
    loads: IntArr            # (V,) arac basi yuk (litre)
    times: IntArr            # (V,) arac basi sure (saniye)
    distances: IntArr        # (V,) arac basi mesafe (metre)
    violations: tuple[str, ...]  # debug (bos ise fizibil)

    @property
    def fuel_load_term_ml(self) -> int:
        """Yakitin YALNIZCA yuk baglasimindan gelen kismi (M.7 negatif bulgusu).

        Sabit-yuk karsiligiyla fark: OR-Tools'un erisemedigi terim tam olarak budur.
        """
        return self.fuel_ml - self.fuel_constload_ml


def _route_metrics(
    containers: IntArr,
    p: VRPProblem,
) -> tuple[int, int, int, int, int, int, int]:
    """Tek rota: (mesafe, fixed, intra, yuk_l, sure, yakit_seyahat, yakit_sabit_yuk).

    Bos rota -> hepsi sifir. Yakit bacak basina yuvarlanir; bacak gruplamasi
    OR-Tools ile ayni (dokum bacagi + dokum->garaj TEK ark olarak yuvarlanir).
    """
    if containers.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0, 0
    dist, tt = p.dist, p.travel_time
    depot, dump = p.depot_index, p.dump_index
    base, slope, nominal = (
        p.fuel_base_ml_per_m, p.fuel_slope_ml_per_m_per_kg, p.nominal_ml_per_m
    )
    nodes = containers + 1  # konteyner -> matris dugumu
    first, last = int(nodes[0]), int(nodes[-1])

    # --- mesafe ---
    d_in = int(dist[depot, first])
    d_dump, d_home = int(dist[last, dump]), int(dist[dump, depot])
    d_out = d_dump + d_home
    fixed = d_in + d_out
    intra_legs = dist[nodes[:-1], nodes[1:]] if nodes.shape[0] > 1 else _EMPTY_INT
    intra = int(intra_legs.sum())
    route_dist = fixed + intra

    # --- yuk profili: bacak basina BIRIKEN kutle (siralamaya duyarli) ---
    cum_mass = np.cumsum(p.mass_kg[containers])          # (k,) i. konteyner sonrasi
    carried = cum_mass[:-1] if cum_mass.shape[0] > 1 else _EMPTY_FLOAT  # ic bacaklar
    total_mass = float(cum_mass[-1])

    # --- yakit (bacak basina yuvarlama) ---
    # garaj->ilk: kamyon BOS ; dokum bacagi: TAM yuk ; dokum->garaj: bosalmis
    fuel_travel = round(base * d_in)
    fuel_travel += int(np.rint((base + slope * carried) * intra_legs).sum())
    fuel_travel += round((base + slope * total_mass) * d_dump)
    fuel_travel += round(base * d_home)

    # sabit-yuk karsiligi: OR-Tools'un cozdugu yuk-kor yaklasim (ayni gruplama)
    fuel_const = round(nominal * d_in)
    fuel_const += int(np.rint(nominal * intra_legs).sum())
    fuel_const += round(nominal * d_out)   # dokum bacagi donus-arkina katli

    # --- sure ---
    tt_intra = int(tt[nodes[:-1], nodes[1:]].sum()) if nodes.shape[0] > 1 else 0
    tt_fixed = int(tt[depot, first]) + int(tt[last, dump]) + int(tt[dump, depot])
    route_time = tt_fixed + tt_intra + int(p.service_time[containers].sum())
    load = int(p.demand[containers].sum())
    return route_dist, fixed, intra, load, route_time, fuel_travel, fuel_const


_EMPTY_INT: IntArr = np.zeros(0, dtype=np.int64)
_EMPTY_FLOAT: FloatArr = np.zeros(0, dtype=np.float64)


class Evaluator:
    """Tek dogruluk kaynagi. debug=True ihlalde arac/dugum/kisit detayi verir."""

    def __init__(self, problem: VRPProblem, *, debug: bool = False) -> None:
        self._p = problem
        self._debug = debug

    def evaluate(self, solution: Solution) -> EvalResult:
        p = self._p
        n = p.n_containers

        loads: list[int] = []
        times: list[int] = []
        dists: list[int] = []
        total_distance = fixed_total = intra_total = 0
        fuel_travel = fuel_const_travel = 0
        visited = np.zeros(n, dtype=np.int64)
        violations: list[str] = []

        for v, route in enumerate(solution.routes):
            containers = np.asarray(route, dtype=np.int64)
            visited[containers] += 1
            rd, fx, intra, load, rt, f_tr, f_const = _route_metrics(containers, p)
            dists.append(rd)
            loads.append(load)
            times.append(rt)
            total_distance += rd
            fixed_total += fx
            intra_total += intra
            fuel_travel += f_tr
            fuel_const_travel += f_const
            if load > p.capacity:
                violations.append(
                    f"arac {v}: yuk {load} > kapasite {p.capacity} (litre)"
                )
            if rt > p.shift_limit:
                violations.append(
                    f"arac {v}: sure {rt} > vardiya {p.shift_limit} (saniye)"
                )

        # Cift ziyaret
        dup = np.where(visited > 1)[0]
        for c in dup:
            violations.append(f"konteyner {int(c)}: {int(visited[c])} kez ziyaret")

        # Atlananlar + skip_penalty + must_visit ihlali
        skipped_mask = visited == 0
        skipped_idx = np.where(skipped_mask)[0]
        skip_cost = int(p.skip_penalty[skipped_mask].sum())
        for c in skipped_idx:
            if p.must_visit[c]:
                violations.append(f"konteyner {int(c)}: must_visit atlandi")

        # Dur-kalk + sikistirma: DUGUM basi (OR-Tools ark gruplamasiyla ayni yuvarlama)
        visited_mask = visited > 0
        n_visited = int(visited_mask.sum())
        fuel_stop = n_visited * p.stop_start_ml
        fuel_compaction = int(
            np.rint(p.compaction_ml_per_liter * p.demand[visited_mask]).sum()
        )

        fuel_ml = fuel_travel + fuel_stop + fuel_compaction
        feasible = len(violations) == 0

        return EvalResult(
            feasible=feasible,
            total_cost=fuel_ml + skip_cost,
            fuel_ml=fuel_ml,
            fuel_travel_ml=fuel_travel,
            fuel_stop_ml=fuel_stop,
            fuel_compaction_ml=fuel_compaction,
            fuel_constload_ml=fuel_const_travel + fuel_stop + fuel_compaction,
            total_distance=total_distance,
            fixed_distance=fixed_total,
            intra_distance=intra_total,
            skip_cost=skip_cost,
            n_skipped=int(skipped_idx.shape[0]),
            n_visited=n_visited,
            loads=np.array(loads, dtype=np.int64),
            times=np.array(times, dtype=np.int64),
            distances=np.array(dists, dtype=np.int64),
            violations=tuple(violations) if self._debug else (),
        )


def pretty_print(
    solution: Solution,
    result: EvalResult,
    problem: VRPProblem,
    names: list[str] | None = None,
) -> str:
    """Cozumu insan okunur hale getirir (zorunlu konfor araci).

    Sayilar Evaluator sonucundan gelir (maliyet hesabi burada TEKRAR EDILMEZ).
    """

    def name(c: int) -> str:
        return names[c] if names is not None else f"k{c}"

    lines = [
        f"COZUM  fizibil={result.feasible}  "
        f"toplam_maliyet={result.total_cost / 1000:,.2f} L yakit-esdeger "
        f"(yakit={result.fuel_ml / 1000:,.2f} + skip={result.skip_cost / 1000:,.2f})",
        f"  yakit kalemleri: seyahat={result.fuel_travel_ml / 1000:,.2f} L  "
        f"dur-kalk={result.fuel_stop_ml / 1000:,.2f} L  "
        f"sikistirma={result.fuel_compaction_ml / 1000:,.2f} L",
        f"  yuk baglasimi (M.7): {result.fuel_load_term_ml / 1000:+,.2f} L  "
        f"(sabit-yuk karsiligina gore)",
        f"  mesafe ayrimi: sabit(deadhead)={result.fixed_distance:,} m  "
        f"bolge-ici={result.intra_distance:,} m",
    ]
    for v, route in enumerate(solution.routes):
        seq = " -> ".join(name(c) for c in route) if route else "(bos)"
        lines.append(
            f"  arac {v}: yuk={int(result.loads[v]):,}/{problem.capacity:,} L  "
            f"sure={int(result.times[v]):,}/{problem.shift_limit:,} sn  "
            f"mesafe={int(result.distances[v]):,} m"
        )
        lines.append(f"      Garaj -> {seq} -> Dokum -> Garaj")
    if result.n_skipped:
        lines.append(f"  atlanan: {result.n_skipped} konteyner")
    if result.violations:
        lines.append("  IHLALLER:")
        lines.extend(f"    - {v}" for v in result.violations)
    return "\n".join(lines)
