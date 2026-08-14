"""Simulasyon motoru - gun dongusu.

Her gun:
  1. Cop uretilir (fill += generate), tasma kontrol edilir (fill > volume)
  2. must_visit hesaplanir (DOLULUK-FARKINDALIKLI, C4b: yarin tasacaksa VEYA
     hijyen tavanina ulastiysa)
  3. build_problem -> solver.solve -> Evaluator (SINIR gecisi; CLAUDE Bolum 4)
  4. Ziyaret edilen konteynerler toplanir (fill -> 0), days_since_visit guncellenir
  5. KPI kaydedilir (warm-up sonrasi)

Sonda: kutle korunumu (uretilen == toplanan + kalan) DOGRULANIR.

KABUK modulu. Cekirdek (build_problem/Evaluator) diziler alir; donusum gunde bir kez.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from config import Config
from data.dataset import BuiltDataset
from domain.evaluator import EvalResult, Evaluator
from domain.problem import VRPProblem, build_problem
from domain.solution import Solution
from sim.containers import generate_daily_fills
from solvers.base import NoFeasibleSolutionError, Solver
from solvers.greedy import GreedySolver


@dataclass(frozen=True)
class DayRecord:
    """Bir gunun KPI'lari (warm-up sonrasi raporlanir)."""

    day: int
    feasible: bool
    # FAZ 2 manset metrikleri (mL; CO2 kabukta yakittan turetilir - M.7)
    fuel_ml: int
    fuel_travel_ml: int
    fuel_stop_ml: int
    fuel_compaction_ml: int
    fuel_load_term_ml: int       # OR-Tools'un erisemedigi terim (M.7 negatif bulgusu)
    # Faz 1 metrikleri - yan metrik olarak KALIR (Bolum 9)
    total_distance: int
    fixed_distance: int
    intra_distance: int
    generated_l: int
    collected_l: int
    n_visited: int               # DURAK sayisi (mansett KPI)
    service_time_s: int          # durak servis suresi (dur-kalk/sikistirma cevrimi)
    n_skipped: int
    overflow_events: int          # o gun fill > volume olan konteyner sayisi
    overflow_l: int               # o gun hacmi asan toplam litre
    mean_fill_pct: float          # ziyaret edilenlerin toplama anindaki ort doluluk
    shift_util_pct: float         # en yuklu aracin vardiya kullanimi
    solver_failed: bool = False   # cozucu cozum bulamadi -> greedy yedegi


@dataclass
class RunResult:
    """Bir cozucu + bir seed icin tum kosunun ozeti."""

    records: list[DayRecord]
    total_generated_l: int
    total_collected_l: int
    remaining_l: int
    mass_ok: bool
    total_overflow_events: int
    infeasible_days: int = 0
    solver_failures: int = 0      # cozucunun cozum bulamadigi gun sayisi
    per_day_details: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class DayState:
    """Bir gunun TAM anlik goruntusu (panel/gorsellestirme icin).

    KPI ozeti (DayRecord) yeterli olmadigi durumda - ozellikle harita, canli
    solve ile o gunku ROTAYI ve ATLANAN konteynerleri cizmek istedigi icin -
    problem/cozum/degerlendirme ve toplama-oncesi doluluk burada saklanir.

    `fill_before`: uretim sonrasi, TOPLAMA ONCESI doluluk (harita rengi = fill/hacim).
    Cekirdek bu sinifi GORMEZ; yalnizca KABUK (sim/ui) kullanir.
    """

    day: int                 # mutlak gun (0..horizon-1)
    report_day: int          # d - warmup (warm-up'ta negatif)
    is_warmup: bool
    fill_before: np.ndarray  # (N,) toplama oncesi doluluk (L)
    demand: np.ndarray       # (N,) rint(fill) - o gunku talep
    must_visit: np.ndarray   # (N,) bool - atlanamaz maske
    volume: np.ndarray       # (N,) nokta hacmi (doluluk orani icin)
    problem: VRPProblem
    solution: Solution
    result: EvalResult
    visited: np.ndarray      # (N,) bool - bu gun ziyaret edilenler
    gen_today_l: int
    collected_l: int
    overflow_events: int
    overflow_l: int
    mean_fill_pct: float
    shift_util_pct: float
    solver_failed: bool = False   # cozucu cozum bulamadi -> greedy yedegi


class Simulator:
    """Bir BuiltDataset + config uzerinde cozucu kosturur."""

    def __init__(self, dataset: BuiltDataset, cfg: Config) -> None:
        self._ds = dataset
        self._cfg = cfg
        self._n = dataset.num_containers
        self._volume = dataset.volume_l.astype(np.int64)
        self._service = (dataset.n_bins * cfg.vehicle.service_time_sec).astype(np.int64)
        self._capacity = cfg.vehicle.effective_capacity_l
        self._shift = cfg.vehicle.shift_seconds
        self._ins_k = cfg.skip_penalty.insertion_cost_k
        # FAZ 2 yakit katsayilari - sinir gecisinde skaler olarak cekirdege gider (M.8)
        self._fuel_kwargs = {
            "kg_per_liter": cfg.building_model.kg_per_liter,
            "fuel_base_ml_per_m": cfg.fuel.base_ml_per_m,
            "fuel_slope_ml_per_m_per_kg": cfg.fuel.slope_ml_per_m_per_kg,
            "stop_start_ml": cfg.fuel.stop_start_ml,
            "compaction_ml_per_liter": cfg.fuel.compaction_ml_per_liter,
            "nominal_load_ratio": cfg.fuel.nominal_load_ratio,
        }
        # DOLULUK-FARKINDALIKLI must_visit (C4b) - sensorun anlik dolulugunu kullanir
        # (projenin tezi: takvim degil, veri). Yarin +k*sigma'da tasacaksa bugun topla.
        self._predict_next = self._ds.base_rate_l * (
            1.0 + cfg.constraints.overflow_predict_sigma_k * cfg.simulation.daily_noise_sigma
        )
        self._hygiene_cap = cfg.constraints.hygiene_cap_days

    def hygiene_cap_override(self, cap: int) -> None:
        """Duyarlilik analizi: hijyen tavanini degistir."""
        self._hygiene_cap = cap

    def feasibility_check(self) -> None:
        """Baslangic fizibilite capasi (C5+C8): en kotu gun <= filo x kapasite.

        Ihlalde HATA verir ve durur (sessizce cop birakmaz)."""
        from sim.containers import peak_daily_total

        peak, _ = peak_daily_total(self._ds.base_rate_l, self._ds.market_day, self._cfg)
        fleet_cap = self._cfg.fleet.num_vehicles * self._capacity
        if peak > fleet_cap:
            raise ValueError(
                f"FILO YETERSIZ: en kotu gun {peak:,.0f} L > filo kapasitesi "
                f"{fleet_cap:,} L ({self._cfg.fleet.num_vehicles} arac). "
                f"num_vehicles artir."
            )

    def _initial_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Baslangic dolulugu + bekleme sayaclari: FAZ-KADEMELI.

        Her konteyner KENDI toplama cevriminde rastgele bir noktada baslar.
        Hicbiri zorunlu dogmaz (faz < 1), dolayisiyla ilk gun toplu bir
        "hepsi birden zorunlu" dalgasi olusmaz.

        Gerekce (olculdu): "hepsi bos baslasin" varsayimi konteynerleri
        SENKRONIZE ediyor; hepsi ayni anda dolup ayni gun zorunlu hale
        geliyordu. Bu, warm-up'ta 7,24 araclik yapay bir tepe uretiyor ve
        filoyu 8'e zorluyordu - oysa raporlanan 90 gunde tepe 6,58 (7 arac).
        Faz-kademeli baslangicta tepe 6,32'ye iner ve filo, belediyenin gercek
        7 aracina oturur. Gercekte de konteynerler bir sabah hep birden bos
        degildir; bu varsayim modelin degil, kolay baslangicin urunuydu.

        Warm-up gunleri zaten atildigi icin raporlanan KPI'lara etkisi minimaldir;
        degistirdigi sey filo boyutlandirmasinin yapay tepeye takilmamasidir.
        """
        n = self._n
        # Zorunlu olana kadar gecen gun: tasma tetikleyicisi ya da hijyen tavani
        head = np.maximum(self._volume - self._predict_next, 0.0)
        to_due = np.minimum(
            self._hygiene_cap, head / np.maximum(self._ds.base_rate_l, 1e-9)
        )
        phase = rng.random(n)                       # cevrimde rastgele nokta [0,1)
        fill = phase * to_due * self._ds.base_rate_l
        days_since = np.floor(phase * self._hygiene_cap).astype(np.int64)
        return fill.astype(np.float64), days_since

    def solve_day(
        self,
        fill: np.ndarray,
        days_since: np.ndarray,
        solver: Solver,
        skip_lambda: float,
        *,
        day: int = 0,
        report_day: int = 0,
        is_warmup: bool = False,
        gen_today_l: int = 0,
    ) -> DayState:
        """TEK gunu coz ve DayState dondur. Durumu ILERLETMEZ.

        Gun dongusunun (`_run_core`) ve prototip karar destek aracinin ORTAK
        cekirdegi - must_visit kurali ve sinir gecisi yalnizca burada yasar,
        kopyalanmaz.

        `fill` toplama ONCESI doluluk (litre), `days_since` son ziyaretten bu yana
        gecen gun. Toplamanin uygulanmasi (fill sifirlama, days_since guncelleme)
        cagirana aittir - bkz. `apply_collection`.
        """
        n = self._n
        over_mask = fill > self._volume
        overflow_events = int(over_mask.sum())
        overflow_l = int(np.maximum(0.0, fill - self._volume).sum())

        # DOLULUK-FARKINDALIKLI must_visit (C4b): yarin +k*sigma'da tasacaksa
        # (fill + tahmini_yarin > hacim) VEYA hijyen tavanina ulastiysa bugun topla.
        must_visit = (fill + self._predict_next > self._volume) | (
            days_since >= self._hygiene_cap
        )
        demand = np.rint(fill).astype(np.int64)

        problem = build_problem(
            dist=self._ds.dist_m,
            travel_time=self._ds.time_s,
            demand=demand,
            service_time=self._service,
            volume=self._volume,
            must_visit=must_visit,
            capacity=self._capacity,
            shift_limit=self._shift,
            skip_lambda=skip_lambda,
            insertion_k=self._ins_k,
            **self._fuel_kwargs,
        )
        # Cozucu cozum bulamazsa gun KAYBEDILMEZ: greedy'ye dusulur ve gun
        # ISARETLENIR. Gerekce: 90 gun x 10 seed x 5 cozucu suren bir deneyde tek
        # bir zor gun tum kosuyu cope atmamali. Gercek operasyon da boyle davranir -
        # optimizasyon yetismezse kamyonlar yine de cikar.
        # Dusulen gun `solver_failed` ile kaydedilir, KPI ortalamalarina GIRMEZ
        # (fizibil olmadigi icin) ve saglik raporunda sayilir. Sessizce yutulmaz.
        solver_failed = False
        try:
            solution = solver.solve(problem)
        except NoFeasibleSolutionError:
            solver_failed = True
            solution = GreedySolver(self._cfg.fleet.num_vehicles).solve(problem)
        result = Evaluator(problem).evaluate(solution)
        visited = solution.visited_mask(n)

        mean_fill_pct = (
            float((fill[visited] / self._volume[visited]).mean() * 100)
            if visited.any() else 0.0
        )
        shift_util = (
            float(result.times.max()) / self._shift * 100 if result.times.size else 0.0
        )
        return DayState(
            day=day,
            report_day=report_day,
            is_warmup=is_warmup,
            fill_before=fill.copy(),
            demand=demand,
            must_visit=must_visit,
            volume=self._volume,
            problem=problem,
            solution=solution,
            result=result,
            visited=visited,
            gen_today_l=gen_today_l,
            collected_l=int(fill[visited].sum()),
            overflow_events=overflow_events,
            overflow_l=overflow_l,
            mean_fill_pct=mean_fill_pct,
            shift_util_pct=shift_util,
            solver_failed=solver_failed,
        )

    @staticmethod
    def apply_collection(
        fill: np.ndarray, days_since: np.ndarray, visited: np.ndarray
    ) -> None:
        """Toplamayi uygula (YERINDE): ziyaret edilenler bosalir, sayaclar doner."""
        fill[visited] = 0.0
        days_since[visited] = 0
        days_since[~visited] += 1

    def _run_core(
        self,
        solver: Solver,
        rng: np.random.Generator,
        skip_lambda: float,
        collect: Callable[[DayState], None],
    ) -> tuple[float, float, float, int, int, int]:
        """Gun dongusunun TEK kaynagi. Her gun bir DayState uretir, `collect`'e
        verir; kutle/tasma toplamlarini dondurur.

        run() ve replay() bu metodun uzerine kurulur (tek dogruluk kaynagi):
        farklari yalnizca `collect` callback'i. Boylece harita canli solve
        yaparken, deney istatistigi ile BIREBIR ayni gun dongusunu kullanir.
        """
        cfg = self._cfg
        horizon = cfg.simulation.warmup_days + cfg.simulation.report_days
        warmup = cfg.simulation.warmup_days

        daily_fills = generate_daily_fills(
            self._ds.base_rate_l, self._ds.market_day, cfg, rng, horizon
        )

        fill, days_since = self._initial_state(rng)
        # Baslangic dolulugu MEVCUT stoktur; kutle denkleminin sol tarafina
        # girer. Yoksa "uretilen = toplanan + kalan" esitligi baslangic stogu
        # kadar acik verir (Bolum 10 kontrol #1 yanlis alarm verirdi).
        total_generated = float(fill.sum())
        total_collected = 0.0
        total_overflow_events = 0
        infeasible_days = 0
        solver_failures = 0

        for d in range(horizon):
            gen_today = daily_fills[d]
            fill += gen_today
            total_generated += float(gen_today.sum())

            is_warmup = d < warmup
            state = self.solve_day(
                fill, days_since, solver, skip_lambda,
                day=d, report_day=d - warmup, is_warmup=is_warmup,
                gen_today_l=int(gen_today.sum()),
            )

            total_collected += float(state.collected_l)
            if state.solver_failed and not is_warmup:
                solver_failures += 1
            if not state.result.feasible and not is_warmup:
                infeasible_days += 1
            if not is_warmup:
                total_overflow_events += state.overflow_events

            collect(state)

            # Toplama uygula (collect SONRASI - fill_before anlik goruntu kalir)
            self.apply_collection(fill, days_since, state.visited)

        remaining = float(fill.sum())
        return (
            total_generated,
            total_collected,
            remaining,
            total_overflow_events,
            infeasible_days,
            solver_failures,
        )

    def run(self, solver: Solver, rng: np.random.Generator, skip_lambda: float) -> RunResult:
        records: list[DayRecord] = []

        def _collect(s: DayState) -> None:
            if s.is_warmup:
                return
            records.append(
                DayRecord(
                    day=s.report_day,
                    feasible=s.result.feasible,
                    fuel_ml=s.result.fuel_ml,
                    fuel_travel_ml=s.result.fuel_travel_ml,
                    fuel_stop_ml=s.result.fuel_stop_ml,
                    fuel_compaction_ml=s.result.fuel_compaction_ml,
                    fuel_load_term_ml=s.result.fuel_load_term_ml,
                    total_distance=s.result.total_distance,
                    fixed_distance=s.result.fixed_distance,
                    intra_distance=s.result.intra_distance,
                    generated_l=s.gen_today_l,
                    collected_l=s.collected_l,
                    n_visited=int(s.visited.sum()),
                    service_time_s=int(self._service[s.visited].sum()),
                    n_skipped=s.result.n_skipped,
                    overflow_events=s.overflow_events,
                    overflow_l=s.overflow_l,
                    mean_fill_pct=s.mean_fill_pct,
                    shift_util_pct=s.shift_util_pct,
                    solver_failed=s.solver_failed,
                )
            )

        gen, col, rem, overflow, infeasible, failures = self._run_core(
            solver, rng, skip_lambda, _collect
        )
        mass_ok = abs(gen - (col + rem)) < 1.0
        return RunResult(
            records=records,
            total_generated_l=int(gen),
            total_collected_l=int(col),
            remaining_l=int(rem),
            mass_ok=mass_ok,
            total_overflow_events=overflow,
            infeasible_days=infeasible,
            solver_failures=failures,
        )

    def replay(
        self,
        solver: Solver,
        rng: np.random.Generator,
        skip_lambda: float,
        *,
        include_warmup: bool = False,
    ) -> list[DayState]:
        """Tum ufku kostur, gun-basi TAM anlik goruntuyu (DayState) dondur.

        Panel/harita icin: run() ile BIREBIR ayni gun dongusu (tek kaynak
        _run_core), ama her gunun rotasi/atlamasi saklanir. Varsayilan olarak
        warm-up gunleri atilir (D3); harita raporlanan gunleri gosterir.
        """
        states: list[DayState] = []

        def _collect(s: DayState) -> None:
            if s.is_warmup and not include_warmup:
                return
            states.append(s)

        self._run_core(solver, rng, skip_lambda, _collect)
        return states
