"""FAZ 2 yakit modeli - elle hesaplanmis vakalar.

En kritik test `test_ordering_changes_fuel_at_equal_distance`: ayni ziyaret kumesi,
AYNI toplam mesafe, farkli sira -> FARKLI yakit. Bu, Faz 2'nin tezinin (rota
siralamasi yakiti etkiler) kodda gercekten var oldugunu kanitlar. Gecmezse yuk
takibi yanlistir.
"""

from __future__ import annotations

import numpy as np

from domain.evaluator import Evaluator
from domain.problem import VRPProblem
from domain.solution import Solution
from solvers.ortools_solver import ORToolsSolver
from tests.factory import vrp

# Dugum: 0=garaj, 1=k0, 2=k1, 3=dokum
# Garajdan ve dokuma mesafeler KASTEN SIMETRIK -> sira mesafeyi degistirmez,
# yalnizca YUK PROFILINI degistirir. Siralama testinin can damari budur.
DIST = np.array(
    [
        [0, 10, 10, 30],
        [10, 0, 5, 20],
        [10, 5, 0, 20],
        [30, 20, 20, 0],
    ],
    dtype=np.int64,
)

BASE = 1.0      # mL/m - bos kamyon
SLOPE = 0.01    # mL/m/kg - yuk egimi (elle hesap icin buyuk secildi)
NOMINAL = 5.0   # mL/m - sabit-yuk referansi


def make_problem(
    *,
    demand: list[int],
    stop_start_ml: int = 0,
    compaction_ml_per_liter: float = 0.0,
    slope: float = SLOPE,
    skip_penalty: list[int] | None = None,
    must_visit: list[bool] | None = None,
) -> VRPProblem:
    n = len(demand)
    return vrp(
        dist=DIST,
        travel_time=DIST.copy(),
        demand=np.array(demand, dtype=np.int64),
        mass_kg=np.array(demand, dtype=np.float64),   # 1 kg/L - yuvarlak hesap
        volume=np.array([1100] * n, dtype=np.int64),
        service_time=np.array([90] * n, dtype=np.int64),
        skip_penalty=np.array(skip_penalty or [0] * n, dtype=np.int64),
        must_visit=np.array(must_visit or [False] * n, dtype=np.bool_),
        capacity=100_000,
        shift_limit=1_000_000,
        fuel_base_ml_per_m=BASE,
        fuel_slope_ml_per_m_per_kg=slope,
        stop_start_ml=stop_start_ml,
        compaction_ml_per_liter=compaction_ml_per_liter,
        nominal_ml_per_m=NOMINAL,
    )


def _ev(problem: VRPProblem, routes: list[list[int]]):
    return Evaluator(problem, debug=True).evaluate(Solution.from_lists(routes))


# ---------------------------------------------------------------- yuk profili


def test_load_profile_hand_computed() -> None:
    """Bacak basina biriken yuk elle dogrulanir (M.7 Adim 4)."""
    p = make_problem(demand=[100, 900])
    r = _ev(p, [[0, 1]])

    # mesafe: garaj->k0=10, k0->k1=5, k1->dokum=20, dokum->garaj=30  => 65
    assert r.total_distance == 65
    assert r.intra_distance == 5
    assert r.fixed_distance == 60

    # yakit, bacak bacak (kamyon garajdan BOS cikar, dokumde bosalir):
    #   garaj->k0 : yuk 0    -> 1.00 * 10 =  10
    #   k0->k1    : yuk 100  -> 2.00 *  5 =  10
    #   k1->dokum : yuk 1000 -> 11.00* 20 = 220
    #   dokum->garaj: yuk 0  -> 1.00 * 30 =  30
    assert r.fuel_travel_ml == 270
    assert r.fuel_ml == 270          # dur-kalk ve sikistirma sifir
    assert r.total_cost == 270       # atlama yok


def test_empty_truck_leg_and_dump_leg_masses() -> None:
    """Garaj bacagi BOS, dokum bacagi TAM yuk (yuk yerlesimi dogru mu)."""
    p = make_problem(demand=[1000, 500])
    # Tek konteyner ziyaret edilir: ic bacak YOK, dokum bacagi tam yuk tasir
    r = _ev(p, [[0]])
    assert r.total_distance == 10 + 20 + 30
    # garaj->k0: yuk 0    -> 1.00 * 10 =  10
    # k0->dokum: yuk 1000 -> 11.00* 20 = 220
    # dokum->garaj: yuk 0 -> 1.00 * 30 =  30
    assert r.fuel_travel_ml == 260

    # Ikinci konteyner eklenince yuk artar -> yakit da artar
    assert _ev(p, [[0, 1]]).fuel_travel_ml > r.fuel_travel_ml


# ------------------------------------------------- FAZ 2'NIN TEZI: SIRALAMA


def test_ordering_changes_fuel_at_equal_distance() -> None:
    """AYNI ziyaret kumesi, AYNI mesafe, farkli sira -> FARKLI yakit.

    Faz 2'nin tezi tam olarak budur. Bu test gecmezse yuk takibi yanlistir.
    """
    p = make_problem(demand=[100, 900])
    heavy_last = _ev(p, [[0, 1]])    # hafif once, agir sona  (iyi)
    heavy_first = _ev(p, [[1, 0]])   # agir once              (kotu)

    # Mesafe OZDES - geometri simetrik secildi
    assert heavy_last.total_distance == heavy_first.total_distance == 65

    # Yakit FARKLI: ic bacakta tasinan kutle 100 kg yerine 900 kg
    #   fark = slope * (900 - 100) * 5 m = 0.01 * 800 * 5 = 40 mL
    assert heavy_first.fuel_travel_ml - heavy_last.fuel_travel_ml == 40
    assert heavy_last.fuel_ml < heavy_first.fuel_ml


def test_no_ordering_effect_when_slope_zero() -> None:
    """slope=0 -> maliyet yuk-kor; siralama yakiti degistirmemeli (kontrol testi)."""
    p = make_problem(demand=[100, 900], slope=0.0)
    assert _ev(p, [[0, 1]]).fuel_ml == _ev(p, [[1, 0]]).fuel_ml


# ------------------------------------------------------- kalemler + sabit-yuk


def test_stop_and_compaction_terms() -> None:
    """Dur-kalk durak sayisiyla, sikistirma toplanan litreyle olcekli."""
    p = make_problem(demand=[100, 900], stop_start_ml=25, compaction_ml_per_liter=0.03)
    r = _ev(p, [[0, 1]])
    assert r.fuel_stop_ml == 2 * 25
    # dugum basi yuvarlama: rint(0.03*100)=3, rint(0.03*900)=27
    assert r.fuel_compaction_ml == 30
    assert r.fuel_ml == r.fuel_travel_ml + r.fuel_stop_ml + r.fuel_compaction_ml
    assert r.n_visited == 2


def test_constload_uses_nominal_rate() -> None:
    """Sabit-yuk buyuklugu nominal orandan gelir; dokum bacagi TEK ark yuvarlanir."""
    p = make_problem(demand=[100, 900])
    r = _ev(p, [[0, 1]])
    # nominal * (garaj->k0) + nominal * (k0->k1) + nominal * (k1->dokum->garaj)
    assert r.fuel_constload_ml == 5 * 10 + 5 * 5 + 5 * (20 + 30)
    # OR-Tools'un erisemedigi terim = gercek - sabit-yuk
    assert r.fuel_load_term_ml == r.fuel_ml - r.fuel_constload_ml


def test_skip_penalty_is_in_fuel_units() -> None:
    """skip_penalty amac fonksiyonuna dogrudan eklenir (ayni birim: mL)."""
    p = make_problem(demand=[100, 900], skip_penalty=[7000, 8000])
    r = _ev(p, [[0]])           # k1 atlandi
    assert r.skip_cost == 8000
    assert r.total_cost == r.fuel_ml + 8000
    assert r.n_skipped == 1


def test_empty_solution_has_zero_fuel() -> None:
    p = make_problem(demand=[100, 900], skip_penalty=[7000, 8000])
    r = _ev(p, [[], []])
    assert r.fuel_ml == 0
    assert r.fuel_travel_ml == 0
    assert r.fuel_constload_ml == 0
    assert r.n_visited == 0
    assert r.total_cost == 15000


# ------------------------------------------------ OR-Tools capraz dogrulama


def test_ortools_cross_validation_with_real_fuel_coefficients() -> None:
    """G2: gercek (notr olmayan) yakit katsayilariyla da esitlik KESIN olmali.

    Bu, tests/test_ortools.py'deki notr-yakit surumunun tamamlayicisidir:
    nominal != base, dur-kalk ve sikistirma sifirdan farkli.
    """
    p = make_problem(
        demand=[100, 900], stop_start_ml=25, compaction_ml_per_liter=0.03,
        skip_penalty=[7000, 8000],
    )
    solver = ORToolsSolver(num_vehicles=1, time_limit_sec=2)
    solution = solver.solve(p)
    r = Evaluator(p, debug=True).evaluate(solution)
    assert r.feasible, r.violations
    assert solver.last_objective() == r.fuel_constload_ml + r.skip_cost


def test_pretty_print_reports_fuel_breakdown() -> None:
    from domain.evaluator import pretty_print

    p = make_problem(demand=[100, 900], stop_start_ml=25)
    sol = Solution.from_lists([[0, 1]])
    text = pretty_print(sol, _ev(p, [[0, 1]]), p)
    assert "yakit kalemleri" in text
    assert "yuk baglasimi" in text
