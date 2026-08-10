"""OR-Tools capraz dogrulama (ZORUNLU).

FAZ 2: OR-Tools gercek yuk duyarli maliyeti cozemez (ark maliyeti
bir boyutun degerine baglanamaz); ona o maliyetin EN IYI YUK-KOR YAKLASIMI verilir.
Dolayisiyla capraz dogrulama artik sabit-yuk buyuklugu uzerinden yapilir:

    OR-Tools objektifi == Evaluator.fuel_constload_ml + skip_cost

Bu esitlik TAM SAYI olarak KESINDIR (ayni bacak gruplamasi, ayni yuvarlama).
Aradaki fark - gercek yakit eksi sabit-yuk - OR-Tools'un erisemedigi terimdir
(EvalResult.fuel_load_term_ml); test_fuel.py onu ayrica sinar.
"""

from __future__ import annotations

import numpy as np

from domain.evaluator import Evaluator
from domain.problem import VRPProblem
from solvers.ortools_solver import ORToolsSolver
from tests.factory import vrp

DIST = np.array(
    [
        [0, 10, 20, 30, 40],
        [10, 0, 5, 15, 25],
        [20, 5, 0, 10, 20],
        [30, 15, 10, 0, 12],
        [40, 25, 20, 12, 0],
    ],
    dtype=np.int64,
)


def make_problem(
    *,
    capacity: int = 1000,
    demand: list[int] | None = None,
    must_visit: list[bool] | None = None,
    skip_penalty: list[int] | None = None,
) -> VRPProblem:
    d = demand if demand is not None else [100, 200, 300]
    mv = must_visit if must_visit is not None else [False, False, False]
    sp = skip_penalty if skip_penalty is not None else [1000, 2000, 3000]
    return vrp(
        dist=DIST,
        travel_time=DIST.copy(),
        demand=np.array(d, dtype=np.int64),
        volume=np.array([1100, 1100, 1100], dtype=np.int64),
        service_time=np.array([90, 90, 90], dtype=np.int64),
        skip_penalty=np.array(sp, dtype=np.int64),
        must_visit=np.array(mv, dtype=np.bool_),
        capacity=capacity,
        shift_limit=1_000_000,
    )


def test_raises_instead_of_returning_empty_routes() -> None:
    """Fizibil cozum yoksa HATA VERIR - bos rota dondurmez (Bolum 6 + Bolum 10).

    Bos rota "0 mesafe / %100 tasarruf" gibi gorunur; bu kusur duyarlilik
    tablosunda kat 3-5 senaryosunu "tasarruf %100, tasma 33.637" gostermisti.
    """
    import pytest

    from solvers.base import NoFeasibleSolutionError

    # Talep filo kapasitesini kat kat asiyor + hepsi must_visit -> atlanamaz
    p = make_problem(
        capacity=100, demand=[5000, 5000, 5000], must_visit=[True, True, True]
    )
    with pytest.raises(NoFeasibleSolutionError, match="fizibil cozum bulamadi"):
        ORToolsSolver(num_vehicles=1, time_limit_sec=2).solve(p)


def _cross_check(problem: VRPProblem) -> None:
    """OR-Tools objektifi == Evaluator sabit-yuk yakiti + skip_cost (G2)."""
    solver = ORToolsSolver(num_vehicles=2, time_limit_sec=2)
    solution = solver.solve(problem)
    result = Evaluator(problem, debug=True).evaluate(solution)
    assert result.feasible, result.violations
    expected = result.fuel_constload_ml + result.skip_cost
    assert solver.last_objective() == expected, (
        f"OR={solver.last_objective()} != Evaluator sabit-yuk={expected}"
    )


def test_cross_validation_all_visited() -> None:
    # Hepsi sigar -> skip yok; objektif == toplam mesafe
    p = make_problem(capacity=1000)
    _cross_check(p)


def test_cross_validation_with_skip() -> None:
    # Dusuk skip_penalty + dar kapasite -> OR bazilarini atlar
    p = make_problem(capacity=250, skip_penalty=[10, 10, 10])
    _cross_check(p)


def test_cross_validation_must_visit() -> None:
    # must_visit disjunction almaz -> atlanamaz; objektif yine eslesir
    p = make_problem(capacity=1000, must_visit=[True, True, True])
    _cross_check(p)


def test_ortools_respects_capacity() -> None:
    # Yuksek skip_penalty -> atlamaz; 2 arac kapasiteye uyar
    p = make_problem(capacity=350, skip_penalty=[100000, 100000, 100000])
    solver = ORToolsSolver(num_vehicles=2, time_limit_sec=2)
    result = Evaluator(p, debug=True).evaluate(solver.solve(p))
    assert result.feasible
    assert all(load <= 350 for load in result.loads.tolist())
    assert solver.last_objective() == result.total_cost


def test_ortools_visits_all_when_penalty_huge() -> None:
    p = make_problem(capacity=1000, skip_penalty=[999999, 999999, 999999])
    solver = ORToolsSolver(num_vehicles=2, time_limit_sec=2)
    sol = solver.solve(p)
    assert sol.visited_mask(3).tolist() == [True, True, True]
