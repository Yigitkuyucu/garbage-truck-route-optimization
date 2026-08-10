"""ABC (toggle-skip) testleri: B1'i yener, fizibil, must_visit korunur."""

from __future__ import annotations

import numpy as np

from domain.evaluator import Evaluator
from domain.problem import VRPProblem
from solvers.abc_solver import ABCSolver
from solvers.threshold_greedy import ThresholdGreedySolver
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
    *, demand: list[int], skip_penalty: list[int], must_visit: list[bool] | None = None
) -> VRPProblem:
    n = len(demand)
    mv = must_visit if must_visit is not None else [False] * n
    return vrp(
        dist=DIST,
        travel_time=DIST.copy(),
        demand=np.array(demand, dtype=np.int64),
        volume=np.array([1100] * n, dtype=np.int64),
        service_time=np.array([90] * n, dtype=np.int64),
        skip_penalty=np.array(skip_penalty, dtype=np.int64),
        must_visit=np.array(mv, dtype=np.bool_),
        capacity=2000,
        shift_limit=1_000_000,
    )


def _abc(problem: VRPProblem, seed: int = 0, t: float = 2.0) -> ABCSolver:
    return ABCSolver(
        num_vehicles=2, colony_size=20, limit=20, time_limit_sec=t,
        rng=np.random.default_rng(seed),
    )


def test_abc_beats_b1() -> None:
    # k0 esigin altinda (fill .45) ama skip_penalty YUKSEK -> B1 atlar (kotu),
    # ABC ziyaret eder (ucuz) -> daha dusuk maliyet.
    p = make_problem(demand=[500, 200, 900], skip_penalty=[5000, 100, 3000])
    ev = Evaluator(p)
    b1 = ev.evaluate(ThresholdGreedySolver(2, 0.7).solve(p))
    abc = ev.evaluate(_abc(p).solve(p))
    assert abc.feasible
    assert abc.total_cost <= b1.total_cost


def test_abc_feasible_and_respects_capacity() -> None:
    p = make_problem(demand=[900, 900, 900], skip_penalty=[100, 100, 100])
    r = Evaluator(p, debug=True).evaluate(_abc(p).solve(p))
    assert r.feasible
    assert all(load <= p.capacity for load in r.loads.tolist())


def test_abc_never_skips_must_visit() -> None:
    # k1 must_visit (dusuk skip_penalty olsa bile atlanamaz)
    p = make_problem(
        demand=[500, 100, 900], skip_penalty=[10, 10, 10], must_visit=[False, True, False]
    )
    sol = _abc(p).solve(p)
    r = Evaluator(p, debug=True).evaluate(sol)
    assert r.feasible
    assert sol.visited_mask(3)[1]  # must_visit ziyaret edildi


def test_abc_iterations_logged() -> None:
    p = make_problem(demand=[500, 200, 900], skip_penalty=[5000, 100, 3000])
    solver = _abc(p, t=1.0)
    solver.solve(p)
    assert solver.iterations > 0


# 5 konteyner, dagilmis dogrusal yerlesim -> rota sirasi onemli (2-opt fark eder)
_POS5 = [0, 5, 1, 4, 2, 8, 10]  # depot, k0..k4, dump
_DIST5 = np.array([[abs(a - b) for b in _POS5] for a in _POS5], dtype=np.int64)


def _problem5() -> VRPProblem:
    n = 5
    return vrp(
        dist=_DIST5,
        travel_time=_DIST5.copy(),
        demand=np.array([100] * n, dtype=np.int64),
        volume=np.array([1100] * n, dtype=np.int64),
        service_time=np.array([90] * n, dtype=np.int64),
        skip_penalty=np.array([9999] * n, dtype=np.int64),
        must_visit=np.array([True] * n, dtype=np.bool_),  # hepsi -> saf rota kalitesi
        capacity=5000,
        shift_limit=1_000_000,
    )


def test_abc_ls_no_worse_than_basic() -> None:
    # X2 (2-opt LS) ayni ziyaret setinde X1'den daha kotu OLMAMALI (ablation).
    p = _problem5()
    ev = Evaluator(p)
    x1 = ABCSolver(1, 20, 20, 1.5, np.random.default_rng(0), use_local_search=False)
    x2 = ABCSolver(1, 20, 20, 1.5, np.random.default_rng(0), use_local_search=True)
    r1 = ev.evaluate(x1.solve(p))
    r2 = ev.evaluate(x2.solve(p))
    assert r1.feasible and r2.feasible
    assert r2.total_distance <= r1.total_distance
    assert x1.code == "X1" and x2.code == "X2"
