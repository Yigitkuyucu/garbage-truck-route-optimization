"""B0 (sabit rota) + B1 (esik+greedy) testleri, Evaluator ile dogrulama."""

from __future__ import annotations

import numpy as np

from domain.evaluator import Evaluator
from domain.problem import VRPProblem
from solvers.greedy import GreedySolver, greedy_routes, nearest_neighbor_order
from solvers.threshold_greedy import ThresholdGreedySolver, threshold_candidates
from tests.factory import vrp

# Dogrusal yerlesim: depot=0, k0..k3 = pos 1..4, dump = pos 10
POS = [0, 1, 2, 3, 4, 10]
DIST = np.array([[abs(a - b) for b in POS] for a in POS], dtype=np.int64)


def make_problem(
    *,
    demand: list[int],
    must_visit: list[bool] | None = None,
    capacity: int = 1500,
) -> VRPProblem:
    n = len(demand)
    mv = must_visit if must_visit is not None else [False] * n
    return vrp(
        dist=DIST,
        travel_time=DIST.copy(),
        demand=np.array(demand, dtype=np.int64),
        volume=np.full(n, 1100, dtype=np.int64),
        service_time=np.full(n, 90, dtype=np.int64),
        skip_penalty=np.array([10, 20, 30, 40], dtype=np.int64),
        must_visit=np.array(mv, dtype=np.bool_),
        capacity=capacity,
        shift_limit=1_000_000,
    )


def test_nn_order_from_depot() -> None:
    p = make_problem(demand=[100, 100, 100, 100])
    assert nearest_neighbor_order(p, [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert nearest_neighbor_order(p, []) == []
    # sira bagimsiz: girdi karisik da olsa NN cografyaya gore dizer
    assert nearest_neighbor_order(p, [3, 1, 2, 0]) == [0, 1, 2, 3]


def test_b0_visits_all() -> None:
    p = make_problem(demand=[1000, 100, 900, 50])
    sol = GreedySolver(num_vehicles=2).solve(p)
    assert sol.visited_mask(p.n_containers).tolist() == [True, True, True, True]
    r = Evaluator(p, debug=True).evaluate(sol)
    assert r.feasible
    assert r.n_skipped == 0
    # kapasite bolunmesi: [k0,k1]=1100, [k2,k3]=950 (ikisi de <=1500)
    assert r.loads.tolist() == [1100, 950]


def test_b0_infeasible_when_over_capacity() -> None:
    p = make_problem(demand=[1000, 100, 900, 50], capacity=500)
    sol = GreedySolver(num_vehicles=2).solve(p)
    r = Evaluator(p, debug=True).evaluate(sol)
    assert not r.feasible  # toplam talep 2 arac x 500'u asar
    assert any("kapasite" in v for v in r.violations)


def test_b1_threshold_filters() -> None:
    # doluluk = demand/1100; esik 0.7 -> 770 L ustu
    p = make_problem(demand=[1000, 100, 900, 50])  # fill: .91,.09,.82,.045
    assert threshold_candidates(p, 0.7) == [0, 2]
    sol = ThresholdGreedySolver(num_vehicles=2, threshold=0.7).solve(p)
    assert sol.visited_mask(p.n_containers).tolist() == [True, False, True, False]
    r = Evaluator(p, debug=True).evaluate(sol)
    assert r.feasible
    assert r.n_skipped == 2
    assert r.skip_cost == 20 + 40  # k1, k3 skip_penalty


def test_b1_must_visit_overrides_threshold() -> None:
    # k1 esigin altinda (fill .09) ama must_visit -> dahil edilmeli
    p = make_problem(demand=[1000, 100, 900, 50], must_visit=[False, True, False, False])
    assert threshold_candidates(p, 0.7) == [0, 1, 2]
    sol = ThresholdGreedySolver(num_vehicles=2, threshold=0.7).solve(p)
    r = Evaluator(p, debug=True).evaluate(sol)
    assert r.feasible  # must_visit karsilandi
    assert sol.visited_mask(p.n_containers)[1]


def test_b1_all_below_threshold_all_skipped() -> None:
    p = make_problem(demand=[100, 100, 100, 100])  # hepsi < 0.7
    assert threshold_candidates(p, 0.7) == []
    sol = ThresholdGreedySolver(num_vehicles=2, threshold=0.7).solve(p)
    r = Evaluator(p, debug=True).evaluate(sol)
    assert r.n_skipped == 4
    assert r.total_distance == 0


def test_greedy_routes_num_vehicles() -> None:
    p = make_problem(demand=[100, 100, 100, 100])
    sol = greedy_routes(p, [0, 1, 2, 3], num_vehicles=3)
    assert sol.n_vehicles == 3


def test_b1_beats_b0_on_distance_when_skipping() -> None:
    # B1 bazi konteynerleri atladigi icin bolge-ici mesafesi B0'dan kucuk olmali
    p = make_problem(demand=[1000, 100, 900, 50])
    b0 = Evaluator(p).evaluate(GreedySolver(2).solve(p))
    b1 = Evaluator(p).evaluate(ThresholdGreedySolver(2, 0.7).solve(p))
    assert b1.intra_distance <= b0.intra_distance
