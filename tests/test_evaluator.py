"""Evaluator capraz dogrulamasi - elle hesaplanmis vakalar (%100 satir kapsami)."""

from __future__ import annotations

import numpy as np

from domain.evaluator import EvalResult, Evaluator, pretty_print
from domain.problem import VRPProblem
from domain.solution import Solution
from tests.factory import vrp

# Dugum: 0=garaj, 1=k0, 2=k1, 3=k2, 4=dokum (simetrik, elle hesaplanabilir)
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
    shift_limit: int = 100_000,
    must_visit: list[bool] | None = None,
    travel_time: np.ndarray | None = None,
) -> VRPProblem:
    mv = must_visit if must_visit is not None else [False, False, False]
    return vrp(
        dist=DIST,
        travel_time=DIST.copy() if travel_time is None else travel_time,
        demand=np.array([100, 200, 300], dtype=np.int64),
        volume=np.array([1100, 1100, 1100], dtype=np.int64),
        service_time=np.array([90, 90, 90], dtype=np.int64),
        skip_penalty=np.array([1000, 2000, 3000], dtype=np.int64),
        must_visit=np.array(mv, dtype=np.bool_),
        capacity=capacity,
        shift_limit=shift_limit,
    )


def ev(problem: VRPProblem, routes: list[list[int]], *, debug: bool = True) -> EvalResult:
    return Evaluator(problem, debug=debug).evaluate(Solution.from_lists(routes))


def test_basic_feasible_single_route() -> None:
    r = ev(make_problem(), [[0, 1, 2]])
    # depot->1->2->3->dump->depot = 10+5+10+12+40 = 77
    assert r.total_distance == 77
    assert r.fixed_distance == 62  # 10 + 12 + 40
    assert r.intra_distance == 15  # 5 + 10
    assert r.fixed_distance + r.intra_distance == r.total_distance
    assert r.skip_cost == 0
    assert r.total_cost == 77
    assert r.feasible
    assert r.n_skipped == 0
    assert r.loads.tolist() == [600]
    assert r.times.tolist() == [77 + 270]  # mesafe + 3*90 servis
    assert r.violations == ()


def test_skip_container_adds_penalty() -> None:
    r = ev(make_problem(), [[0, 2]])  # k1 atlanir
    # depot->1->3->dump->depot = 10+15+12+40 = 77
    assert r.total_distance == 77
    assert r.intra_distance == 15  # d[1,3]
    assert r.skip_cost == 2000  # skip_penalty[1]
    assert r.total_cost == 2077
    assert r.n_skipped == 1
    assert r.feasible


def test_two_vehicles_decomposition() -> None:
    r = ev(make_problem(), [[0], [1, 2]])
    # v0: depot->1->dump->depot = 10+25+40 = 75 (hepsi fixed)
    # v1: depot->2->3->dump->depot = 20+10+12+40 = 82; fixed=72 intra=10
    assert r.distances.tolist() == [75, 82]
    assert r.total_distance == 157
    assert r.fixed_distance == 147
    assert r.intra_distance == 10
    assert r.loads.tolist() == [100, 500]
    assert r.feasible


def test_empty_route() -> None:
    r = ev(make_problem(), [[], [0, 1, 2]])
    assert r.distances.tolist() == [0, 77]
    assert r.loads.tolist() == [0, 600]
    assert r.times.tolist() == [0, 347]
    assert r.feasible


def test_capacity_violation() -> None:
    r = ev(make_problem(capacity=500), [[0, 1, 2]])  # yuk 600 > 500
    assert not r.feasible
    assert any("yuk 600 > kapasite 500" in v for v in r.violations)


def test_shift_violation() -> None:
    r = ev(make_problem(shift_limit=100), [[0, 1, 2]])  # sure 347 > 100
    assert not r.feasible
    assert any("sure 347 > vardiya 100" in v for v in r.violations)


def test_must_visit_skipped() -> None:
    r = ev(make_problem(must_visit=[False, True, False]), [[0, 2]])  # k1 must_visit
    assert not r.feasible
    assert any("must_visit atlandi" in v for v in r.violations)


def test_duplicate_visit() -> None:
    r = ev(make_problem(), [[0], [0, 1]])  # k0 iki kez
    assert not r.feasible
    assert any("2 kez ziyaret" in v for v in r.violations)


def test_time_uses_travel_time_not_dist() -> None:
    # travel_time = 10*dist -> sure seyahat bacaklari 10 kat
    p = make_problem(shift_limit=100_000, travel_time=DIST * 10)
    r = ev(p, [[0, 1, 2]])
    assert r.times.tolist() == [770 + 270]  # 10*77 seyahat + 270 servis
    assert r.total_distance == 77  # mesafe travel_time'dan ETKILENMEZ


def test_violations_reported_without_debug() -> None:
    """Ihlal sebebi debug bayragindan BAGIMSIZ dondurulur.

    Eskiden debug=False sebebi gizliyordu; metinler zaten kosulsuz uretildigi
    icin bu hicbir sey kazandirmiyor, yalnizca prototip aracin "Plan
    uygulanabilir degil" deyip nedenini soyleyememesine yol aciyordu.
    """
    r = ev(make_problem(capacity=1), [[0, 1, 2]], debug=False)
    assert not r.feasible
    assert r.violations  # sebep artik gorunur
    assert any("kapasite" in v for v in r.violations)


def test_all_skipped_empty_solution() -> None:
    r = ev(make_problem(), [[]])
    assert r.total_distance == 0
    assert r.n_skipped == 3
    assert r.skip_cost == 6000  # 1000+2000+3000
    assert r.total_cost == 6000


def test_pretty_print_with_names_and_skip() -> None:
    p = make_problem()
    sol = Solution.from_lists([[0, 2]])
    r = Evaluator(p, debug=True).evaluate(sol)
    out = pretty_print(sol, r, p, names=["Carsi", "Balikhane", "Kenar"])
    assert "Carsi" in out and "Kenar" in out
    assert "Garaj ->" in out and "Dokum -> Garaj" in out
    assert "atlanan: 1" in out
    assert "sabit(deadhead)" in out


def test_pretty_print_no_names_empty_route_and_violations() -> None:
    p = make_problem(capacity=1)
    sol = Solution.from_lists([[], [0, 1, 2]])
    r = Evaluator(p, debug=True).evaluate(sol)
    out = pretty_print(sol, r, p)
    assert "k0" in out  # isimsiz -> k0
    assert "(bos)" in out
    assert "IHLALLER" in out


def test_invariant_fixed_plus_intra() -> None:
    p = make_problem()
    for routes in ([[0, 1, 2]], [[0], [1, 2]], [[0, 2], [1]], [[]]):
        r = ev(p, routes)
        assert r.fixed_distance + r.intra_distance == r.total_distance


def test_solution_flat_roundtrip() -> None:
    from domain.solution import decode_solution

    sol = Solution.from_lists([[0, 2], [1]])
    flat, lengths = sol.flat()
    assert flat.tolist() == [0, 2, 1]
    assert lengths.tolist() == [2, 1]
    assert decode_solution(flat, lengths).routes == sol.routes


def test_solution_flat_all_empty() -> None:
    sol = Solution.from_lists([[], []])
    flat, lengths = sol.flat()
    assert flat.shape == (0,)
    assert lengths.tolist() == [0, 0]
    assert sol.visited_mask(3).tolist() == [False, False, False]


def test_solution_nvehicles_and_visited_mask() -> None:
    sol = Solution.from_lists([[0], [2]])  # k1 ziyaret edilmez
    assert sol.n_vehicles == 2
    assert sol.visited_mask(3).tolist() == [True, False, True]
