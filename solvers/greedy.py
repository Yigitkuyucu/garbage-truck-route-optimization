"""B0 (sabit rota) + ortak greedy rota kurucu.

B0 = mevcut sabit-rota pratigi: her gun TUM konteynerler toplanir. Rota, en yakin
komsu (NN) sirasiyla kurulur (cografyaya bagli, gunden gune ayni sira) ve
kapasiteye gore araclara bolunur.

Baseline durustlugu: B0 kasten kotu kurulmaz - NN makul bir rotadir.

KABUK (solvers/ sarmalayici). Maliyet HESAPLANMAZ; Evaluator'a birakilir.
"""

from __future__ import annotations

from domain.problem import VRPProblem
from domain.solution import Solution
from solvers.base import Solver


def nearest_neighbor_order(problem: VRPProblem, candidates: list[int]) -> list[int]:
    """Adaylari garajdan baslayarak en yakin komsu sirasina diz (metre)."""
    if not candidates:
        return []
    dist = problem.dist
    remaining = list(candidates)
    order: list[int] = []
    current = problem.depot_index
    while remaining:
        nxt = min(remaining, key=lambda c: int(dist[current, problem.node_of(c)]))
        order.append(nxt)
        remaining.remove(nxt)
        current = problem.node_of(nxt)
    return order


def greedy_routes(
    problem: VRPProblem, candidates: list[int], num_vehicles: int
) -> Solution:
    """Adaylari NN sirasinda kapasiteye gore araclara bol.

    Kapasite yetmezse yeni araca gecer; araclar biterse fazlalik son araca kalir
    (Evaluator infeasible isaretler - sessizce dusurmez).
    """
    order = nearest_neighbor_order(problem, candidates)
    routes: list[list[int]] = [[] for _ in range(num_vehicles)]
    v = 0
    load = 0
    for c in order:
        d = int(problem.demand[c])
        if load + d > problem.capacity and v < num_vehicles - 1:
            v += 1
            load = 0
        routes[v].append(c)
        load += d
    return Solution.from_lists(routes)


class GreedySolver(Solver):
    """B0 - sabit rota: her gun tum konteynerler."""

    code = "B0"
    name = "Sabit rota (greedy, tum konteynerler)"

    def __init__(self, num_vehicles: int) -> None:
        self._num_vehicles = num_vehicles

    def solve(self, problem: VRPProblem) -> Solution:
        candidates = list(range(problem.n_containers))
        return greedy_routes(problem, candidates, self._num_vehicles)
