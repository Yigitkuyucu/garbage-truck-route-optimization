"""B1 (esik + greedy) - naif baseline.

Doluluk (demand/volume) > b1_threshold olan konteynerler toplanir; geri kalan
ATLANIR. must_visit konteynerleri (sert kisit) esikten bagimsiz her zaman dahil.
Aday kume greedy_routes ile rotalanir.

Bu "naif esik kurali", ceza-tabanli optimizasyona (ABC) kiyas noktasidir:
esigi SEN secersin (neden %70?), algoritma degil.

KABUK.
"""

from __future__ import annotations

import numpy as np

from domain.problem import VRPProblem
from domain.solution import Solution
from solvers.base import Solver
from solvers.greedy import greedy_routes


def threshold_candidates(problem: VRPProblem, threshold: float) -> list[int]:
    """Doluluk > threshold VEYA must_visit olan konteyner indeksleri."""
    fill = problem.demand.astype(np.float64) / problem.volume.astype(np.float64)
    keep = (fill > threshold) | problem.must_visit
    return [int(c) for c in np.nonzero(keep)[0]]


class ThresholdGreedySolver(Solver):
    """B1 - esik + greedy."""

    code = "B1"
    name = "Esik + greedy"

    def __init__(self, num_vehicles: int, threshold: float) -> None:
        self._num_vehicles = num_vehicles
        self._threshold = threshold

    def solve(self, problem: VRPProblem) -> Solution:
        candidates = threshold_candidates(problem, self._threshold)
        return greedy_routes(problem, candidates, self._num_vehicles)
