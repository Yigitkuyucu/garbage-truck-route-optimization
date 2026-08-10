"""ABC (Artificial Bee Colony) - X1/X2.

FAZ 1 (bu dosya, ilk tur): SADECE toggle-skip operatoru + cheapest-insertion +
repair. Amac: ABC'nin AKILLI ATLAMA karari verebildigini gostermek (B1'i yener).
Rotalama operatorleri (2-opt, Or-opt, relocate, swap) sonra eklenecek.

Kararlar:
- Cozum temsili: dogrudan rota listesi list[list[int]] (E1).
- Fizibilite: koruyucu operator + repair; infeasible cozum tutulmaz (E3).
  Kisit ihlali cezasi YOK. skip_penalty amac fonksiyonunun parcasi (kisit degil).
- Baslangic: %20 greedy-seeded + %80 rastgele-ama-fizibil (E5).
- Durdurma: duvar saati. Iterasyon loglanir.
- Maliyet YALNIZCA Evaluator'dan (G2).

Not: Faz 1'de numba yok; kod numba'ya tasinabilir tutulur (F2-ek).
"""

from __future__ import annotations

import time
from itertools import pairwise

import numpy as np

from domain.evaluator import EvalResult, Evaluator
from domain.problem import VRPProblem
from domain.solution import Solution
from solvers.base import Solver
from solvers.greedy import greedy_routes

Routes = list[list[int]]


def _cheapest_insertion(
    routes: Routes, loads: list[int], c: int, problem: VRPProblem
) -> tuple[int, int, int] | None:
    """Konteyner c icin en ucuz FIZIBIL (kapasite) ekleme: (arac, pozisyon, delta).

    Hicbir aracta yer yoksa None.
    """
    dist = problem.dist
    cnode = problem.node_of(c)
    dc = int(problem.demand[c])
    best: tuple[int, int, int] | None = None
    for v, route in enumerate(routes):
        if loads[v] + dc > problem.capacity:
            continue
        for pos in range(len(route) + 1):
            prev = problem.depot_index if pos == 0 else problem.node_of(route[pos - 1])
            nxt = problem.dump_index if pos == len(route) else problem.node_of(route[pos])
            delta = int(dist[prev, cnode]) + int(dist[cnode, nxt]) - int(dist[prev, nxt])
            if best is None or delta < best[2]:
                best = (v, pos, delta)
    return best


def _insert(
    routes: Routes, loads: list[int], c: int, v: int, pos: int, problem: VRPProblem
) -> None:
    routes[v].insert(pos, c)
    loads[v] += int(problem.demand[c])


def _greedy_all(problem: VRPProblem, num_vehicles: int) -> tuple[Routes, list[int]]:
    """B0 (NN greedy) tohumu: tum konteynerler. ABC en az B0 kalitesinden baslar."""
    sol = greedy_routes(problem, list(range(problem.n_containers)), num_vehicles)
    routes: Routes = [list(r) for r in sol.routes]
    loads = [int(sum(int(problem.demand[c]) for c in r)) for r in routes]
    return routes, loads


def _random_feasible(
    problem: VRPProblem, num_vehicles: int, rng: np.random.Generator, non_must: list[int]
) -> tuple[Routes, list[int]]:
    """must_visit + rastgele alt kume, cheapest-insertion ile fizibil yerlestir."""
    routes: Routes = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles
    must = [c for c in range(problem.n_containers) if bool(problem.must_visit[c])]
    for c in must:
        ins = _cheapest_insertion(routes, loads, c, problem)
        v, pos = (ins[0], ins[1]) if ins else (min(range(num_vehicles), key=lambda x: loads[x]), 0)
        _insert(routes, loads, c, v, pos, problem)
    for c in non_must:
        if rng.random() < 0.5:
            ins = _cheapest_insertion(routes, loads, c, problem)
            if ins is not None:
                _insert(routes, loads, c, ins[0], ins[1], problem)
    return routes, loads


# Operator donusu: (routes, loads, etkilenen_araclar) - hedefli LS icin.
Neighbor = tuple[Routes, list[int], tuple[int, ...]]


def _detour(route: list[int], i: int, problem: VRPProblem) -> int:
    """route[i]'yi cikarmanin mesafe tasarrufu (sapma maliyeti)."""
    dist, node = problem.dist, problem.node_of
    prev = problem.depot_index if i == 0 else node(route[i - 1])
    nxt = problem.dump_index if i == len(route) - 1 else node(route[i + 1])
    c = node(route[i])
    return int(dist[prev, c]) + int(dist[c, nxt]) - int(dist[prev, nxt])


def _directed_toggle(
    routes: Routes, loads: list[int], problem: VRPProblem,
    rng: np.random.Generator, non_must: list[int],
) -> Neighbor | None:
    """YONLU toggle-skip: amac fonksiyonunu en cok iyilestiren toggle'i sec.

    Cikarma kazanci = detour(c) - skip_penalty[c]  (>0: cikarmak iyilestirir -
      cografi olarak PAHALI (uzak) konteyneri atlar).
    Ekleme kazanci  = skip_penalty[c] - cheapest_insertion(c)  (>0: eklemek iyilestirir).
    En yuksek pozitif kazancli hamle uygulanir. Iyilestiren yoksa None.
    Ekleme adaylari orneklenir (hiz + stokastiklik).
    """
    if not non_must:
        return None
    nm = set(non_must)
    sp = problem.skip_penalty
    best_gain = 0.0
    best: tuple[str, int, int] | None = None  # ("rm", v, i) | ("ins", c, -1)
    visited: set[int] = set()
    for v, r in enumerate(routes):
        for i, c in enumerate(r):
            visited.add(c)
            if c not in nm:
                continue
            gain = _detour(r, i, problem) - int(sp[c])
            if gain > best_gain:
                best_gain, best = float(gain), ("rm", v, i)

    skipped = [c for c in non_must if c not in visited]
    if skipped:
        sample = rng.choice(skipped, size=min(20, len(skipped)), replace=False)
        for c in sample:
            ins = _cheapest_insertion(routes, loads, int(c), problem)
            if ins is None:
                continue
            gain = int(sp[c]) - ins[2]
            if gain > best_gain:
                best_gain, best = float(gain), ("ins", int(c), ins[0])

    if best is None:
        return None
    new = [list(r) for r in routes]
    new_loads = list(loads)
    if best[0] == "rm":
        v, i = best[1], best[2]
        c = new[v].pop(i)
        new_loads[v] -= int(problem.demand[c])
        return new, new_loads, (v,)
    c = best[1]
    ins2 = _cheapest_insertion(new, new_loads, c, problem)
    if ins2 is None:
        return None
    _insert(new, new_loads, c, ins2[0], ins2[1], problem)
    return new, new_loads, (ins2[0],)


def _routed_positions(routes: Routes) -> list[tuple[int, int]]:
    return [(v, i) for v, r in enumerate(routes) for i in range(len(r))]


def _relocate(
    routes: Routes, loads: list[int], problem: VRPProblem, rng: np.random.Generator
) -> Neighbor | None:
    """Rotalar-arasi: bir konteyneri baska araca/pozisyona tasi (kapasite-fizibil)."""
    pos = _routed_positions(routes)
    if not pos:
        return None
    v, i = pos[int(rng.integers(0, len(pos)))]
    c = routes[v][i]
    dc = int(problem.demand[c])
    tv = int(rng.integers(0, len(routes)))
    new = [list(r) for r in routes]
    new_loads = list(loads)
    new[v].pop(i)
    new_loads[v] -= dc
    if new_loads[tv] + dc > problem.capacity:
        return None
    tpos = int(rng.integers(0, len(new[tv]) + 1))
    new[tv].insert(tpos, c)
    new_loads[tv] += dc
    return new, new_loads, (v, tv)


def _swap(
    routes: Routes, loads: list[int], problem: VRPProblem, rng: np.random.Generator
) -> Neighbor | None:
    """Rotalar-arasi: iki konteyneri yer degistir (kapasite-fizibil)."""
    pos = _routed_positions(routes)
    if len(pos) < 2:
        return None
    a, b = rng.choice(len(pos), size=2, replace=False)
    v1, i1 = pos[int(a)]
    v2, i2 = pos[int(b)]
    c1, c2 = routes[v1][i1], routes[v2][i2]
    if v1 == v2:
        new = [list(r) for r in routes]
        new[v1][i1], new[v1][i2] = c2, c1
        return new, list(loads), (v1,)  # ayni rota, yuk degismez
    d1, d2 = int(problem.demand[c1]), int(problem.demand[c2])
    new_loads = list(loads)
    new_loads[v1] += d2 - d1
    new_loads[v2] += d1 - d2
    if new_loads[v1] > problem.capacity or new_loads[v2] > problem.capacity:
        return None
    new = [list(r) for r in routes]
    new[v1][i1], new[v2][i2] = c2, c1
    return new, new_loads, (v1, v2)


def _or_opt(
    routes: Routes, loads: list[int], problem: VRPProblem, rng: np.random.Generator
) -> Neighbor | None:
    """Rota-ici/arasi: 1-3 ardisik konteyneri baska yere tasi (kapasite-fizibil)."""
    nonempty = [v for v, r in enumerate(routes) if r]
    if not nonempty:
        return None
    v = int(nonempty[int(rng.integers(0, len(nonempty)))])
    seg_len = int(rng.integers(1, min(3, len(routes[v])) + 1))
    start = int(rng.integers(0, len(routes[v]) - seg_len + 1))
    seg = routes[v][start : start + seg_len]
    seg_demand = int(sum(int(problem.demand[c]) for c in seg))
    tv = int(rng.integers(0, len(routes)))
    new = [list(r) for r in routes]
    new_loads = list(loads)
    del new[v][start : start + seg_len]
    new_loads[v] -= seg_demand
    if new_loads[tv] + seg_demand > problem.capacity:
        return None
    tpos = int(rng.integers(0, len(new[tv]) + 1))
    new[tv][tpos:tpos] = seg
    new_loads[tv] += seg_demand
    return new, new_loads, (v, tv)


def _route_distance(route: list[int], problem: VRPProblem) -> int:
    """Tek rota mesafesi: Garaj -> konteynerler -> Dokum -> Garaj (metre)."""
    if not route:
        return 0
    dist = problem.dist
    d = int(dist[problem.depot_index, problem.node_of(route[0])])
    for a, b in pairwise(route):
        d += int(dist[problem.node_of(a), problem.node_of(b)])
    d += int(dist[problem.node_of(route[-1]), problem.dump_index])
    d += int(dist[problem.dump_index, problem.depot_index])
    return d


def _two_opt_route(
    route: list[int], problem: VRPProblem, max_pass: int, window: int
) -> list[int]:
    """KISA 2-opt (E6): yalnizca <= window uzunlukta segmentleri ters cevir.

    Tam 2-opt 187 dugumde O(k^3) -> cok yavas; pencereli surum O(k*window*k) ile
    hizli ve yerel iyilesmelerin cogunu yakalar.
    """
    if len(route) < 3:
        return list(route)
    best = list(route)
    best_d = _route_distance(best, problem)
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, min(i + window, len(best))):
                cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                cd = _route_distance(cand, problem)
                if cd < best_d:
                    best, best_d = cand, cd
                    improved = True
    return best


class ABCSolver(Solver):
    """X1 abc-basic / X2 abc-ls. Operatorler: toggle-skip, relocate, swap, or-opt.

    use_local_search=True (X2) -> scout ve greedy tohum sonrasi kisa 2-opt (E6 ablation).
    """

    def __init__(
        self,
        num_vehicles: int,
        colony_size: int,
        limit: int,
        time_limit_sec: float,
        rng: np.random.Generator,
        *,
        use_local_search: bool = False,
        local_search_iters: int = 6,
        local_search_window: int = 8,
    ) -> None:
        self._num_vehicles = num_vehicles
        self._n_sources = max(2, colony_size // 2)
        self._limit = limit
        self._time_limit = time_limit_sec
        self._rng = rng
        self._use_ls = use_local_search
        self._ls_iters = local_search_iters
        self._ls_window = local_search_window
        self.code = "X2" if use_local_search else "X1"
        self.name = "abc-ls" if use_local_search else "abc-basic"
        self.iterations = 0  # loglanir (durdurmaz)

    def _cost(self, problem: VRPProblem, routes: Routes) -> tuple[int, bool]:
        r: EvalResult = Evaluator(problem).evaluate(Solution.from_lists(routes))
        return r.total_cost, r.feasible

    def _maybe_ls(self, problem: VRPProblem, routes: Routes) -> Routes:
        """abc-ls: her rotaya kisa 2-opt uygula (yuk/skip degismez)."""
        if not self._use_ls:
            return routes
        return [_two_opt_route(r, problem, self._ls_iters, self._ls_window) for r in routes]

    def _neighbor(
        self, routes: Routes, loads: list[int], problem: VRPProblem, non_must: list[int]
    ) -> Neighbor | None:
        """Operator sec. toggle daha sik (skip karari asil deger); yonlu.

        Kabul edilirse etkilenen rotalara (X2) hedefli 2-opt uygulanir - skip'in
        actigi 'delik' hemen kapatilir.
        """
        op = int(self._rng.integers(0, 5))
        if op <= 1:  # yonlu toggle-skip (2/5 olasilik)
            nb = _directed_toggle(routes, loads, problem, self._rng, non_must)
        elif op == 2:
            nb = _relocate(routes, loads, problem, self._rng)
        elif op == 3:
            nb = _swap(routes, loads, problem, self._rng)
        else:
            nb = _or_opt(routes, loads, problem, self._rng)
        if nb is None or not self._use_ls:
            return nb
        # Hedefli LS: yalnizca etkilenen rotalara pencereli 2-opt (delik kapansin)
        routes2, loads2, affected = nb
        for v in set(affected):
            routes2[v] = _two_opt_route(routes2[v], problem, self._ls_iters, self._ls_window)
        return routes2, loads2, affected

    def solve(self, problem: VRPProblem) -> Solution:
        rng = self._rng
        non_must = [c for c in range(problem.n_containers) if not bool(problem.must_visit[c])]
        ns = self._n_sources

        # Baslangic: %20 greedy, %80 rastgele-fizibil
        n_greedy = max(1, ns // 5)
        pop: list[Routes] = []
        pop_loads: list[list[int]] = []
        for i in range(ns):
            if i < n_greedy:
                r, ld = _greedy_all(problem, self._num_vehicles)
            else:
                r, ld = _random_feasible(problem, self._num_vehicles, rng, non_must)
            pop.append(self._maybe_ls(problem, r))
            pop_loads.append(ld)
        costs = [self._cost(problem, r)[0] for r in pop]
        trials = [0] * ns

        best_i = int(np.argmin(costs))
        best_routes = [list(r) for r in pop[best_i]]
        best_cost = costs[best_i]

        start = time.monotonic()
        while time.monotonic() - start < self._time_limit:
            self.iterations += 1
            # Employed
            for i in range(ns):
                self._try_neighbor(problem, pop, pop_loads, costs, trials, i, non_must)
            # Onlooker (fitness-orantili secim)
            fitness = np.array([1.0 / (1.0 + c) for c in costs])
            probs = fitness / fitness.sum()
            for _ in range(ns):
                i = int(rng.choice(ns, p=probs))
                self._try_neighbor(problem, pop, pop_loads, costs, trials, i, non_must)
            # Scout (abc-ls: yeni cozume kisa 2-opt)
            for i in range(ns):
                if trials[i] > self._limit:
                    r, ld = _random_feasible(problem, self._num_vehicles, rng, non_must)
                    r = self._maybe_ls(problem, r)
                    c, feas = self._cost(problem, r)
                    if feas:
                        pop[i], pop_loads[i], costs[i], trials[i] = r, ld, c, 0
            # En iyiyi guncelle
            mi = int(np.argmin(costs))
            if costs[mi] < best_cost:
                best_cost = costs[mi]
                best_routes = [list(r) for r in pop[mi]]

        return Solution.from_lists(best_routes)

    def _try_neighbor(
        self,
        problem: VRPProblem,
        pop: list[Routes],
        pop_loads: list[list[int]],
        costs: list[int],
        trials: list[int],
        i: int,
        non_must: list[int],
    ) -> None:
        """Bir komsu uret, fizibil ve daha iyiyse degistir (greedy secim)."""
        nb = self._neighbor(pop[i], pop_loads[i], problem, non_must)
        if nb is None:
            trials[i] += 1
            return
        routes, loads, _affected = nb
        cost, feasible = self._cost(problem, routes)
        if feasible and cost < costs[i]:
            pop[i], pop_loads[i], costs[i], trials[i] = routes, loads, cost, 0
        else:
            trials[i] += 1
