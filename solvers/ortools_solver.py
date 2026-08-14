"""B2 - OR-Tools referans cozucu.

OR-Tools bir RAKIP degil, dogru cevabin nasil gorundugunu ogreten bir HATA
AYIKLAMA aracidir (G4).

Modelleme puf noktasi - dokumu donus-arkina katla:
    Rota = Garaj -> konteynerler -> Dokum -> Garaj. OR-Tools'a tek depolu standart
    CVRP verilir (dugumler: garaj + konteynerler). Dokum, "garaja donus" arkina
    gomulur:
        arc(garaj, c)   = dist[garaj, c]
        arc(c, c')      = dist[c, c']
        arc(c, garaj)   = dist[c, dokum] + dist[dokum, garaj]

FAZ 2: ark maliyeti mesafe DEGIL, YAKIT (mL).

    OR-Tools'un rotalama kutuphanesinde ark maliyeti bir boyutun (dimension)
    degerine BAGLANAMAZ - yani gercek yuk duyarli maliyeti dogrudan cozemez.
    Bu yuzden B2'ye o maliyetin EN IYI YUK-KOR YAKLASIMI verilir:

        ark_yakiti = round(nominal_ml_per_m * mesafe)      (sabit referans yuk)
                   + dur_kalk_ml                            (hedef dugum konteyner ise)
                   + round(sikistirma_ml_per_l * talep)     (hedef dugum konteyner ise)

    Boylece OR-Tools objektifi == Evaluator'in fuel_constload_ml + skip_cost
    degeridir - capraz dogrulama TAM SAYI olarak KESIN kalir (G2).

    ABC ise gercek yuk duyarli maliyeti optimize eder. Ikisi arasindaki fark
    tam olarak "OR-Tools'un erisemedigi terim"dir (EvalResult.fuel_load_term_ml).
    > M.7 olctu: bu terim bu olcekte yakitin ~%0.5-1'i. Kucuk oldugu BILINEREK
    > raporlanir; B2 one gecerse o da raporlanir.

skip_penalty -> disjunction cezasi (Faz 2'de mL biriminde). must_visit -> disjunction
YOK (zorunlu). Kapasite -> boyut. Vardiya -> zaman boyutu (seyahat + servis).

KABUK (sarmalayici). Maliyet OR-Tools'un degil, cozum Evaluator'dan gecer (G2).
"""

from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from domain.problem import VRPProblem
from domain.solution import Solution
from solvers.base import NoFeasibleSolutionError, Solver

# Ilk-cozum stratejisi zinciri (sirayla denenir; ilk basarili olan kullanilir).
# PARALLEL_CHEAPEST_INSERTION buyuk ve sikisik orneklerde tek calisan seceneklerden
# biri; PATH_CHEAPEST_ARC kucuk orneklerde hizli ve iyi baslangic verir.
_FIRST_SOLUTION_CHAIN = (
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
)


class ORToolsSolver(Solver):
    """B2 - OR-Tools referans ust siniri."""

    code = "B2"
    name = "OR-Tools"

    def __init__(self, num_vehicles: int, time_limit_sec: int) -> None:
        self._num_vehicles = num_vehicles
        self._time_limit = time_limit_sec
        self._last_objective = 0

    def solve(self, problem: VRPProblem) -> Solution:
        n = problem.n_containers
        depot = 0  # OR-Tools dugumu 0 = garaj; 1..n = konteynerler (c -> or-dugum c+1)
        dump = problem.dump_index
        dist = problem.dist
        tt = problem.travel_time

        num_nodes = n + 1  # garaj + konteynerler (dokum gomulu)
        manager = pywrapcp.RoutingIndexManager(num_nodes, self._num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        nominal = problem.nominal_ml_per_m
        stop_ml = problem.stop_start_ml
        compaction = problem.compaction_ml_per_liter

        def fuel_cb(from_index: int, to_index: int) -> int:
            """Ark yakiti (mL): sabit-yuk seyahat + hedef dugumun dur-kalk/sikistirmasi.

            Yuvarlama Evaluator ile BIREBIR ayni gruplamada (bkz. _route_metrics).
            """
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            if i == depot and j == depot:
                return 0
            if j == depot:  # garaja donus: dokum uzerinden - TEK ark olarak yuvarlanir
                return round(nominal * (int(dist[i, dump]) + int(dist[dump, depot])))
            # hedef bir konteyner: seyahat + o duragin dur-kalk ve sikistirma yakiti
            node_cost = stop_ml + round(compaction * int(problem.demand[j - 1]))
            return round(nominal * int(dist[i, j])) + node_cost

        transit_idx = routing.RegisterTransitCallback(fuel_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Kapasite boyutu
        def demand_cb(from_index: int) -> int:
            node = manager.IndexToNode(from_index)
            return 0 if node == depot else int(problem.demand[node - 1])

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [problem.capacity] * self._num_vehicles, True, "Capacity"
        )

        # Vardiya (zaman) boyutu: seyahat (dokum gomulu) + servis <= shift_limit
        def time_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            if i == depot and j == depot:
                travel = 0
            elif j == depot:
                travel = int(tt[i, dump]) + int(tt[dump, depot])
            else:
                travel = int(tt[i, j])
            service = 0 if j == depot else int(problem.service_time[j - 1])
            return travel + service

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_idx, 0, problem.shift_limit, True, "Time")

        # skip_penalty: must_visit disinda her konteyner atlanabilir (disjunction)
        for c in range(n):
            if not bool(problem.must_visit[c]):
                node_index = manager.NodeToIndex(c + 1)
                routing.AddDisjunction([node_index], int(problem.skip_penalty[c]))

        params = pywrapcp.DefaultRoutingSearchParameters()
        # Ilk-cozum stratejisi SIRAYLA denenir. Gerekce olculdu: 465 dugum ve
        # ~%90 kapasite kullaniminda PATH_CHEAPEST_ARC 120 sn'de bile fizibil
        # cozum bulamiyor; ekleme (insertion) tabanli stratejiler buluyor.
        # Kucuk orneklerde ilk strateji zaten hemen basarili olur, dolayisiyla
        # zincir yalnizca zor orneklerde devreye girer.
        assignment = None
        for strategy in _FIRST_SOLUTION_CHAIN:
            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = strategy
            params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            params.time_limit.FromSeconds(self._time_limit)
            assignment = routing.SolveWithParameters(params)
            if assignment is not None:
                break

        if assignment is None:
            # Bos rota DONDURULMEZ: "0 mesafe / %100 tasarruf" gibi gorunur ve bir
            # hatayi zafer kiligina sokar.
            self._last_objective = 0
            # Mesaj SEBEBI soylesin: baglayan sey zorunlu ziyaret yuku mu, yoksa
            # cozucunun sure limiti mi? Ikisi tamamen farkli mudahale gerektirir.
            fleet_l = self._num_vehicles * problem.capacity
            mandatory = int(problem.demand[problem.must_visit].sum())
            need = mandatory / problem.capacity
            cause = (
                "KAPASITE: zorunlu yuk filoyu asiyor -> num_vehicles artir"
                if mandatory > fleet_l
                else f"SURE LIMITI: kapasite yeterli ({need:.2f} <= "
                     f"{self._num_vehicles} arac) ama cozucu {self._time_limit} sn'de "
                     f"cozum bulamadi -> limiti artir"
            )
            raise NoFeasibleSolutionError(
                f"OR-Tools fizibil cozum bulamadi. SEBEP -> {cause}. "
                f"[{n} nokta, zorunlu {int(problem.must_visit.sum())} nokta / "
                f"{mandatory:,} L = {need:.2f} arac, filo {self._num_vehicles} x "
                f"{problem.capacity:,} L = {fleet_l:,} L, "
                f"toplam doluluk {int(problem.demand.sum()):,} L]"
            )
        self._last_objective = int(assignment.ObjectiveValue())

        routes: list[list[int]] = []
        for v in range(self._num_vehicles):
            idx = routing.Start(v)
            route: list[int] = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != depot:
                    route.append(node - 1)  # OR-dugum -> konteyner indeksi
                idx = assignment.Value(routing.NextVar(idx))
            routes.append(route)
        return Solution.from_lists(routes)

    def last_objective(self) -> int:
        """Son cozumun OR-Tools objektif degeri (capraz dogrulama icin, G2)."""
        return self._last_objective
