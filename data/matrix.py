"""Asimetrik mesafe (m) / sure (sn) matrisi.

Dugum sirasi (sabit kural):
    index 0            -> garaj (depot)
    index 1..N         -> konteynerler
    index N+1          -> dokum sahasi (dump)

Konteyner/garaj/dokum en yakin yol agi dugumune eslenir; NetworkX ile dugumler
arasi en kisa yollar hesaplanir. Mesafe 'length' (m), sure 'travel_time' (sn)
kenar agirliklariyla ayri Dijkstra kosularindan gelir. Tam sayiya yuvarlanir
(OR-Tools float kabul etmez).

KABUK modulu. Ciktilar cekirdege sinir gecisinde diziyle girer.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox


@dataclass(frozen=True)
class DistanceMatrices:
    """(K x K) asimetrik matrisler; K = N konteyner + 2 (garaj, dokum)."""

    dist_m: np.ndarray      # int64, metre
    time_s: np.ndarray      # int64, saniye
    node_ids: np.ndarray    # her terminal noktanin eslendigi yol agi dugum id'si


def _prepare_graph(graph_proj: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Guclu bagli en buyuk bileseni al (inf mesafeyi onler) + hiz/sure ekle."""
    g = ox.truncate.largest_component(graph_proj, strongly=True)
    g = ox.routing.add_edge_speeds(g)
    g = ox.routing.add_edge_travel_times(g)
    return g


def build_matrices(
    graph_proj: nx.MultiDiGraph, points_xy: np.ndarray
) -> DistanceMatrices:
    """points_xy: (K, 2) projeli koordinatlar (0=garaj, 1..N=konteyner, K-1=dokum).

    Donen matrisler ayni siralamada (K x K).
    """
    g = _prepare_graph(graph_proj)
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    node_ids = np.asarray(ox.distance.nearest_nodes(g, X=xs, Y=ys), dtype=np.int64)

    k = len(points_xy)
    dist = np.zeros((k, k), dtype=np.int64)
    time = np.zeros((k, k), dtype=np.int64)

    # Kaynak dugum -> tum hedeflere en kisa; sadece terminal dugumleri okunur.
    uniq_sources = {int(n) for n in node_ids}
    dist_cache: dict[int, dict[int, float]] = {}
    time_cache: dict[int, dict[int, float]] = {}
    for src in uniq_sources:
        dist_cache[src] = nx.single_source_dijkstra_path_length(g, src, weight="length")
        time_cache[src] = nx.single_source_dijkstra_path_length(g, src, weight="travel_time")

    for i in range(k):
        si = int(node_ids[i])
        dmap = dist_cache[si]
        tmap = time_cache[si]
        for j in range(k):
            if i == j:
                continue
            sj = int(node_ids[j])
            if sj not in dmap:
                raise ValueError(
                    f"Dugum {i}->{j} ulasilamaz. Yol agi baglantisiz olabilir."
                )
            dist[i, j] = round(dmap[sj])
            time[i, j] = round(tmap[sj])

    return DistanceMatrices(dist_m=dist, time_s=time, node_ids=node_ids)
