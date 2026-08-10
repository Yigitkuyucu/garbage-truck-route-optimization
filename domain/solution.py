"""Solution - rota temsili + sinir donusumleri + pretty_print.

Cozum temsili: her arac icin konteyner indeksi dizisi (list[list[int]]).
Garaj/dokum implicit (her rota Garaj -> ... -> Dokum -> Garaj). Rotada olmayan
konteyner ATLANMIS sayilir.

Bu modul cekirdek maliyet hesabi ICERMEZ - sayilar Evaluator'dan gelir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Solution:
    """Her arac icin konteyner (0-tabanli) dizisi. Immutable."""

    routes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_lists(cls, routes: Sequence[Sequence[int]]) -> Solution:
        return cls(tuple(tuple(int(c) for c in r) for r in routes))

    @property
    def n_vehicles(self) -> int:
        return len(self.routes)

    def flat(self) -> tuple[IntArr, IntArr]:
        """Cekirdek kodlamasi: (routes_flat, route_lengths).

        routes_flat = tum araclarin konteynerleri ardarda; route_lengths = her
        aracin konteyner sayisi. Evaluator bunlari alir (numba-hazir).
        """
        lengths = np.array([len(r) for r in self.routes], dtype=np.int64)
        if lengths.sum() == 0:
            return np.empty(0, dtype=np.int64), lengths
        flat = np.array([c for r in self.routes for c in r], dtype=np.int64)
        return flat, lengths

    def visited_mask(self, n_containers: int) -> BoolArr:
        """Ziyaret edilen konteyner maskesi (N,)."""
        mask = np.zeros(n_containers, dtype=np.bool_)
        for r in self.routes:
            for c in r:
                mask[c] = True
        return mask


def decode_solution(routes_flat: IntArr, route_lengths: IntArr) -> Solution:
    """SINIR: cekirdek diziler -> Solution."""
    routes: list[tuple[int, ...]] = []
    off = 0
    for length in route_lengths:
        seg = routes_flat[off : off + int(length)]
        routes.append(tuple(int(c) for c in seg))
        off += int(length)
    return Solution(tuple(routes))
