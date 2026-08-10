"""Solver arayuzu - tum cozuculer ayni imzayi uygular.

    solve(problem: VRPProblem) -> Solution

Simulator hangi cozucunun calistigini bilmez. Bir cozucuyu degistirmek sistemin
geri kalanini etkilemez.

Cozucular kendi maliyetini HESAPLAMAZ (Bolum G2) - degerlendirme Evaluator'da.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.problem import VRPProblem
from domain.solution import Solution


class NoFeasibleSolutionError(RuntimeError):
    """Cozucu FIZIBIL cozum bulamadi (or. talep filo kapasitesini asiyor).

    Kural: sistem HATA VERIR VE DURUR, sessizce cop birakmaz.
    Bos rota dondurmek 'sifir mesafe, %100 tasarruf' gibi gorunur - bir hatanin
    zafer kiligina girdigi en tehlikeli durum (Bolum 10).
    """


class Solver(ABC):
    """Tum cozucuierin taban sinifi."""

    code: str = ""   # deney kodu: B0, B1, B2, X1, X2
    name: str = ""

    @abstractmethod
    def solve(self, problem: VRPProblem) -> Solution:
        """Bir gunluk problemi coz, Solution dondur.

        Fizibil cozum yoksa NoFeasibleSolutionError yukseltir - bos rota DONDURMEZ.
        """
        ...
