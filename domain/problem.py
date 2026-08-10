"""VRPProblem - CEKIRDEK veri yapisi.

SADECE np.ndarray (int64/float64/bool) + skaler sayilar. Sinif yalnizca bu
dizileri paketler; sicak dongudeki hesaplar (evaluator) dizilerle calisir
(numba-hazir).

Dugum sirasi (sabit):
    index 0            -> garaj (depot)
    index 1..N         -> konteynerler
    index N+1          -> dokum (dump)

Konteyner j (0-tabanli problem indeksi) -> matris dugumu j+1.

Birimler: mesafe metre, sure saniye, hacim/talep litre - hepsi int.

FAZ 2: amac fonksiyonu birimi TAM SAYI MILILITRE yakit.
Yakit katsayilari burada SKALER olarak tasinir; konteyner kutlesi (kg) sinir
gecisinde onceden hesaplanip dizi olarak verilir (sicak dongude carpma olmasin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class VRPProblem:
    """Bir gunluk rotalama problemi - cikplak diziler.

    Tum konteyner-basi diziler N uzunlugunda; matris (N+2)x(N+2).
    """

    dist: IntArr            # (K, K) metre, asimetrik
    travel_time: IntArr     # (K, K) saniye, asimetrik
    demand: IntArr          # (N,) bugunku talep (litre)
    mass_kg: FloatArr       # (N,) talebin kutlesi - yakit modeli kutleyle calisir
    volume: IntArr          # (N,) konteyner (nokta) hacmi - doluluk orani icin
    service_time: IntArr    # (N,) saniye/konteyner
    skip_penalty: IntArr    # (N,) atlama cezasi (mL YAKIT - Faz 2'de metre degil!)
    must_visit: BoolArr     # (N,) atlanamaz maske
    capacity: int           # etkin arac kapasitesi (litre)
    shift_limit: int        # vardiya limiti (saniye)

    # --- FAZ 2 yakit katsayilari (skaler) ---
    fuel_base_ml_per_m: float        # bos kamyon surus orani (mL/m)
    fuel_slope_ml_per_m_per_kg: float  # tasinan kutle basina ek oran
    stop_start_ml: int               # durak basi dur-kalk yakiti
    compaction_ml_per_liter: float   # sikistirma cevrimi (toplanan litre basina)
    nominal_ml_per_m: float          # SABIT-YUK orani: B2 ark maliyeti + skip_penalty
                                     # referansi (M.7: OR-Tools yuk-kor yaklasimi cozer)

    @property
    def n_containers(self) -> int:
        return int(self.demand.shape[0])

    @property
    def depot_index(self) -> int:
        return 0

    @property
    def dump_index(self) -> int:
        return self.n_containers + 1

    def node_of(self, container: int) -> int:
        """Konteyner (0-tabanli) -> matris dugum indeksi."""
        return container + 1

    def validate(self) -> None:
        """Yapisal tutarlilik (sinir gecisinde bir kez cagrilir)."""
        n = self.n_containers
        k = n + 2
        if self.dist.shape != (k, k) or self.travel_time.shape != (k, k):
            raise ValueError(f"matris {self.dist.shape}, beklenen {(k, k)}")
        for name, arr in (
            ("mass_kg", self.mass_kg),
            ("volume", self.volume),
            ("service_time", self.service_time),
            ("skip_penalty", self.skip_penalty),
            ("must_visit", self.must_visit),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} {arr.shape}, beklenen {(n,)}")
        if self.capacity <= 0 or self.shift_limit <= 0:
            raise ValueError("capacity ve shift_limit pozitif olmali")
        if self.fuel_base_ml_per_m <= 0 or self.nominal_ml_per_m <= 0:
            raise ValueError("yakit oranlari pozitif olmali (M.8)")
        if self.fuel_slope_ml_per_m_per_kg < 0 or self.compaction_ml_per_liter < 0:
            raise ValueError("yakit egimi ve sikistirma negatif olamaz")


def avg_insertion_cost(dist: IntArr, n_containers: int, k: int) -> IntArr:
    """Her konteynerin en yakin k KONTEYNER komsusuna ort. mesafesi (metre).

    skip_penalty'yi mesafeyle ayni birime tasir. Cikti (N,) int64.
    Konteyner dugumleri 1..N; garaj/dokum haric.
    """
    sub = dist[1 : n_containers + 1, 1 : n_containers + 1].astype(np.float64)
    np.fill_diagonal(sub, np.inf)  # kendine mesafe haric
    sub.sort(axis=1)
    kk = min(k, n_containers - 1) if n_containers > 1 else 1
    if n_containers <= 1:
        return np.zeros(n_containers, dtype=np.int64)
    nearest = sub[:, :kk]
    return np.rint(nearest.mean(axis=1)).astype(np.int64)


def build_problem(
    dist: IntArr,
    travel_time: IntArr,
    demand: IntArr,
    service_time: IntArr,
    volume: IntArr,
    must_visit: BoolArr,
    *,
    capacity: int,
    shift_limit: int,
    skip_lambda: float,
    insertion_k: int,
    kg_per_liter: float,
    fuel_base_ml_per_m: float,
    fuel_slope_ml_per_m_per_kg: float,
    stop_start_ml: int,
    compaction_ml_per_liter: float,
    nominal_load_ratio: float,
) -> VRPProblem:
    """SINIR: duz diziler -> VRPProblem.

    skip_penalty(i) = lambda * doluluk_orani(i) * ort_ekleme_maliyeti(i) * nominal_oran
        doluluk_orani(i) = demand(i) / volume(i)

    FAZ 2 (M.7): amac fonksiyonu mL yakit birimindedir, bu yuzden skip_penalty de
    mL'ye tasinir - SABIT-YUK orani (nominal_ml_per_m) ile carpilarak. Metre ile mL
    toplanirsa lambda sessizce anlamini degistirir.
    > Faz 1'de kalibre edilen lambda TASINMAZ; yeniden taranmalidir (M.7 Adim 6).

    Kabuk (sim) bu fonksiyonu gunde bir kez cagirir; diziler burada uretilir.
    """
    n = int(demand.shape[0])
    nominal_mass_kg = capacity * kg_per_liter * nominal_load_ratio
    nominal_ml_per_m = (
        fuel_base_ml_per_m + fuel_slope_ml_per_m_per_kg * nominal_mass_kg
    )

    fill_ratio = demand.astype(np.float64) / volume.astype(np.float64)
    ins = avg_insertion_cost(dist, n, insertion_k).astype(np.float64)
    skip_penalty = np.rint(
        skip_lambda * fill_ratio * ins * nominal_ml_per_m
    ).astype(np.int64)

    problem = VRPProblem(
        dist=dist.astype(np.int64),
        travel_time=travel_time.astype(np.int64),
        demand=demand.astype(np.int64),
        mass_kg=demand.astype(np.float64) * kg_per_liter,
        volume=volume.astype(np.int64),
        service_time=service_time.astype(np.int64),
        skip_penalty=skip_penalty,
        must_visit=must_visit.astype(np.bool_),
        capacity=int(capacity),
        shift_limit=int(shift_limit),
        fuel_base_ml_per_m=float(fuel_base_ml_per_m),
        fuel_slope_ml_per_m_per_kg=float(fuel_slope_ml_per_m_per_kg),
        stop_start_ml=int(stop_start_ml),
        compaction_ml_per_liter=float(compaction_ml_per_liter),
        nominal_ml_per_m=float(nominal_ml_per_m),
    )
    problem.validate()
    return problem
