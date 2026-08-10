"""VRPProblem + build_problem + avg_insertion_cost testleri."""

from __future__ import annotations

import numpy as np
import pytest

from domain.problem import VRPProblem, avg_insertion_cost, build_problem
from tests.factory import vrp

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


def test_avg_insertion_cost_k1() -> None:
    # konteyner alt-matrisi (1..3): en yakin 1 komsu
    out = avg_insertion_cost(DIST, n_containers=3, k=1)
    # k0->{5,15}=5, k1->{5,10}=5, k2->{10,15}=10
    assert out.tolist() == [5, 5, 10]


def test_avg_insertion_cost_k_clamps() -> None:
    out = avg_insertion_cost(DIST, n_containers=3, k=99)  # k > n-1 -> hepsi
    # k0->mean(5,15)=10, k1->mean(5,10)=7.5->8, k2->mean(10,15)=12.5->12
    assert out.tolist() == [10, 8, 12]


def test_avg_insertion_cost_single_container() -> None:
    d = np.array([[0, 10, 20], [10, 0, 30], [20, 30, 0]], dtype=np.int64)
    assert avg_insertion_cost(d, n_containers=1, k=3).tolist() == [0]


def test_build_problem_skip_penalty() -> None:
    p = build_problem(
        dist=DIST,
        travel_time=DIST,
        demand=np.array([100, 200, 300], dtype=np.int64),
        service_time=np.array([90, 90, 90], dtype=np.int64),
        volume=np.array([1100, 1100, 1100], dtype=np.int64),
        must_visit=np.array([False, False, False], dtype=np.bool_),
        capacity=65000,
        shift_limit=28800,
        skip_lambda=2.0,
        insertion_k=1,
        # Yakit: nominal oran TAM 10 mL/m cikacak sekilde secildi (elle hesap icin)
        #   nominal_mass = 65000 L * 0.1 kg/L * 0.5 = 3250 kg
        #   nominal = 3.5 + 0.002 * 3250 = 10.0 mL/m
        kg_per_liter=0.1,
        fuel_base_ml_per_m=3.5,
        fuel_slope_ml_per_m_per_kg=0.002,
        stop_start_ml=25,
        compaction_ml_per_liter=0.03,
        nominal_load_ratio=0.5,
    )
    # FAZ 2 (M.7): skip_penalty MILILITRE yakit biriminde - metre degil.
    # skip_penalty = rint(lambda * demand/volume * ins * nominal), ins=[5,5,10]
    # [0.909*10, 1.818*10, 5.454*10] = [9.09, 18.18, 54.54] -> [9, 18, 55]
    assert p.nominal_ml_per_m == 10.0
    assert p.skip_penalty.tolist() == [9, 18, 55]
    assert p.mass_kg.tolist() == [10.0, 20.0, 30.0]
    assert p.n_containers == 3
    assert p.depot_index == 0
    assert p.dump_index == 4
    assert p.node_of(2) == 3


def test_validate_bad_matrix_shape() -> None:
    p = vrp(
        dist=np.zeros((3, 3), dtype=np.int64),  # yanlis: N=3 icin 5x5 olmali
        travel_time=np.zeros((3, 3), dtype=np.int64),
        demand=np.zeros(3, dtype=np.int64),
        volume=np.full(3, 1100, dtype=np.int64),
        service_time=np.zeros(3, dtype=np.int64),
        skip_penalty=np.zeros(3, dtype=np.int64),
        must_visit=np.zeros(3, dtype=np.bool_),
        capacity=100,
        shift_limit=100,
    )
    with pytest.raises(ValueError, match="matris"):
        p.validate()


def test_validate_bad_container_array_shape() -> None:
    p = vrp(
        dist=np.zeros((5, 5), dtype=np.int64),
        travel_time=np.zeros((5, 5), dtype=np.int64),
        demand=np.zeros(3, dtype=np.int64),
        volume=np.full(3, 1100, dtype=np.int64),
        service_time=np.zeros(2, dtype=np.int64),  # yanlis uzunluk
        skip_penalty=np.zeros(3, dtype=np.int64),
        must_visit=np.zeros(3, dtype=np.bool_),
        capacity=100,
        shift_limit=100,
    )
    with pytest.raises(ValueError, match="service_time"):
        p.validate()


def test_validate_bad_scalars() -> None:
    p = vrp(
        dist=np.zeros((5, 5), dtype=np.int64),
        travel_time=np.zeros((5, 5), dtype=np.int64),
        demand=np.zeros(3, dtype=np.int64),
        volume=np.full(3, 1100, dtype=np.int64),
        service_time=np.zeros(3, dtype=np.int64),
        skip_penalty=np.zeros(3, dtype=np.int64),
        must_visit=np.zeros(3, dtype=np.bool_),
        capacity=0,  # gecersiz
        shift_limit=100,
    )
    with pytest.raises(ValueError, match="pozitif"):
        p.validate()


def _fuel_problem(**fuel: float | int) -> VRPProblem:
    """Yakit katsayisi dogrulamasi icin iskelet problem (FAZ 2, M.8)."""
    return vrp(
        dist=np.zeros((5, 5), dtype=np.int64),
        travel_time=np.zeros((5, 5), dtype=np.int64),
        demand=np.zeros(3, dtype=np.int64),
        volume=np.full(3, 1100, dtype=np.int64),
        service_time=np.zeros(3, dtype=np.int64),
        skip_penalty=np.zeros(3, dtype=np.int64),
        must_visit=np.zeros(3, dtype=np.bool_),
        capacity=100,
        shift_limit=100,
        **fuel,
    )


def test_validate_bad_fuel_rates() -> None:
    """Yakit oranlari pozitif olmali - sifir oran sessizce 'bedava yakit' demektir."""
    with pytest.raises(ValueError, match="yakit oranlari pozitif"):
        _fuel_problem(fuel_base_ml_per_m=0.0).validate()
    with pytest.raises(ValueError, match="yakit oranlari pozitif"):
        _fuel_problem(nominal_ml_per_m=0.0).validate()


def test_validate_negative_fuel_slope_or_compaction() -> None:
    """Negatif egim/sikistirma = yuk tasimak yakit KAZANDIRIR; fizik disi."""
    with pytest.raises(ValueError, match="negatif olamaz"):
        _fuel_problem(fuel_slope_ml_per_m_per_kg=-1.0).validate()
    with pytest.raises(ValueError, match="negatif olamaz"):
        _fuel_problem(compaction_ml_per_liter=-0.5).validate()
