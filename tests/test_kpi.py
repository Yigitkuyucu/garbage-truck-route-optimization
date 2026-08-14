"""sim/kpi.py testleri - toplama (feasible-only), saglik kontrolleri.

FAZ 2 (M.7): manset metrik YAKIT. %30 kurali yakit uzerinden, bolge-ici
tasarruf suphe esigi mesafe uzerinden (yan kontrol), ayrica yuk baglasimi capasi.

M.11: "yapisal tavan" kontrolu bir kategori hatasiydi (pay ile tasarrufu
kiyasliyordu); duzeltildi ve regresyon testi eklendi.
"""

from __future__ import annotations

from sim.engine import DayRecord, RunResult
from sim.kpi import aggregate, health_checks

CO2 = 2.68  # dizel emisyon faktoru (M.8)


def _rec(
    day: int,
    *,
    feasible: bool,
    intra_km: float,
    overflow: int = 0,
    fuel_l: float = 30.0,
    load_term_l: float = -0.3,
) -> DayRecord:
    return DayRecord(
        day=day, feasible=feasible,
        fuel_ml=int(fuel_l * 1000),
        fuel_travel_ml=int(fuel_l * 1000 * 0.75),
        fuel_stop_ml=int(fuel_l * 1000 * 0.13),
        fuel_compaction_ml=int(fuel_l * 1000 * 0.12),
        fuel_load_term_ml=int(load_term_l * 1000),
        total_distance=int((intra_km + 50) * 1000), fixed_distance=50_000,
        intra_distance=int(intra_km * 1000),
        generated_l=1000, collected_l=1000, n_visited=100, service_time_s=9000,
        n_skipped=87, overflow_events=overflow, overflow_l=0,
        mean_fill_pct=60.0, shift_util_pct=40.0,
    )


def _run(records: list[DayRecord], *, mass_ok: bool = True, overflow: int = 0) -> RunResult:
    return RunResult(
        records=records, total_generated_l=10, total_collected_l=6, remaining_l=4,
        mass_ok=mass_ok, total_overflow_events=overflow,
    )


def test_aggregate_feasible_only() -> None:
    # 2 feasible (intra 10, 12) + 1 infeasible (intra 99) -> ort sadece feasible
    recs = [
        _rec(0, feasible=True, intra_km=10.0),
        _rec(1, feasible=True, intra_km=12.0),
        _rec(2, feasible=False, intra_km=99.0),
    ]
    k = aggregate("X1", "abc", [_run(recs)], co2_kg_per_l=CO2)
    assert k.infeasible_days == 1
    assert k.n_days_total == 3
    assert abs(k.mean_intra_km - 11.0) < 1e-6  # (10+12)/2, infeasible haric


def test_aggregate_fuel_and_co2() -> None:
    """FAZ 2 manset: yakit ortalamasi + CO2 turevi + kalem ayrimi."""
    recs = [
        _rec(0, feasible=True, intra_km=10.0, fuel_l=30.0),
        _rec(1, feasible=True, intra_km=10.0, fuel_l=40.0),
    ]
    k = aggregate("X1", "abc", [_run(recs)], co2_kg_per_l=CO2)
    assert abs(k.mean_fuel_l - 35.0) < 1e-6
    assert abs(k.mean_co2_kg - 35.0 * CO2) < 1e-6
    # kalemler toplami yakita esit olmali (yuvarlama toleransiyla)
    kalem = k.mean_fuel_travel_l + k.mean_fuel_stop_l + k.mean_fuel_compaction_l
    assert abs(kalem - k.mean_fuel_l) < 0.01


def test_aggregate_overflow_and_mass() -> None:
    recs = [_rec(0, feasible=True, intra_km=10.0, overflow=3)]
    k = aggregate("B1", "esik", [_run(recs, mass_ok=False, overflow=3)], co2_kg_per_l=CO2)
    assert k.total_overflow_events == 3
    assert not k.mass_ok


def test_health_check_mass_and_overflow() -> None:
    b0 = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=10.0)])],
                   co2_kg_per_l=CO2)
    bad = aggregate(
        "B1", "esik", [_run([_rec(0, feasible=True, intra_km=9.5, overflow=5)],
                            mass_ok=False, overflow=5)],
        co2_kg_per_l=CO2,
    )
    warns = health_checks([b0, bad])
    assert any("KUTLE" in w for w in warns)
    assert any("TASMA" in w for w in warns)


def test_health_check_thirty_percent_rule_on_fuel() -> None:
    """FAZ 2: %30 kurali MANSET metrige (yakit) uygulanir (Bolum 10)."""
    b0 = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=10.0, fuel_l=30.0)])],
                   co2_kg_per_l=CO2)
    x1 = aggregate("X1", "abc", [_run([_rec(0, feasible=True, intra_km=9.0, fuel_l=15.0)])],
                   co2_kg_per_l=CO2)
    warns = health_checks([b0, x1])
    assert any("%30 KURALI" in w for w in warns)


def test_health_check_intra_saving_suspect() -> None:
    # B0 intra 10, X1 intra 5 -> %50 tasarruf > %40 suphe esigi -> uyari
    b0 = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=10.0)])],
                   co2_kg_per_l=CO2)
    x1 = aggregate("X1", "abc", [_run([_rec(0, feasible=True, intra_km=5.0)])],
                   co2_kg_per_l=CO2)
    warns = health_checks([b0, x1])
    assert any("BOLGE-ICI TASARRUF SUPHELI" in w for w in warns)


def test_intra_saving_below_suspect_is_clean() -> None:
    """M.11 kategori hatasi regresyonu: %26 bolge-ici tasarruf UYARI DEGILDIR.

    Eski kod bunu "yapisal tavan %17 asildi" diye isaretliyordu; %17 aslinda
    optimize-edilebilir PAY'di (bolge-ici/toplam), tasarruf esigi degil.
    Gercek kosuda B1 tam bu degeri uretti ve yanlis alarm verdi.
    """
    b0 = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=59.5)])],
                   co2_kg_per_l=CO2)
    b1 = aggregate("B1", "esik", [_run([_rec(0, feasible=True, intra_km=43.75)])],
                   co2_kg_per_l=CO2)
    warns = health_checks([b0, b1])
    assert not any("BOLGE-ICI" in w for w in warns)


def test_optimizable_share_is_measured_not_threshold() -> None:
    """Optimize-edilebilir pay: bolge-ici / toplam (M.11). Olcum, esik degil."""
    k = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=59.5)])],
                  co2_kg_per_l=CO2)
    # _rec: total = intra + 50 km sabit -> 59.5 / 109.5
    assert abs(k.optimizable_share - 59.5 / 109.5) < 1e-6


def test_health_check_load_term_anchor() -> None:
    """M.7 capasi: yuk baglasimi yakitin ~%1'i olmali; buyurse uyar."""
    k = aggregate(
        "X1", "abc",
        [_run([_rec(0, feasible=True, intra_km=10.0, fuel_l=30.0, load_term_l=-5.0)])],
        co2_kg_per_l=CO2,
    )
    warns = health_checks([k])
    assert any("YUK TERIMI" in w for w in warns)


def test_health_check_clean() -> None:
    # B0 10, X1 8.7 (%13 < %40); yakit esit; yuk terimi kucuk -> uyari yok
    b0 = aggregate("B0", "B0", [_run([_rec(0, feasible=True, intra_km=10.0)])],
                   co2_kg_per_l=CO2)
    x1 = aggregate("X1", "abc", [_run([_rec(0, feasible=True, intra_km=8.7)])],
                   co2_kg_per_l=CO2)
    assert health_checks([b0, x1]) == []
