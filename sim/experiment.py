"""Deney orkestratoru.

Iki kademeli protokol (butce):
  - UZUN: tum ufuk (warmup+report) x seed, kisa cozucu limiti -> KPI trendi.
  - ODAKLI: birkac temsili gun x seed, tam 60 sn -> adil F2 karsilastirmasi.

lambda taramasi -> Pareto (mesafe vs tasma); calisma noktasi tasma=0 en dusuk mesafe.

Cikti: runs/<ts>_<hash>/ altina config kopyasi + CSV'ler + saglik raporu +
4-metrikli tablo + Pareto (CSV + PNG).

KABUK modulu.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # basliksiz PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from data.build import build_dataset
from data.dataset import BuiltDataset
from sim.engine import RunResult, Simulator
from sim.kpi import SolverKPIs, aggregate, health_checks
from solvers.abc_solver import ABCSolver
from solvers.base import Solver
from solvers.greedy import GreedySolver
from solvers.ortools_solver import ORToolsSolver
from solvers.threshold_greedy import ThresholdGreedySolver

SOLVER_ORDER = ["B0", "B1", "B2", "X1", "X2"]
SOLVER_NAMES = {
    "B0": "Sabit rota", "B1": "Esik+greedy", "B2": "OR-Tools",
    "X1": "abc-basic", "X2": "abc-ls",
}


def _fill_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng([seed, i, 0])


def _solver_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng([seed, i, 1])


def make_solver(code: str, cfg: Config, time_limit: float, rng: np.random.Generator) -> Solver:
    nv = cfg.fleet.num_vehicles
    if code == "B0":
        return GreedySolver(nv)
    if code == "B1":
        return ThresholdGreedySolver(nv, cfg.solvers.b1_threshold)
    if code == "B2":
        return ORToolsSolver(nv, int(time_limit))
    if code == "X1":
        return ABCSolver(nv, cfg.abc.colony_size, cfg.abc.limit, time_limit, rng)
    if code == "X2":
        return ABCSolver(
            nv, cfg.abc.colony_size, cfg.abc.limit, time_limit, rng,
            use_local_search=True, local_search_iters=cfg.abc.local_search_iters,
            local_search_window=cfg.abc.local_search_window,
        )
    raise ValueError(f"bilinmeyen cozucu: {code}")


def run_solver(
    dataset: BuiltDataset, cfg: Config, code: str, num_seeds: int,
    time_limit: float, skip_lambda: float,
) -> list[RunResult]:
    """Bir cozucuyu num_seeds seed'de kostur. Ayni seed'de fill AYNIDIR (D5)."""
    sim = Simulator(dataset, cfg)
    out: list[RunResult] = []
    for i in range(num_seeds):
        solver = make_solver(code, cfg, time_limit, _solver_rng(cfg.seed, i))
        out.append(sim.run(solver, _fill_rng(cfg.seed, i), skip_lambda))
    return out


@dataclass
class ParetoPoint:
    skip_lambda: float
    mean_fuel_l: float       # FAZ 2 calisma noktasi bunun uzerinden secilir
    mean_intra_km: float     # yan metrik (Faz 1 karsilastirilabilirligi icin)
    overflow_events: int


def lambda_sweep(
    dataset: BuiltDataset, cfg: Config, num_seeds: int, time_limit: float,
    sweep_solver: str = "B2",
) -> tuple[list[ParetoPoint], float]:
    """lambda tara -> (Pareto noktalari, lambda*).

    FAZ 2 calisma noktasi: tasma=0 olan en dusuk YAKIT (manset metrik degisti, M.7).
    > Faz 1'de kalibre edilen lambda TASINMAZ: skip_penalty artik mL biriminde.
    """
    s = cfg.skip_penalty.lambda_sweep
    lambdas = np.geomspace(s.start, s.stop, s.num)
    points: list[ParetoPoint] = []
    for lam in lambdas:
        results = run_solver(dataset, cfg, sweep_solver, num_seeds, time_limit, float(lam))
        k = aggregate(
            sweep_solver, SOLVER_NAMES[sweep_solver], results,
            co2_kg_per_l=cfg.fuel.co2_kg_per_l,
        )
        points.append(
            ParetoPoint(
                float(lam), k.mean_fuel_l, k.mean_intra_km, k.total_overflow_events
            )
        )
    feasible = [p for p in points if p.overflow_events == 0]
    if feasible:
        lam_star = min(feasible, key=lambda p: p.mean_fuel_l).skip_lambda
    else:  # hicbiri tasma=0 degilse en yuksek lambda (en guvenli)
        lam_star = float(lambdas[-1])
    return points, lam_star


def _kpi_table(kpis: list[SolverKPIs]) -> pd.DataFrame:
    """FAZ 2 tablosu: MANSET yakit/CO2 once, mesafe yan metrik (M.7).

    Yakit kalemleri ayrisik (seyahat/dur-kalk/sikistirma) - birlestirilirse
    sikistirma sabiti tasarruf yuzdesini yaniltici sekilde kucultur (M.8 #2).
    """
    rows = []
    for k in kpis:
        rows.append({
            "kod": k.code,
            "cozucu": k.name,
            "yakit_L": round(k.mean_fuel_l, 2),
            "yakit_std": round(k.std_fuel_l, 2),
            "co2_kg": round(k.mean_co2_kg, 1),
            "yk_seyahat": round(k.mean_fuel_travel_l, 2),
            "yk_durkalk": round(k.mean_fuel_stop_l, 2),
            "yk_sikistirma": round(k.mean_fuel_compaction_l, 2),
            "yuk_terimi_L": round(k.mean_load_term_l, 3),
            "durak": round(k.mean_stops, 1),
            "servis_dk": round(k.mean_service_s / 60, 1),
            "doluluk_%": round(k.mean_fill_pct, 1),
            "bolge_ici_km": round(k.mean_intra_km, 2),
            "toplam_km": round(k.mean_total_km, 2),
            "sabit_km": round(k.mean_fixed_km, 2),
            "vardiya_%": round(k.mean_shift_util, 1),
            "infeasible_gun": k.infeasible_days,
            "tasma": k.total_overflow_events,
        })
    return pd.DataFrame(rows)


def _write_pareto(points: list[ParetoPoint], lam_star: float, out: Path) -> None:
    df = pd.DataFrame([
        {
            "lambda": p.skip_lambda,
            "yakit_L": p.mean_fuel_l,
            "bolge_ici_km": p.mean_intra_km,
            "tasma": p.overflow_events,
        }
        for p in points
    ])
    df.to_csv(out / "lambda_pareto.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xscale("log")
    ax1.plot(df["lambda"], df["yakit_L"], "o-", color="tab:blue", label="yakit L/gun")
    ax1.set_xlabel("lambda (skip cezasi olcegi)")
    ax1.set_ylabel("yakit (L/gun)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["lambda"], df["tasma"], "s--", color="tab:red", label="tasma")
    ax2.set_ylabel("tasma olayi", color="tab:red")
    ax1.axvline(lam_star, color="green", ls=":", label=f"lambda*={lam_star:.2g}")
    ax1.set_title("lambda kalibrasyonu - Pareto (yakit vs tasma)")
    fig.tight_layout()
    fig.savefig(out / "lambda_pareto.png", dpi=120)
    plt.close(fig)


def sensitivity_analysis(
    dataset: BuiltDataset, cfg: Config, num_seeds: int, time_limit: float,
    lam: float, solver_code: str = "B2",
) -> pd.DataFrame:
    """Hijyen tavani duyarliligi (kullanici istegi): cap 3/5/7 -> (tasarruf, tasma,
    en-uzun-bekleme). Beklenti: tasarruf benzer -> 'sonuclar tavana dayanikli'."""
    co2 = cfg.fuel.co2_kg_per_l
    b0 = aggregate(
        "B0", "B0", run_solver(dataset, cfg, "B0", num_seeds, time_limit, lam),
        co2_kg_per_l=co2,
    )
    rows = []
    for cap in cfg.constraints.hygiene_cap_sensitivity:
        sim = Simulator(dataset, cfg)
        results: list[RunResult] = []
        for i in range(num_seeds):
            sim.hygiene_cap_override(cap)
            solver = make_solver(solver_code, cfg, time_limit, _solver_rng(cfg.seed, i))
            results.append(sim.run(solver, _fill_rng(cfg.seed, i), lam))
        k = aggregate(solver_code, solver_code, results, co2_kg_per_l=co2)
        saving = (
            (b0.mean_fuel_l - k.mean_fuel_l) / b0.mean_fuel_l * 100
            if b0.mean_fuel_l else 0.0
        )
        rows.append({
            "hijyen_tavan_gun": cap, "yakit_L": round(k.mean_fuel_l, 2),
            "bolge_ici_km": round(k.mean_intra_km, 2),
            "tasarruf_%": round(saving, 1), "tasma": k.total_overflow_events,
            "en_uzun_bekleme_gun": cap,  # hijyen tavani = en uzun bekleme
        })
    return pd.DataFrame(rows)


def levels_sensitivity(
    cfg: Config, num_seeds: int, time_limit: float, lam: float,
) -> pd.DataFrame:
    """Konut kat dagilimi duyarliligi (2.5/3/4) - B5c ticari-katsayi felsefesi.

    Her senaryo icin dataset YENIDEN kurulur (save=False; ana cache korunur),
    B0 + B2 kosulur. Sonuc kat varsayimina duyarliysa ACIKCA raporlanir.

    FIZIBILITE CAPASI (Bolum 6): yuksek kat senaryosunda talep filo kapasitesini
    asabilir. O durumda senaryo "FILO YETERSIZ" olarak KAYDEDILIR ve atlanir -
    kosulursa cozucu bos rota dondurur ve sonuc "%100 tasarruf" gibi gorunur.
    """
    rows = []
    for lr in cfg.building_model.levels_sensitivity:
        c2 = cfg.model_copy(deep=True)
        object.__setattr__(c2.building_model.levels, "residential", lr)
        ds = build_dataset(c2, force=True, save=False)

        try:
            Simulator(ds, c2).feasibility_check()
        except ValueError as exc:
            rows.append({
                "konut_kat": f"{lr.min}-{lr.max}",
                "nufus": int(ds.residents.sum()),
                "bin": int(ds.n_bins.sum()),
                "durum": "FILO YETERSIZ - senaryo kosulmadi",
                "not": str(exc).replace("\n", " "),
            })
            continue
        co2 = c2.fuel.co2_kg_per_l
        b0 = aggregate(
            "B0", "B0", run_solver(ds, c2, "B0", num_seeds, time_limit, lam),
            co2_kg_per_l=co2,
        )
        b2 = aggregate(
            "B2", "B2", run_solver(ds, c2, "B2", num_seeds, time_limit, lam),
            co2_kg_per_l=co2,
        )
        saving = (
            (b0.mean_fuel_l - b2.mean_fuel_l) / b0.mean_fuel_l * 100
            if b0.mean_fuel_l else 0.0
        )
        rows.append({
            "konut_kat": f"{lr.min}-{lr.max}",
            "nufus": int(ds.residents.sum()),
            "bin": int(ds.n_bins.sum()),
            "durum": "OK",
            "B0_durak": round(b0.mean_stops, 1),
            "B2_durak": round(b2.mean_stops, 1),
            "B0_yakit_L": round(b0.mean_fuel_l, 2),
            "B2_yakit_L": round(b2.mean_fuel_l, 2),
            "B2_dolulukpct": round(b2.mean_fill_pct, 1),
            "tasarruf_pct": round(saving, 1),
            "tasma": b2.total_overflow_events,
        })
    return pd.DataFrame(rows)


def _make_run_dir(cfg: Config) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"{ts}_{cfg.config_hash}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_experiment(
    cfg: Config, *, num_seeds: int, config_path: str = "config.yaml",
    aux_seeds: int | None = None,
) -> Path:
    """Tam deney: lambda taramasi + uzun + odakli kademe + saglik + runs/ cikti.

    num_seeds: ANA (uzun) kademe replikasyonu. aux_seeds: yardimci kademeler
    (lambda taramasi, odakli, duyarlilik) - pahali olduklari icin daha az
    (varsayilan min(3, num_seeds)).
    """
    aux = aux_seeds if aux_seeds is not None else min(3, num_seeds)
    dataset = build_dataset(cfg)
    out = _make_run_dir(cfg)
    shutil.copy(config_path, out / "config.yaml")

    sim = Simulator(dataset, cfg)
    sim.feasibility_check()

    long_limit = float(cfg.solvers.time_limit_long_sec)
    focused_limit = float(cfg.solvers.time_limit_focused_sec)

    # 1) lambda taramasi (uzun-sim limiti, az seed) -> Pareto + lambda*
    points, lam_star = lambda_sweep(dataset, cfg, aux, long_limit)
    _write_pareto(points, lam_star, out)

    # 2) UZUN kademe: tum cozucuier, lambda*, uzun limit -> ana KPI tablosu
    long_kpis: list[SolverKPIs] = []
    for code in SOLVER_ORDER:
        results = run_solver(dataset, cfg, code, num_seeds, long_limit, lam_star)
        long_kpis.append(
            aggregate(code, SOLVER_NAMES[code], results,
                      co2_kg_per_l=cfg.fuel.co2_kg_per_l)
        )
    long_table = _kpi_table(long_kpis)
    long_table.to_csv(out / "kpi_long.csv", index=False)

    # 3) ODAKLI kademe: tam limit, temsili gun sayisi (adil F2)
    focused_cfg = cfg.model_copy(deep=True)
    object.__setattr__(focused_cfg.simulation, "report_days", cfg.solvers.focused_days)
    focused_kpis: list[SolverKPIs] = []
    for code in SOLVER_ORDER:
        results = run_solver(dataset, focused_cfg, code, aux, focused_limit, lam_star)
        focused_kpis.append(
            aggregate(code, SOLVER_NAMES[code], results,
                      co2_kg_per_l=cfg.fuel.co2_kg_per_l)
        )
    _kpi_table(focused_kpis).to_csv(out / "kpi_focused.csv", index=False)

    # 4) Hijyen tavani duyarlilik analizi (cap 3/5/7, az seed)
    sens = sensitivity_analysis(dataset, cfg, aux, long_limit, lam_star)
    sens.to_csv(out / "hygiene_sensitivity.csv", index=False)

    # 5) Konut kat dagilimi duyarliligi (2.5/3/4, az seed)
    lv_sens = levels_sensitivity(cfg, aux, long_limit, lam_star)
    lv_sens.to_csv(out / "levels_sensitivity.csv", index=False)

    # 6) Saglik kontrolleri (uzun kademe uzerinde)
    warnings = health_checks(long_kpis)
    _write_health(
        out, cfg, lam_star, num_seeds, long_kpis, warnings, long_table, sens, lv_sens
    )
    return out


def _write_health(
    out: Path, cfg: Config, lam_star: float, num_seeds: int,
    kpis: list[SolverKPIs], warnings: list[str], table: pd.DataFrame,
    sens: pd.DataFrame, lv_sens: pd.DataFrame,
) -> None:
    lines = [
        "=== DENEY SAGLIK RAPORU (FAZ 2 - yakit/CO2) ===",
        f"config_hash: {cfg.config_hash}   seed: {cfg.seed}   replikasyon: {num_seeds}",
        f"lambda* (tasma=0 en dusuk YAKIT): {lam_star:.4g}",
        f"filo: {cfg.fleet.num_vehicles} arac   ufuk: {cfg.simulation.report_days} gun",
        f"yakit katsayilari (M.8): taban {cfg.fuel.base_l_per_100km} L/100km  "
        f"egim {cfg.fuel.slope_l_per_100km_per_tonne} L/100km/t  "
        f"dur-kalk {cfg.fuel.stop_start_ml} mL  CO2 {cfg.fuel.co2_kg_per_l} kg/L",
        "",
        "--- KPI TABLOSU (uzun kademe) ---",
        table.to_string(index=False),
        "",
        "--- HIJYEN TAVANI DUYARLILIK (cap 3/5/7) ---",
        sens.to_string(index=False),
        "",
        "--- KONUT KAT DAGILIMI DUYARLILIK (2.5/3/4) ---",
        lv_sens.to_string(index=False),
        "",
        "--- SAGLIK KONTROLLERI (Bolum 10 + H1) ---",
    ]
    if warnings:
        lines.extend(f"  !! {w}" for w in warnings)
    else:
        lines.append("  Tumu gecti: kutle korunumu OK, tasma=0, yapisal tavan asilmadi.")
    (out / "health_report.txt").write_text("\n".join(lines), encoding="utf-8")
