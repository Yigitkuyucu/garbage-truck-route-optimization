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

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # basliksiz PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config, load_config
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


# ---------------------------------------------------------------------------
# Paralel kosum
# ---------------------------------------------------------------------------
# PARALELLIK VARSAYILAN OLARAK KAPALIDIR. Sebebi olculdu:
#
#   Ayni is (B2, 2 seed, 44 gun, 30 sn limit), ayni config:
#       seri            45.2 dk  ->  yakit 85.6 / 85.0 L
#       paralel (7)     22.7 dk  ->  yakit 95.9 / 95.5 L      %12 DAHA KOTU
#
# Cozucu butcesi DUVAR SAATIdir. Es zamanli kosan her surec, digerlerinin
# duvar saniyesi basina dusen islem gucunu azaltir; 30 saniyede daha az arama
# yapilir ve cozum kalitesi duser. Zor gunlerde fizibil cozum hic bulunamaz
# (bu, iki deney kosumunun cokme sebebiydi).
#
# Dahasi yanlilik SISTEMATIK DEGIL: pool.map 10 seed'i 7 isciye dagitinca ilk
# parti 7, ikinci parti 3 surecle kosar - son seed'ler daha bos bir makinede
# daha iyi cozum alir. Ortalama +/- standart sapma istatistigi bozulur.
#
# Bu yuzden CLAUDE.md Bolum 8'in "tek cekirdek, paralellik kapali" kurali
# bicimsel bir titizlik degil, tasiyici bir kisittir. Olculen %12'lik fark,
# olcmeye calistigimiz etkiden (B2 vs B0) buyuk.
#
# Kod duruyor cunku cozucuye bagli OLMAYAN isler icin guvenlidir; acmak icin
# ROTA_WORKERS=N. Deney icin ACMAYIN.

_WORKERS_ENV = "ROTA_WORKERS"
_worker_ds: BuiltDataset | None = None
_pool: ProcessPoolExecutor | None = None


def worker_count() -> int:
    """Isci sayisi. VARSAYILAN 1 (seri) - yukaridaki olcume bakiniz.

    ROTA_WORKERS ile acilabilir, ama cozucu kalitesini dusurur; deney
    sonuclari icin kullanilmamalidir.
    """
    return max(1, int(os.environ.get(_WORKERS_ENV, "1")))


def _init_worker(config_path: str) -> None:
    """Her iscide bir kez: cozucu ici paralelligi kapat, veri kumesini yukle."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    global _worker_ds
    _worker_ds = build_dataset(load_config(config_path))   # cache'ten; aga dokunmaz


def _run_seed(task: tuple[Config, str, int, float, float, int | None]) -> RunResult:
    """Tek (cozucu, seed) kosumu - isci surecte calisir.

    Config GOREV ICINDE tasinir, iscinin diskten okudugu kopya KULLANILMAZ.
    Gerekce: cagiran cfg'yi degistirmis olabilir (ufuk, hijyen tavani...); isci
    kendi kopyasini kullansaydi bu degisiklikler SESSIZCE yok sayilirdi.
    Isci yalnizca agir veri kumesini (cache'li) tasir.
    """
    cfg, code, i, time_limit, skip_lambda, hygiene_cap = task
    assert _worker_ds is not None
    sim = Simulator(_worker_ds, cfg)
    if hygiene_cap is not None:
        sim.hygiene_cap_override(hygiene_cap)
    solver = make_solver(code, cfg, time_limit, _solver_rng(cfg.seed, i))
    return sim.run(solver, _fill_rng(cfg.seed, i), skip_lambda)


def get_pool(config_path: str = "config.yaml") -> ProcessPoolExecutor | None:
    """Sureci boyunca tek havuz (veri kumesi isci basina bir kez yuklensin)."""
    global _pool
    n = worker_count()
    if n <= 1:
        return None
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=n, initializer=_init_worker, initargs=(config_path,)
        )
    return _pool


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown()
        _pool = None


def run_solver(
    dataset: BuiltDataset, cfg: Config, code: str, num_seeds: int,
    time_limit: float, skip_lambda: float, *,
    hygiene_cap: int | None = None,
    report_days: int | None = None,
    parallel: bool = True,
) -> list[RunResult]:
    """Bir cozucuyu num_seeds seed'de kostur. Ayni seed'de fill AYNIDIR (D5).

    Seed'ler bagimsizdir; havuz varsa paralel kosulur. Sonuc SIRASI korunur,
    dolayisiyla cikti seri kosumla BIREBIR aynidir (tekrarlanabilirlik).

    parallel=False: veri kumesi ana config'inkinden farkliysa (or. kat
    duyarliligi kendi dataset'ini kurar) isciler yanlis veriyi kullanirdi.
    """
    c = cfg
    if report_days is not None:
        c = cfg.model_copy(deep=True)
        object.__setattr__(c.simulation, "report_days", report_days)

    pool = get_pool() if parallel else None
    if pool is not None:
        tasks = [
            (c, code, i, time_limit, skip_lambda, hygiene_cap)
            for i in range(num_seeds)
        ]
        return list(pool.map(_run_seed, tasks))

    sim = Simulator(dataset, c)
    if hygiene_cap is not None:
        sim.hygiene_cap_override(hygiene_cap)
    out: list[RunResult] = []
    for i in range(num_seeds):
        solver = make_solver(code, c, time_limit, _solver_rng(c.seed, i))
        out.append(sim.run(solver, _fill_rng(c.seed, i), skip_lambda))
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
    # Tarama CALISMA NOKTASI secer, istatistik uretmez: kisa ufuk + az seed
    # yeterlidir. Tam ufukla kosmak deneyin yarisindan fazlasini yiyordu.
    points: list[ParetoPoint] = []
    for lam in lambdas:
        results = run_solver(
            dataset, cfg, sweep_solver, s.sweep_seeds, time_limit, float(lam),
            report_days=s.sweep_days,
        )
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
            "cozucu_hata": k.solver_failures,
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
        results = run_solver(
            dataset, cfg, solver_code, num_seeds, time_limit, lam, hygiene_cap=cap
        )
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
        # parallel=False: bu senaryonun dataset'i ANA config'inkinden farkli;
        # isciler kendi (ana) veri kumesini yukledigi icin paralel kosulamaz.
        b0 = aggregate(
            "B0", "B0",
            run_solver(ds, c2, "B0", num_seeds, time_limit, lam, parallel=False),
            co2_kg_per_l=co2,
        )
        b2 = aggregate(
            "B2", "B2",
            run_solver(ds, c2, "B2", num_seeds, time_limit, lam, parallel=False),
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


_T0 = time.perf_counter()


def _stage(msg: str) -> None:
    """Uzun kosuda ilerleme (saatlerce sessiz kalmasin)."""
    print(f"  [{(time.perf_counter() - _T0) / 60:6.1f} dk] {msg}", flush=True)


def run_experiment(
    cfg: Config, *, num_seeds: int, config_path: str = "config.yaml",
    aux_seeds: int | None = None, sens_seeds: int | None = None,
) -> Path:
    """Tam deney: lambda taramasi + uzun + odakli kademe + saglik + runs/ cikti.

    num_seeds: ANA (uzun) kademe replikasyonu. aux_seeds: yardimci kademeler
    (lambda taramasi, odakli, duyarlilik) - pahali olduklari icin daha az
    (varsayilan min(3, num_seeds)).
    """
    aux = aux_seeds if aux_seeds is not None else min(3, num_seeds)
    # Duyarlilik analizleri YAN bulgudur; manset KPI tablosu (uzun kademe,
    # num_seeds) ve adil karsilastirma (odakli, aux) tam gucte kalir.
    sens = sens_seeds if sens_seeds is not None else min(2, num_seeds)
    dataset = build_dataset(cfg)
    out = _make_run_dir(cfg)
    shutil.copy(config_path, out / "config.yaml")

    sim = Simulator(dataset, cfg)
    sim.feasibility_check()

    long_limit = float(cfg.solvers.time_limit_long_sec)
    focused_limit = float(cfg.solvers.time_limit_focused_sec)

    _stage("1/6 lambda taramasi")
    points, lam_star = lambda_sweep(dataset, cfg, aux, long_limit)
    # (tarama kendi butcesini config'den okur: num / sweep_days / sweep_seeds)
    _write_pareto(points, lam_star, out)

    _stage(f"2/6 uzun kademe (lambda*={lam_star:.3g}, {num_seeds} seed)")
    long_kpis: list[SolverKPIs] = []
    for code in SOLVER_ORDER:
        results = run_solver(dataset, cfg, code, num_seeds, long_limit, lam_star)
        long_kpis.append(
            aggregate(code, SOLVER_NAMES[code], results,
                      co2_kg_per_l=cfg.fuel.co2_kg_per_l)
        )
    long_table = _kpi_table(long_kpis)
    long_table.to_csv(out / "kpi_long.csv", index=False)

    _stage("3/6 odakli kademe (adil tam limit)")
    focused_kpis: list[SolverKPIs] = []
    for code in SOLVER_ORDER:
        results = run_solver(
            dataset, cfg, code, aux, focused_limit, lam_star,
            report_days=cfg.solvers.focused_days,
        )
        focused_kpis.append(
            aggregate(code, SOLVER_NAMES[code], results,
                      co2_kg_per_l=cfg.fuel.co2_kg_per_l)
        )
    _kpi_table(focused_kpis).to_csv(out / "kpi_focused.csv", index=False)

    _stage("4/6 hijyen tavani duyarliligi")
    sens = sensitivity_analysis(dataset, cfg, sens, long_limit, lam_star)
    sens.to_csv(out / "hygiene_sensitivity.csv", index=False)

    _stage("5/6 kat dagilimi duyarliligi (SERI - dataset yeniden kurulur)")
    lv_sens = levels_sensitivity(cfg, sens, long_limit, lam_star)
    lv_sens.to_csv(out / "levels_sensitivity.csv", index=False)

    # 6) Saglik kontrolleri (uzun kademe uzerinde)
    _stage("6/6 saglik kontrolleri")
    shutdown_pool()
    warnings = health_checks(long_kpis)
    _write_health(
        out, cfg, lam_star, num_seeds, long_kpis, warnings, long_table, sens, lv_sens
    )
    return out


def _optimizable_share_line(kpis: list[SolverKPIs], b0_code: str = "B0") -> str:
    """B0 uzerinde olculen yapisal ayrim - ESIK DEGIL, OLCUM (M.11).

    Eski kodda bu buyukluk (%17) yanlislikla bir tasarruf esigi olarak
    kullaniliyordu. Artik oldugu sey olarak raporlanir: tasarruf yuzdelerinin
    hangi payda uzerinde konustugunu soyleyen yapisal bir gercek.
    """
    b0 = next((k for k in kpis if k.code == b0_code), None)
    if b0 is None or b0.mean_total_km <= 0:
        return "yapisal ayrim: olculemedi"
    share = b0.optimizable_share
    return (
        f"yapisal ayrim (B0): optimize-edilebilir pay %{share * 100:.0f} "
        f"({b0.mean_intra_km:.1f} km bolge-ici / {b0.mean_total_km:.1f} km toplam), "
        f"deadhead %{(1 - share) * 100:.0f} - dokunulamaz"
    )


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
        _optimizable_share_line(kpis),
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
