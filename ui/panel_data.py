"""Panel veri erisim katmani (KABUK) - runs/ CSV'leri + cache'li dataset/replay.

Streamlit paneli buradan beslenir. Iki kaynak:
  1. runs/<ts>_<hash>/  -> deneyin URETTIGI CSV'ler (KPI, Pareto, duyarlilik,
     saglik). Bunlar STATIKtir; panel yeniden hesaplamaz, okur.
  2. Cache'li BuiltDataset + canli replay() -> HARITA icin o gunku rota/atlama.

Cekirdek (domain/) buraya girmez; sinir gecisi Simulator icinde (gunde bir kez).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import Config, load_config
from data.build import build_dataset
from data.dataset import BuiltDataset
from sim.engine import DayState, Simulator
from solvers.abc_solver import ABCSolver
from solvers.base import Solver
from solvers.greedy import GreedySolver
from solvers.ortools_solver import ORToolsSolver
from solvers.threshold_greedy import ThresholdGreedySolver

RUNS_DIR = Path("runs")

SOLVER_ORDER = ["B0", "B1", "B2", "X1", "X2"]
SOLVER_NAMES = {
    "B0": "Sabit rota (B0)",
    "B1": "Esik+greedy (B1)",
    "B2": "OR-Tools (B2)",
    "X1": "abc-basic (X1)",
    "X2": "abc-ls (X2)",
}
# Haritada saga secilebilecek "akilli" cozucuier (B0 her zaman referans, solda).
SMART_SOLVERS = ["B1", "B2", "X1", "X2"]


# ----------------------------------------------------------------------------
# runs/ kesif + CSV okuma (STATIK deney ciktilari)
# ----------------------------------------------------------------------------


def list_runs() -> list[Path]:
    """runs/ altindaki tum kosu klasorleri, en YENI once."""
    if not RUNS_DIR.exists():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def latest_run() -> Path | None:
    runs = list_runs()
    return runs[0] if runs else None


def _read_csv(run: Path, name: str) -> pd.DataFrame | None:
    path = run / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_kpi(run: Path, stage: str = "long") -> pd.DataFrame | None:
    """KPI tablosu: stage 'long' (tum ufuk trendi) ya da 'focused' (adil F2)."""
    return _read_csv(run, f"kpi_{stage}.csv")


def load_pareto(run: Path) -> pd.DataFrame | None:
    return _read_csv(run, "lambda_pareto.csv")


def load_hygiene_sensitivity(run: Path) -> pd.DataFrame | None:
    return _read_csv(run, "hygiene_sensitivity.csv")


def load_levels_sensitivity(run: Path) -> pd.DataFrame | None:
    return _read_csv(run, "levels_sensitivity.csv")


def load_health(run: Path) -> str:
    path = run / "health_report.txt"
    return path.read_text(encoding="utf-8") if path.exists() else "(saglik raporu yok)"


def lambda_star(run: Path) -> float:
    """Pareto'dan calisma noktasi: tasma=0 olan en dusuk YAKIT.

    Manset metrik Faz 2'de yakit oldu; eski kosularda (yakit_L kolonu yok)
    bolge-ici mesafeye duser.
    """
    df = load_pareto(run)
    if df is None or df.empty:
        return 0.1
    metric = "yakit_L" if "yakit_L" in df.columns else "bolge_ici_km"
    feasible = df[df["tasma"] == 0]
    if not feasible.empty:
        return float(feasible.loc[feasible[metric].idxmin(), "lambda"])
    return float(df["lambda"].max())


# ----------------------------------------------------------------------------
# Cache'li dataset + config (HARITA icin)
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    return load_config("config.yaml")


@st.cache_resource(show_spinner="Veri kumesi yukleniyor (cache)...")
def get_dataset() -> BuiltDataset:
    """Cache'li BuiltDataset (aga dokunmaz)."""
    return build_dataset(get_config())


def _make_solver(
    code: str, cfg: Config, time_limit: float, seed: int
) -> Solver:
    """Deney orkestratoru (sim.experiment) ile ayni fabrika mantigii."""
    nv = cfg.fleet.num_vehicles
    if code == "B0":
        return GreedySolver(nv)
    if code == "B1":
        return ThresholdGreedySolver(nv, cfg.solvers.b1_threshold)
    if code == "B2":
        return ORToolsSolver(nv, int(time_limit))
    rng = np.random.default_rng([cfg.seed, seed, 1])
    if code == "X1":
        return ABCSolver(nv, cfg.abc.colony_size, cfg.abc.limit, time_limit, rng)
    if code == "X2":
        return ABCSolver(
            nv, cfg.abc.colony_size, cfg.abc.limit, time_limit, rng,
            use_local_search=True, local_search_iters=cfg.abc.local_search_iters,
            local_search_window=cfg.abc.local_search_window,
        )
    raise ValueError(f"bilinmeyen cozucu: {code}")


@st.cache_data(show_spinner="Rota hesaplaniyor (canli solve)...")
def replay_solver(
    code: str, seed: int, skip_lambda: float, report_days: int, time_limit: float
) -> list[DayState]:
    """Bir cozucuyu (config seed + verilen fill-seed) kostur, gun-basi DayState
    dondur. Cache anahtari argumanlar; gun kaydiricisi degisince YENIDEN
    hesaplama YOK (ayni liste).

    report_days: HARITA ufku (deneyin 90'i degil - panel yanit suresi icin kisa).
    time_limit:  B2/ABC icin solve limiti (harita gorseli; adil F2 degil).
    """
    cfg = get_config()
    ds = get_dataset()
    c = cfg.model_copy(deep=True)
    object.__setattr__(c.simulation, "report_days", report_days)
    sim = Simulator(ds, c)
    solver = _make_solver(code, c, time_limit, seed)
    fill_rng = np.random.default_rng([cfg.seed, seed, 0])
    return sim.replay(solver, fill_rng, skip_lambda)
