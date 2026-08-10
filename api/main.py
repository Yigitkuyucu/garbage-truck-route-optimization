"""FastAPI uygulamasi - prototip karar destek araci (FAZ 3).

Akis:  doluluk gir  ->  bugunu coz  ->  plani incele  ->  uygula ve kaydet

Mimari kurallar burada da gecerlidir:
  - Kural B: maliyet YALNIZCA Evaluator'da hesaplanir; bu katman bicimler.
  - Zorunlu ziyaret karari `Simulator.solve_day` icindedir - simulasyonla ayni kod.
  - Kalici durum `sim.operations.OperationalState` (diskte JSON).

Prototip siniri (M.9): tek kullanici, bellek ici oturum, kimlik dogrulama yok,
gercek sensor entegrasyonu yok.

Calistir:  uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from api.schemas import (
    ApplyRequest,
    Bootstrap,
    FillSummary,
    Plan,
    SimulateRequest,
    SolveRequest,
    StateSummary,
)
from api.service import (
    build_bootstrap,
    build_fill_summary,
    build_plan,
    build_state_summary,
)
from api.session import SESSION, get_config, get_dataset, get_simulator
from sim.experiment import make_solver
from sim.operations import (
    FillCsvError,
    OperationalState,
    fill_template,
    parse_fill_csv,
    simulate_next_day,
    stop_list,
)
from solvers.base import NoFeasibleSolutionError

_ROOT = Path(__file__).resolve().parent.parent
WEB = _ROOT / "web"

app = FastAPI(title="Atik Toplama Planlama", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _ops() -> OperationalState:
    """Diskteki kalici durumu oku (her istekte taze - tek dogruluk kaynagi disk)."""
    cfg, ds = get_config(), get_dataset()
    return OperationalState.load(cfg.config_hash, ds.num_containers)


FALLBACK_LAMBDA = 0.1


def _default_lambda() -> float:
    """Deneyin calisma noktasi (tasma=0 en dusuk yakit); yoksa varsayilan.

    Not: bu kurulumda lambda sonucu pratik olarak etkilemez;
    yine de deneyle tutarli baslamak icin runs/ ciktisindan okunur.
    """
    runs = sorted(
        (p for p in (_ROOT / "runs").glob("*/lambda_pareto.csv")),
        key=lambda p: p.parent.name,
    )
    if not runs:
        return FALLBACK_LAMBDA
    df = pd.read_csv(runs[-1])
    metric = "yakit_L" if "yakit_L" in df.columns else "bolge_ici_km"
    ok = df[df["tasma"] == 0]
    if ok.empty:
        return FALLBACK_LAMBDA
    return float(round(ok.loc[ok[metric].idxmin(), "lambda"], 4))


# --------------------------------------------------------------------- sayfa


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


# ----------------------------------------------------------------------- api


@app.get("/api/bootstrap", response_model=Bootstrap)
def bootstrap() -> Bootstrap:
    return build_bootstrap(_default_lambda())


@app.get("/api/state", response_model=StateSummary)
def state() -> StateSummary:
    return build_state_summary(_ops())


@app.get("/api/fill", response_model=FillSummary | None)
def current_fill() -> FillSummary | None:
    """Girilmis ama henuz uygulanmamis doluluk (sayfa yenilenince kaybolmasin)."""
    if SESSION.pending_fill is None:
        return None
    return build_fill_summary(SESSION.pending_fill, SESSION.pending_source or "-", [])


@app.get("/api/template.csv")
def template_csv() -> StreamingResponse:
    csv = fill_template(get_dataset(), _ops())
    return StreamingResponse(
        io.StringIO(csv),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="doluluk_{date.today().isoformat()}.csv"'
        },
    )


@app.post("/api/fill/upload", response_model=FillSummary)
async def upload_fill(file: UploadFile = File(...)) -> FillSummary:
    raw = await file.read()
    try:
        fills, warnings = parse_fill_csv(raw, get_dataset())
    except FillCsvError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    SESSION.set_fill(fills, f"CSV: {file.filename}")
    return build_fill_summary(fills, SESSION.pending_source or "-", warnings)


@app.post("/api/fill/simulate", response_model=FillSummary)
def simulate_fill(req: SimulateRequest) -> FillSummary:
    """Sensor verisi yokken gosterim: mevcut dolulugun uzerine BIR gunluk uretim.

    Simulasyonla ayni uretim fonksiyonu - ayri bir 'sahte veri' yolu yoktur.
    """
    cfg, ds = get_config(), get_dataset()
    rng = np.random.default_rng([cfg.seed, req.seed, 99])
    fills = _ops().fill_l + simulate_next_day(ds, cfg, rng)
    SESSION.set_fill(fills, f"model uretimi (seed {req.seed})")
    return build_fill_summary(fills, SESSION.pending_source or "-", [])


@app.post("/api/solve", response_model=Plan)
def solve(req: SolveRequest) -> Plan:
    if SESSION.pending_fill is None:
        raise HTTPException(status_code=409, detail="Once doluluk verisi girin.")

    cfg = get_config()
    try:
        # Deney orkestratoruyle AYNI cozucu fabrikasi (sim.experiment) - panel ve
        # API ayri fabrika tutmaz, yoksa zamanla ayrisirlar.
        solver = make_solver(
            req.solver, cfg, float(req.time_limit_sec),
            np.random.default_rng([cfg.seed, 0, 1]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        day = get_simulator().solve_day(
            SESSION.pending_fill, _ops().days_since, solver, req.skip_lambda
        )
    except NoFeasibleSolutionError as exc:
        SESSION.clear_plan()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    SESSION.plan = day
    SESSION.plan_solver = req.solver
    return build_plan(day, req.solver)


@app.get("/api/plan", response_model=Plan | None)
def current_plan() -> Plan | None:
    if SESSION.plan is None:
        return None
    return build_plan(SESSION.plan, SESSION.plan_solver or "?")


@app.get("/api/stops.csv")
def stops_csv() -> StreamingResponse:
    if SESSION.plan is None:
        raise HTTPException(status_code=409, detail="Once bir plan hesaplayin.")
    df: pd.DataFrame = stop_list(SESSION.plan, get_dataset())
    name = f"rota_{date.today().isoformat()}_{SESSION.plan_solver}.csv"
    return StreamingResponse(
        io.StringIO(df.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.post("/api/apply", response_model=StateSummary)
def apply(req: ApplyRequest) -> StateSummary:
    if SESSION.plan is None:
        raise HTTPException(status_code=409, detail="Uygulanacak plan yok.")
    if not SESSION.plan.result.feasible:
        raise HTTPException(
            status_code=422,
            detail="Plan kisitlari ihlal ediyor; uygulanamaz.",
        )
    ops = _ops()
    ops.apply(SESSION.plan, req.collection_date, SESSION.plan_solver or "?")
    ops.save()
    SESSION.clear_all()
    return build_state_summary(ops)


@app.post("/api/reset", response_model=StateSummary)
def reset() -> StateSummary:
    ops = _ops()
    ops.reset()
    ops.save()
    SESSION.clear_all()
    return build_state_summary(ops)
