"""API ucu testleri (FAZ 3).

Gercek dataset uzerinde kosar (cache'li, aga dokunmaz). Cozucu limitleri kisa
tutulur; amac rota kalitesi degil, **sozlesme ve akis** dogrulugudur.

Kalici durum izole edilir: her test tmp bir STATE_DIR kullanir, kullanicinin
gercek ops_state/ dosyasina DOKUNULMAZ.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.session import SESSION, get_dataset
from sim import operations as ops


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Her test: temiz disk durumu + temiz bellek oturumu."""
    monkeypatch.setattr(ops, "STATE_DIR", tmp_path)
    SESSION.clear_all()
    yield
    SESSION.clear_all()


@pytest.fixture
def client():
    return TestClient(app)


def _seed_fill(client, seed: int = 0):
    return client.post("/api/fill/simulate", json={"seed": seed})


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_describes_the_problem(client) -> None:
    b = client.get("/api/bootstrap").json()
    ds = get_dataset()
    assert b["n_containers"] == ds.num_containers
    assert len(b["containers"]) == ds.num_containers
    assert len(b["depot"]) == 2 and len(b["dump"]) == 2
    assert {s["code"] for s in b["solvers"]} == {"B0", "B1", "B2", "X1", "X2"}
    assert b["co2_kg_per_l"] > 0


def test_state_starts_empty(client) -> None:
    s = client.get("/api/state").json()
    assert s["last_date"] is None
    assert s["total_fill_l"] == 0
    assert s["history"] == []


def test_template_csv_is_downloadable_and_parsable(client) -> None:
    r = client.get("/api/template.csv")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    fills, warnings = ops.parse_fill_csv(r.text, get_dataset())
    assert fills.shape[0] == get_dataset().num_containers
    assert warnings == []


# ------------------------------------------------------------------- doluluk


def test_simulate_sets_pending_fill(client) -> None:
    f = _seed_fill(client).json()
    assert f["total_l"] > 0
    assert "model uretimi" in f["source"]
    assert client.get("/api/fill").json()["total_l"] == f["total_l"]


def test_upload_rejects_bad_csv(client) -> None:
    r = client.post(
        "/api/fill/upload",
        files={"file": ("kotu.csv", b"konteyner_id,yanlis\n0,1\n", "text/csv")},
    )
    assert r.status_code == 422
    assert "Eksik sutun" in r.json()["detail"]


def test_upload_roundtrip_through_template(client) -> None:
    csv = client.get("/api/template.csv").text
    r = client.post("/api/fill/upload",
                    files={"file": ("dolu.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    assert r.json()["source"].startswith("CSV:")


# --------------------------------------------------------------------- cozum


def test_solve_requires_fill_first(client) -> None:
    r = client.post("/api/solve", json={"solver": "B2", "time_limit_sec": 1})
    assert r.status_code == 409


def test_solve_returns_consistent_plan(client) -> None:
    _seed_fill(client)
    p = client.post(
        "/api/solve",
        json={"solver": "B1", "skip_lambda": 0.1, "time_limit_sec": 1},
    ).json()

    assert p["feasible"] is True
    k = p["kpi"]
    assert k["fuel_l"] > 0
    # Kalemler toplami yakita esit (yuvarlama toleransiyla)
    parts = k["fuel_travel_l"] + k["fuel_stop_l"] + k["fuel_compaction_l"]
    assert abs(parts - k["fuel_l"]) < 0.05
    # CO2 yakittan turetilir
    co2_factor = client.get("/api/bootstrap").json()["co2_kg_per_l"]
    assert abs(k["co2_kg"] - k["fuel_l"] * co2_factor) < 0.2
    # Ziyaret + atlanan = tum konteynerler
    assert len(p["visited_ids"]) + len(p["skipped_ids"]) == get_dataset().num_containers
    assert len(p["visited_ids"]) == k["stops"]
    # must_visit'lerin hepsi ziyaret edilmis olmali (sert kisit)
    assert set(p["must_visit_ids"]) <= set(p["visited_ids"])


def test_solve_stop_list_ends_each_route_at_dump(client) -> None:
    _seed_fill(client)
    p = client.post("/api/solve",
                    json={"solver": "B1", "time_limit_sec": 1}).json()
    for veh in {s["vehicle"] for s in p["stops"]}:
        rows = [s for s in p["stops"] if s["vehicle"] == veh]
        assert rows[-1]["container_id"] == -1, "rota dokumle bitmeli"
        assert [r["order"] for r in rows] == list(range(1, len(rows) + 1))


def test_unknown_solver_is_rejected(client) -> None:
    _seed_fill(client)
    r = client.post("/api/solve", json={"solver": "ZZ", "time_limit_sec": 1})
    assert r.status_code == 400


def test_new_fill_invalidates_previous_plan(client) -> None:
    _seed_fill(client)
    client.post("/api/solve", json={"solver": "B1", "time_limit_sec": 1})
    assert client.get("/api/plan").json() is not None
    _seed_fill(client, seed=1)
    assert client.get("/api/plan").json() is None


# -------------------------------------------------------------------- uygula


def test_apply_conserves_mass(client) -> None:
    """Atlanan konteynerlerin copu KAYBOLMAZ (Bolum 10 kontrol #1)."""
    fill = _seed_fill(client).json()["total_l"]
    p = client.post("/api/solve",
                    json={"solver": "B1", "time_limit_sec": 1}).json()

    r = client.post("/api/apply", json={"collection_date": "2026-08-07"})
    assert r.status_code == 200
    st = r.json()

    assert st["last_date"] == "2026-08-07"
    # toplanan + kalan == girilen  (yuvarlama toleransi)
    assert abs(p["kpi"]["collected_l"] + st["total_fill_l"] - fill) <= 2
    assert st["history"][0]["durak"] == p["kpi"]["stops"]


def test_apply_requires_a_plan(client) -> None:
    assert client.post(
        "/api/apply", json={"collection_date": "2026-08-07"}).status_code == 409


def test_apply_clears_session(client) -> None:
    _seed_fill(client)
    client.post("/api/solve", json={"solver": "B1", "time_limit_sec": 1})
    client.post("/api/apply", json={"collection_date": "2026-08-07"})
    assert client.get("/api/plan").json() is None
    assert client.get("/api/fill").json() is None


def test_days_since_advances_for_skipped(client) -> None:
    """Atlanan konteynerin bekleme sayaci artar - hijyen tavani buna dayanir."""
    _seed_fill(client)
    p = client.post("/api/solve",
                    json={"solver": "B1", "time_limit_sec": 1}).json()
    client.post("/api/apply", json={"collection_date": "2026-08-07"})
    st = client.get("/api/state").json()
    assert st["max_days_waiting"] == (1 if p["kpi"]["skipped"] else 0)


def test_reset_clears_everything(client) -> None:
    _seed_fill(client)
    client.post("/api/solve", json={"solver": "B1", "time_limit_sec": 1})
    client.post("/api/apply", json={"collection_date": "2026-08-07"})
    st = client.post("/api/reset").json()
    assert st["total_fill_l"] == 0
    assert st["history"] == []
    assert st["last_date"] is None


def test_stops_csv_needs_a_plan(client) -> None:
    assert client.get("/api/stops.csv").status_code == 409
    _seed_fill(client)
    client.post("/api/solve", json={"solver": "B1", "time_limit_sec": 1})
    r = client.get("/api/stops.csv")
    assert r.status_code == 200
    assert "konteyner_id" in r.text.splitlines()[0]


def test_index_page_serves(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Atık Toplama Planlama" in r.text
    assert "prototip" in r.text


def test_plan_matches_simulator_directly(client) -> None:
    """API'nin dondurdugu yakit, Simulator+Evaluator'in dogrudan verdigiyle AYNI.

    API bir bicimleme katmanidir; maliyet hesaplamaz (Kural B). Bu test o
    sozlesmeyi kilitler.
    """
    from api.session import get_simulator
    from sim.experiment import make_solver
    from sim.operations import OperationalState

    fill_resp = _seed_fill(client).json()
    p = client.post("/api/solve",
                    json={"solver": "B1", "skip_lambda": 0.1,
                          "time_limit_sec": 1}).json()

    cfg = get_simulator()._cfg
    ds = get_dataset()
    ops_state = OperationalState.load(cfg.config_hash, ds.num_containers)
    rng = np.random.default_rng([cfg.seed, 0, 99])
    fill = ops_state.fill_l + ops.simulate_next_day(ds, cfg, rng)
    assert abs(int(fill.sum()) - fill_resp["total_l"]) <= 1

    solver = make_solver("B1", cfg, 1.0, np.random.default_rng([cfg.seed, 0, 1]))
    direct = get_simulator().solve_day(fill, ops_state.days_since, solver, 0.1)
    assert round(direct.result.fuel_ml / 1000, 2) == p["kpi"]["fuel_l"]
    assert int(direct.visited.sum()) == p["kpi"]["stops"]
