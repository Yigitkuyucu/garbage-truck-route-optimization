"""sim/operations.py - operasyonel durum, CSV g/c, surucu ciktisi (FAZ 3, M.9)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from sim import operations as ops
from sim.operations import FillCsvError, OperationalState


class _FakeDataset:
    """BuiltDataset'in operations.py'nin kullandigi minimal yuzeyi."""

    def __init__(self, n: int = 3) -> None:
        self.num_containers = n
        self.container_coords = [(40.0 + i / 100, 26.0 + i / 100) for i in range(n)]
        self.dump_coord = (40.5, 26.5)
        self.volume_l = np.full(n, 1100, dtype=np.int64)


# ------------------------------------------------------------------ durum


def test_fresh_state_is_empty() -> None:
    s = OperationalState.fresh("abc123", 5)
    assert s.fill_l.tolist() == [0.0] * 5
    assert s.days_since.tolist() == [0] * 5
    assert s.last_date is None
    assert s.history == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ops, "STATE_DIR", tmp_path)
    s = OperationalState.fresh("hash1", 4)
    s.fill_l = np.array([10.5, 0.0, 900.25, 3.0])
    s.days_since = np.array([0, 2, 1, 5])
    s.last_date = "2026-08-07"
    s.save()

    back = OperationalState.load("hash1", 4)
    assert np.allclose(back.fill_l, s.fill_l)
    assert back.days_since.tolist() == [0, 2, 1, 5]
    assert back.last_date == "2026-08-07"


def test_load_missing_file_returns_fresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ops, "STATE_DIR", tmp_path)
    s = OperationalState.load("yok", 3)
    assert s.fill_l.tolist() == [0.0, 0.0, 0.0]


def test_load_with_wrong_size_discards_stale_state(tmp_path, monkeypatch) -> None:
    """Config degisince konteyner sayisi degisir - eski durum TASINMAZ."""
    monkeypatch.setattr(ops, "STATE_DIR", tmp_path)
    s = OperationalState.fresh("h", 4)
    s.fill_l = np.array([1.0, 2.0, 3.0, 4.0])
    s.save()
    back = OperationalState.load("h", 7)   # farkli N
    assert back.fill_l.shape == (7,)
    assert back.fill_l.sum() == 0.0


def test_reset_clears_everything() -> None:
    s = OperationalState.fresh("h", 3)
    s.fill_l = np.array([5.0, 5.0, 5.0])
    s.days_since = np.array([1, 2, 3])
    s.last_date = "2026-01-01"
    s.history = [{"tarih": "2026-01-01"}]
    s.reset()
    assert s.fill_l.sum() == 0.0
    assert s.days_since.sum() == 0
    assert s.last_date is None and s.history == []


# ------------------------------------------------------------ apply / kutle


class _FakeResult:
    feasible = True
    n_skipped = 1
    fuel_ml = 12_345
    total_distance = 6_789


class _FakeDayState:
    """DayState'in `apply`'in kullandigi minimal yuzeyi."""

    def __init__(self, fill_before: np.ndarray, visited: np.ndarray) -> None:
        self.fill_before = fill_before
        self.visited = visited
        self.collected_l = int(fill_before[visited].sum())
        self.overflow_events = 0
        self.result = _FakeResult()


def test_apply_preserves_skipped_container_fill() -> None:
    """ATLANAN konteynerin copu KAYBOLMAZ (kutle korunumu, Bolum 10 #1).

    Regresyon: `apply` bir zamanlar durumun bayat `fill_l`'ini kullaniyordu;
    gunluk girdi (CSV/model) hic yazilmadigi icin atlanan konteynerler de
    bosalmis gorunuyordu.
    """
    ops_state = OperationalState.fresh("h", 3)          # fill_l = [0, 0, 0]
    todays_fill = np.array([500.0, 900.0, 300.0])       # CSV'den gelen doluluk
    visited = np.array([True, False, True])             # 1 numarali ATLANDI

    ops_state.apply(_FakeDayState(todays_fill, visited), date(2026, 8, 7), "B2")

    assert ops_state.fill_l.tolist() == [0.0, 900.0, 0.0], (
        "atlanan konteynerin dolulugu korunmali"
    )
    assert ops_state.days_since.tolist() == [0, 1, 0]


def test_apply_records_history() -> None:
    s = OperationalState.fresh("h", 2)
    s.apply(
        _FakeDayState(np.array([100.0, 200.0]), np.array([True, True])),
        date(2026, 8, 7), "X1",
    )
    assert s.last_date == "2026-08-07"
    assert len(s.history) == 1
    assert s.history[0]["cozucu"] == "X1"
    assert s.history[0]["toplanan_l"] == 300


def test_apply_twice_accumulates_days_since() -> None:
    """Ust uste iki gun: hic ziyaret edilmeyen konteynerin sayaci artar."""
    s = OperationalState.fresh("h", 2)
    for _ in range(2):
        s.apply(
            _FakeDayState(np.array([10.0, 10.0]), np.array([True, False])),
            date(2026, 8, 7), "B2",
        )
    assert s.days_since.tolist() == [0, 2]


# -------------------------------------------------------------------- CSV


def _csv(rows: str, header: str = "konteyner_id,doluluk_l") -> str:
    return f"{header}\n{rows}"


def test_parse_valid_csv() -> None:
    fills, warns = ops.parse_fill_csv(_csv("0,100\n1,200\n2,300"), _FakeDataset())
    assert fills.tolist() == [100.0, 200.0, 300.0]
    assert warns == []


def test_parse_respects_id_order() -> None:
    """Satirlar karisik siradaysa bile id'ye gore yerlesir."""
    fills, _ = ops.parse_fill_csv(_csv("2,300\n0,100\n1,200"), _FakeDataset())
    assert fills.tolist() == [100.0, 200.0, 300.0]


def test_parse_extra_reference_columns_ignored() -> None:
    text = "konteyner_id,lat,lon,hacim_l,doluluk_l\n0,40.0,26.0,1100,50\n" \
           "1,40.1,26.1,1100,60\n2,40.2,26.2,1100,70"
    fills, _ = ops.parse_fill_csv(text, _FakeDataset())
    assert fills.tolist() == [50.0, 60.0, 70.0]


def test_parse_missing_column() -> None:
    with pytest.raises(FillCsvError, match="Eksik sutun"):
        ops.parse_fill_csv("konteyner_id,dolu\n0,1", _FakeDataset())


def test_parse_negative_fill() -> None:
    with pytest.raises(FillCsvError, match="negatif olamaz"):
        ops.parse_fill_csv(_csv("0,100\n1,-5\n2,300"), _FakeDataset())


def test_parse_missing_container_ids() -> None:
    with pytest.raises(FillCsvError, match="tamamini icermeli"):
        ops.parse_fill_csv(_csv("0,100\n1,200"), _FakeDataset())


def test_parse_unknown_container_id() -> None:
    with pytest.raises(FillCsvError, match="tamamini icermeli"):
        ops.parse_fill_csv(_csv("0,1\n1,2\n2,3\n99,4"), _FakeDataset())


def test_parse_empty_cell() -> None:
    with pytest.raises(FillCsvError, match="bos hucre"):
        ops.parse_fill_csv(_csv("0,100\n1,\n2,300"), _FakeDataset())


def test_parse_overflow_warns_but_accepts() -> None:
    """Hacmi asan doluluk gercek hayatta olur - engellenmez, UYARILIR."""
    fills, warns = ops.parse_fill_csv(_csv("0,100\n1,5000\n2,300"), _FakeDataset())
    assert fills[1] == 5000.0
    assert any("tasmis" in w for w in warns)


def test_template_roundtrips_through_parser() -> None:
    """Uretilen sablon, ayristiricidan gecmeli (sozlesme kendi kendine tutarli)."""
    ds = _FakeDataset()
    st = OperationalState.fresh("h", ds.num_containers)
    st.fill_l = np.array([12.0, 0.0, 640.0])
    fills, warns = ops.parse_fill_csv(ops.fill_template(ds, st), ds)
    assert fills.tolist() == [12.0, 0.0, 640.0]
    assert warns == []
