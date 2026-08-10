"""Operasyonel durum + gunluk girdi/cikti (FAZ 3).

Prototip karar destek aracinin KABUK katmani. Simulasyon 90 gunu tek seferde
kosar; operasyonel kullanim ise **gun gun** ilerler ve iki sey hatirlamak zorundadir:

    fill_l      -> her konteynerin su anki dolulugu (litre)
    days_since  -> son ziyaretten bu yana gecen gun (hijyen tavani icin)

Bu durum diske yazilir; ertesi gun kaldigi yerden devam eder. Anahtar
`config_hash`tir: config degisirse konteyner kumesi de degisir, eski durum
tasinamaz.

> **Prototip siniri (M.9):** dosya tabanli durum, tek kullanici, kimlik dogrulama
> yok, gercek sensor entegrasyonu yok. Doluluk CSV ile girilir.

Bu modul MALIYET HESAPLAMAZ (Kural B) - rota ve yakit Evaluator'dan gelir.
Burada yalnizca durum tasima, CSV ayristirma ve surucu ciktisi vardir.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from config import Config
from data.dataset import BuiltDataset
from sim.containers import generate_daily_fills
from sim.engine import DayState

STATE_DIR = Path("ops_state")

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]

# CSV sutun adlari - kullaniciya gorunen sozlesme
COL_ID = "konteyner_id"
COL_FILL = "doluluk_l"
REQUIRED_COLS = (COL_ID, COL_FILL)


class FillCsvError(ValueError):
    """Yuklenen doluluk CSV'si kullanilamaz. Mesaj kullaniciya gosterilir."""


@dataclass
class OperationalState:
    """Gunler arasi tasinan operasyonel durum."""

    config_hash: str
    n_containers: int
    fill_l: FloatArr
    days_since: IntArr
    last_date: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def fresh(cls, config_hash: str, n: int) -> OperationalState:
        """Bos baslangic: tum konteynerler bos, hepsi bugun toplanmis sayilir."""
        return cls(
            config_hash=config_hash,
            n_containers=n,
            fill_l=np.zeros(n, dtype=np.float64),
            days_since=np.zeros(n, dtype=np.int64),
        )

    # ---------------------------------------------------------------- kaliciik

    @staticmethod
    def path_for(config_hash: str) -> Path:
        return STATE_DIR / f"{config_hash}_ops.json"

    @classmethod
    def load(cls, config_hash: str, n: int) -> OperationalState:
        """Diskten oku; yoksa ya da uyumsuzsa temiz durum dondur."""
        path = cls.path_for(config_hash)
        if not path.exists():
            return cls.fresh(config_hash, n)
        raw = json.loads(path.read_text(encoding="utf-8"))
        fill = np.asarray(raw["fill_l"], dtype=np.float64)
        if fill.shape[0] != n:  # config degismis - eski durum gecersiz
            return cls.fresh(config_hash, n)
        return cls(
            config_hash=config_hash,
            n_containers=n,
            fill_l=fill,
            days_since=np.asarray(raw["days_since"], dtype=np.int64),
            last_date=raw.get("last_date"),
            history=raw.get("history", []),
        )

    def save(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = self.path_for(self.config_hash)
        path.write_text(
            json.dumps(
                {
                    "config_hash": self.config_hash,
                    "last_date": self.last_date,
                    "fill_l": [round(v, 3) for v in self.fill_l.tolist()],
                    "days_since": self.days_since.tolist(),
                    "history": self.history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def reset(self) -> None:
        self.fill_l = np.zeros(self.n_containers, dtype=np.float64)
        self.days_since = np.zeros(self.n_containers, dtype=np.int64)
        self.last_date = None
        self.history = []

    # ---------------------------------------------------------------- ilerletme

    def apply(self, state: DayState, when: date, solver_code: str) -> None:
        """Toplamayi uygula ve gunu gecmise yaz. Durum ILERLER.

        Doluluk, **cozumun dayandigi** anlik goruntudan (`state.fill_before`)
        alinir - durumun kendi `fill_l`'inden DEGIL. Aksi halde CSV'den ya da
        modelden gelen gunluk girdi hic yazilmamis olur ve ATLANAN konteynerlerin
        birikmis copu kaybolur (kutle korunumu ihlali, Bolum 10 kontrol #1).
        """
        from sim.engine import Simulator

        self.fill_l = np.asarray(state.fill_before, dtype=np.float64).copy()
        Simulator.apply_collection(self.fill_l, self.days_since, state.visited)
        self.last_date = when.isoformat()
        self.history.append(
            {
                "tarih": when.isoformat(),
                "cozucu": solver_code,
                "durak": int(state.visited.sum()),
                "atlanan": int(state.result.n_skipped),
                "toplanan_l": int(state.collected_l),
                "yakit_l": round(state.result.fuel_ml / 1000, 2),
                "mesafe_km": round(state.result.total_distance / 1000, 2),
                "tasma_olayi": int(state.overflow_events),
            }
        )


# --------------------------------------------------------------------- CSV g/c


def fill_template(dataset: BuiltDataset, state: OperationalState | None = None) -> str:
    """Doldurulacak doluluk sablonu (CSV metni).

    lat/lon ve hacim REFERANS sutunlaridir (okunmaz, kullaniciya yol gosterir);
    okunan yalnizca konteyner_id ve doluluk_l'dir.
    """
    fill = state.fill_l if state is not None else np.zeros(dataset.num_containers)
    df = pd.DataFrame(
        {
            COL_ID: np.arange(dataset.num_containers),
            "lat": [c[0] for c in dataset.container_coords],
            "lon": [c[1] for c in dataset.container_coords],
            "hacim_l": dataset.volume_l,
            COL_FILL: np.rint(fill).astype(np.int64),
        }
    )
    return df.to_csv(index=False)


def parse_fill_csv(text: str | bytes, dataset: BuiltDataset) -> tuple[FloatArr, list[str]]:
    """Yuklenen CSV -> (doluluk dizisi, uyarilar). Hatada FillCsvError.

    Kati olan: sutunlar, id kumesi, negatif deger. Uyari (engel degil): hacmi
    asan doluluk - gercek hayatta olur (tasmis konteyner) ve model bunu tasir.
    """
    buf = io.BytesIO(text) if isinstance(text, bytes) else io.StringIO(text)
    try:
        df = pd.read_csv(buf)
    except Exception as exc:
        raise FillCsvError(f"CSV okunamadi: {exc}") from exc

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise FillCsvError(
            f"Eksik sutun: {', '.join(missing)}. "
            f"Gerekli sutunlar: {', '.join(REQUIRED_COLS)}. Sablonu indirip doldurun."
        )

    n = dataset.num_containers
    try:
        ids = df[COL_ID].to_numpy(dtype=np.int64)
        fills = df[COL_FILL].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise FillCsvError(f"Sayisal olmayan deger var: {exc}") from exc

    if np.isnan(fills).any():
        raise FillCsvError(f"{COL_FILL} sutununda bos hucre var.")
    if (fills < 0).any():
        raise FillCsvError(f"{COL_FILL} negatif olamaz.")

    expected = set(range(n))
    got = set(ids.tolist())
    if got != expected:
        eksik, fazla = sorted(expected - got)[:5], sorted(got - expected)[:5]
        parts = [f"CSV {n} konteynerin tamamini icermeli (0..{n - 1})."]
        if eksik:
            parts.append(f"Eksik id ornegi: {eksik}")
        if fazla:
            parts.append(f"Taninmayan id ornegi: {fazla}")
        raise FillCsvError(" ".join(parts))
    if len(ids) != n:
        raise FillCsvError(f"Tekrarlanan konteyner_id var ({len(ids)} satir, {n} bekleniyor).")

    out = np.zeros(n, dtype=np.float64)
    out[ids] = fills

    warnings: list[str] = []
    over = int((out > dataset.volume_l).sum())
    if over:
        warnings.append(
            f"{over} konteynerde doluluk hacmi asiyor (tasmis). Model bunu tasir "
            f"ve ziyaret edildiginde fazlasiyla toplar."
        )
    return out, warnings


def simulate_next_day(
    dataset: BuiltDataset, cfg: Config, rng: np.random.Generator
) -> FloatArr:
    """Sensor verisi yoksa: modelden BIR gunluk uretim (demo/gosterim icin).

    Simulasyonun kullandigi ayni uretim fonksiyonu - ayri bir 'sahte veri'
    yolu yoktur.
    """
    return generate_daily_fills(
        dataset.base_rate_l, dataset.market_day, cfg, rng, 1
    )[0]


# ------------------------------------------------------------- surucu ciktisi


def stop_list(state: DayState, dataset: BuiltDataset) -> pd.DataFrame:
    """Arac basina SIRALI durak listesi - surucuye verilecek cikti.

    Maliyet hesabi YOKTUR (Kural B): mesafe dogrudan matristen okunur, yakit
    ve fizibilite Evaluator'dan gelir.
    """
    dist = state.problem.dist
    depot, dump = state.problem.depot_index, state.problem.dump_index
    rows: list[dict[str, Any]] = []

    for v, route in enumerate(state.solution.routes):
        if not route:
            continue
        prev = depot
        cum_l = 0.0
        cum_m = 0
        for order, c in enumerate(route, start=1):
            node = c + 1
            leg = int(dist[prev, node])
            cum_m += leg
            cum_l += float(state.fill_before[c])
            lat, lon = dataset.container_coords[c]
            rows.append({
                "arac": v + 1,
                "sira": order,
                "konteyner_id": int(c),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "doluluk_l": round(float(state.fill_before[c])),
                "doluluk_%": round(
                    state.fill_before[c] / state.volume[c] * 100, 1
                ),
                "zorunlu": bool(state.must_visit[c]),
                "bacak_m": leg,
                "kumulatif_m": cum_m,
                "kamyon_yuku_l": round(cum_l),
            })
            prev = node
        # Dokum bacagi - rotanin sonu
        leg = int(dist[prev, dump])
        cum_m += leg
        rows.append({
            "arac": v + 1,
            "sira": len(route) + 1,
            "konteyner_id": -1,
            "lat": round(dataset.dump_coord[0], 6),
            "lon": round(dataset.dump_coord[1], 6),
            "doluluk_l": 0,
            "doluluk_%": 0.0,
            "zorunlu": True,
            "bacak_m": leg,
            "kumulatif_m": cum_m,
            "kamyon_yuku_l": round(cum_l),
        })
    return pd.DataFrame(rows)
