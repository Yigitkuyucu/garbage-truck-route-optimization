"""Insa edilen veri kumesi + cache (KABUK).

BuiltDataset, cekirdegin (Adim 2) ihtiyac duydugu tum statik veriyi tutar:
mesafe/sure matrisleri, konteyner talep hizlari, must_visit icin gerekli
meta. Cekirdek bunu GORMEZ; sinir gecisinde (build_problem) diziye cevrilir.

Cache config_hash'e gore anahtarlanir: config degisirse yeniden insa edilir
(saniyeler, aga dokunmaz).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache")

Coord = tuple[float, float]


@dataclass(frozen=True)
class BuiltDataset:
    """Dugum sirasi: 0=garaj, 1..N=konteyner, N+1=dokum. K = N+2."""

    config_hash: str
    region_key: str
    container_source: str

    depot_coord: Coord
    dump_coord: Coord
    container_coords: list[Coord]     # N (lat, lon)

    base_rate_l: np.ndarray           # (N,) float64 - gunluk temel doluluk hizi (L)
    volume_l: np.ndarray              # (N,) int64 - nokta kapasitesi (n_bin * bin_hacmi)
    n_bins: np.ndarray                # (N,) int64 - noktadaki bin sayisi
    residents: np.ndarray             # (N,) float64 - nufus capasi icin
    commercial_type: list[str]        # (N,) baskin tur
    market_day: list[str | None]      # (N,) pazar gunu ya da None

    dist_m: np.ndarray                # (K, K) int64, metre
    time_s: np.ndarray                # (K, K) int64, saniye
    node_ids: np.ndarray              # (K,) int64 - eslenen yol agi dugumleri

    @property
    def num_containers(self) -> int:
        return len(self.container_coords)

    @property
    def depot_index(self) -> int:
        return 0

    @property
    def dump_index(self) -> int:
        return self.num_containers + 1

    def container_index(self, i: int) -> int:
        """Konteyner i (0-based) -> matris indeksi."""
        return i + 1

    # -------- cache --------

    def _paths(self) -> tuple[Path, Path, Path]:
        h = self.config_hash
        return (
            CACHE_DIR / f"{h}_matrix.npz",
            CACHE_DIR / f"{h}_containers.parquet",
            CACHE_DIR / f"{h}_meta.json",
        )

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        npz, pq, meta = self._paths()
        np.savez_compressed(
            npz, dist_m=self.dist_m, time_s=self.time_s, node_ids=self.node_ids
        )
        lats = [c[0] for c in self.container_coords]
        lons = [c[1] for c in self.container_coords]
        pd.DataFrame(
            {
                "lat": lats,
                "lon": lons,
                "base_rate_l": self.base_rate_l,
                "volume_l": self.volume_l,
                "n_bins": self.n_bins,
                "residents": self.residents,
                "commercial_type": self.commercial_type,
                "market_day": pd.array(self.market_day, dtype="string"),
            }
        ).to_parquet(pq)
        meta.write_text(
            json.dumps(
                {
                    "config_hash": self.config_hash,
                    "region_key": self.region_key,
                    "container_source": self.container_source,
                    "depot_coord": list(self.depot_coord),
                    "dump_coord": list(self.dump_coord),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def cache_exists(cls, config_hash: str) -> bool:
        return all(
            p.exists()
            for p in (
                CACHE_DIR / f"{config_hash}_matrix.npz",
                CACHE_DIR / f"{config_hash}_containers.parquet",
                CACHE_DIR / f"{config_hash}_meta.json",
            )
        )

    @classmethod
    def load(cls, config_hash: str) -> BuiltDataset:
        npz = np.load(CACHE_DIR / f"{config_hash}_matrix.npz")
        df = pd.read_parquet(CACHE_DIR / f"{config_hash}_containers.parquet")
        meta = json.loads((CACHE_DIR / f"{config_hash}_meta.json").read_text(encoding="utf-8"))
        mday = [None if pd.isna(v) else str(v) for v in df["market_day"]]
        return cls(
            config_hash=meta["config_hash"],
            region_key=meta["region_key"],
            container_source=meta["container_source"],
            depot_coord=tuple(meta["depot_coord"]),
            dump_coord=tuple(meta["dump_coord"]),
            container_coords=list(zip(df["lat"].tolist(), df["lon"].tolist(), strict=True)),
            base_rate_l=df["base_rate_l"].to_numpy(dtype=np.float64),
            volume_l=df["volume_l"].to_numpy(dtype=np.int64),
            n_bins=df["n_bins"].to_numpy(dtype=np.int64),
            residents=df["residents"].to_numpy(dtype=np.float64),
            commercial_type=df["commercial_type"].tolist(),
            market_day=mday,
            dist_m=npz["dist_m"].astype(np.int64),
            time_s=npz["time_s"].astype(np.int64),
            node_ids=npz["node_ids"].astype(np.int64),
        )
