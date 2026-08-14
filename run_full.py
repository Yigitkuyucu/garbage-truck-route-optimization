"""Tam deney kosumu: config.yaml'daki butceyle, sonuclar runs/ altina.

Butce ARTIK config.yaml'dadir; burada override YOKTUR. Boylece koda gomulu
ikinci bir gercek olusmaz (kosulan sey ile config'te yazan sey ayni).

Paralellik: bagimsiz seed'ler ayri SURECLERDE kosar. Adil butce protokolu
korunur - kural cozucunun kendi icinde cok is parcacigi kullanmamasidir, ki
iscilerde kapatiliyor. Isci sayisi:  ROTA_WORKERS=4 uv run python run_full.py

Windows NOT: ProcessPoolExecutor surecleri bu dosyayi yeniden ice aktarir;
__main__ korumasi OLMADAN her isci deneyi bastan baslatir. Kaldirma.
"""

from __future__ import annotations

import time

from config import load_config
from sim.experiment import run_experiment, worker_count


def main() -> None:
    cfg = load_config()
    s = cfg.skip_penalty.lambda_sweep
    print("=" * 66)
    print("  TAM DENEY")
    print("=" * 66)
    print(f"  bolge/kirpma : {cfg.region.name} / {cfg.region.clip_mode}")
    print(f"  ufuk         : {cfg.simulation.warmup_days} warmup + "
          f"{cfg.simulation.report_days} rapor gunu")
    print(f"  replikasyon  : {cfg.simulation.num_seeds} seed")
    print(f"  cozucu limiti: uzun {cfg.solvers.time_limit_long_sec} sn / "
          f"odakli {cfg.solvers.time_limit_focused_sec} sn")
    print(f"  lambda       : {s.num} nokta x {s.sweep_seeds} seed x {s.sweep_days} gun")
    print(f"  filo         : {cfg.fleet.num_vehicles} arac")
    print(f"  kosum        : SERI (isci={worker_count()}) - paralellik cozucu")
    print("                 kalitesini dusurur, olculdu: DECISIONS M.10.6")
    print("=" * 66, flush=True)

    t0 = time.perf_counter()
    out = run_experiment(cfg, num_seeds=cfg.simulation.num_seeds,
                         aux_seeds=3, sens_seeds=2)
    dt = time.perf_counter() - t0

    print(f"\nBITTI ({dt / 3600:.2f} saat): {out}")
    print((out / "health_report.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
