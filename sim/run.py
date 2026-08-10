"""Deney kosum giris noktasi.

    uv run python -m sim.run --config config.yaml [--seeds N]

Sonuc runs/<zaman>_<confighash>/ altina yazilir (config kopyasi + CSV'ler +
saglik raporu + Pareto).
"""

from __future__ import annotations

import argparse

from config import load_config
from sim.experiment import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Deney kosumu (Adim 7)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="replikasyon (vars: config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    num_seeds = args.seeds if args.seeds is not None else cfg.simulation.num_seeds
    out = run_experiment(cfg, num_seeds=num_seeds, config_path=args.config)
    print((out / "health_report.txt").read_text(encoding="utf-8"))
    print(f"\nCikti klasoru: {out}")


if __name__ == "__main__":
    main()
