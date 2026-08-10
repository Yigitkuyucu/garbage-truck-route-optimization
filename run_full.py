"""Temsili tam kosu: 10 seed x 90 gun, dusuk cozucu limiti (per-container must_visit
skip room'u belirledigi icin limit sonucu neredeyse degistirmez).

FAZ 2: manset metrik YAKIT/CO2. lambda SIFIRDAN turetilir -
skip_penalty mL birimine gectigi icin Faz 1'in lambda'si TASINMAZ. Bu yuzden
tarama 4 -> 12 noktaya cikarildi: Pareto egrisi raporun ana gorsellerinden biri.
"""
from config import load_config
from sim.experiment import run_experiment

c = load_config().model_copy(deep=True)
# ufuk: TAM (warmup 14 + report 90), 10 seed
object.__setattr__(c.simulation, "warmup_days", 14)
object.__setattr__(c.simulation, "report_days", 90)
# lambda: Faz 2'de yeniden turetiliyor -> ince tarama (M.7 Adim 6)
object.__setattr__(c.skip_penalty.lambda_sweep, "num", 12)
# cozucu limitleri dusuruldu (temsili); yardimci kademeler az seed
object.__setattr__(c.solvers, "time_limit_long_sec", 2)
object.__setattr__(c.solvers, "time_limit_focused_sec", 8)
object.__setattr__(c.solvers, "focused_days", 6)
object.__setattr__(c.abc, "colony_size", 20)

out = run_experiment(c, num_seeds=10, aux_seeds=2)
print(f"BITTI: {out}")
print((out / "health_report.txt").read_text(encoding="utf-8"))
