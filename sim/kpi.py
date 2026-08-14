"""KPI toplama + saglik kontrolleri.

FAZ 2 mansett KPI: YAKIT (L/gun) + CO2 (kg/gun). Durak ve doluluk yaninda kalir;
mesafe YAN metriktir. Gerekce (M.7): atik toplamanin gercek maliyet birimi yakittir
ve ABC'nin durak avantaji ancak yakitta mansete yansir.

Yakit kalemleri AYRISIK raporlanir (seyahat / dur-kalk / sikistirma): sikistirma
tum cozuculerde ayni (kutle korunumu) ve yalnizca paydayi sisirir - birlestirilirse
tasarruf yuzdeleri yaniltici sekilde kuculur (M.8 zayiflik #2).

B1 infeasible gunleri ortalamalara KATILMAZ (yaniltici olur) - ayri metrik.

KABUK modulu. CO2 burada yakittan turetilir (amac fonksiyonunun parcasi DEGIL).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from sim.engine import RunResult

ML_TO_L = 1000  # birim donusumu (Bolum 12 istisnasi)


@dataclass
class SolverKPIs:
    """Bir cozucunun tum seed'ler uzerinde toplanmis KPI'lari (feasible gunler)."""

    code: str
    name: str
    n_days_total: int         # 90 x seed
    infeasible_days: int      # ortalamalara KATILMAYAN gunler
    solver_failures: int      # cozucunun cozum bulamayip greedy'ye dustugu gun
    # --- FAZ 2 MANSET: yakit + CO2 (feasible gunler ort±std) ---
    mean_fuel_l: float
    std_fuel_l: float
    mean_co2_kg: float
    mean_fuel_travel_l: float      # kalem: seyahat (mesafe + yuk)
    mean_fuel_stop_l: float        # kalem: dur-kalk (durak sayisiyla olcekli)
    mean_fuel_compaction_l: float  # kalem: sikistirma (cozuculer arasi ~SABIT)
    mean_load_term_l: float        # M.7: OR-Tools'un erisemedigi terim
    # --- Faz 1 metrikleri (yan metrik olarak kalir) ---
    mean_stops: float
    mean_service_s: float     # durak servis suresi (dur-kalk/sikistirma proxy)
    mean_fill_pct: float      # ortalama toplama dolulugu (gercek basari gostergesi)
    mean_intra_km: float
    std_intra_km: float
    mean_total_km: float
    std_total_km: float
    mean_fixed_km: float
    # saglik
    total_overflow_events: int
    mass_ok: bool
    mean_shift_util: float

    @property
    def infeasible_rate_per_90(self) -> float:
        if self.n_days_total == 0:
            return 0.0
        return self.infeasible_days / self.n_days_total * 90

    @property
    def optimizable_share(self) -> float:
        """Optimize-EDILEBILIR pay: bolge-ici km / toplam km.

        Yapisal bir OLCUM'dur, esik degil (M.11): geri kalani garaj-bolge-dokum
        deadhead'idir ve hicbir cozucu ona dokunamaz. B0 uzerinde okunur;
        tasarruf yuzdelerinin hangi payda uzerinde konustugunu belirler.
        """
        if self.mean_total_km <= 0:
            return 0.0
        return self.mean_intra_km / self.mean_total_km


def aggregate(
    code: str, name: str, per_seed: list[RunResult], *, co2_kg_per_l: float
) -> SolverKPIs:
    """Bir cozucunun seed-bazli kosularini topla. Ortalamalar YALNIZCA feasible
    gunler uzerinden (B1 infeasibility karari).

    co2_kg_per_l: config'den gelir (dizel emisyon faktoru, M.8). CO2 yakittan
    TUREV bir metriktir - amac fonksiyonunun parcasi degil.
    """
    feasible_recs = [r for run in per_seed for r in run.records if r.feasible]
    n_days_total = sum(len(run.records) for run in per_seed)
    infeasible_days = n_days_total - len(feasible_recs)
    overflow = sum(run.total_overflow_events for run in per_seed)
    mass_ok = all(run.mass_ok for run in per_seed)

    def _mean(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    def _std(xs: list[float]) -> float:
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    intra = [r.intra_distance / 1000 for r in feasible_recs]
    total = [r.total_distance / 1000 for r in feasible_recs]
    fuel = [r.fuel_ml / ML_TO_L for r in feasible_recs]
    return SolverKPIs(
        code=code,
        name=name,
        n_days_total=n_days_total,
        infeasible_days=infeasible_days,
        solver_failures=sum(r.solver_failures for r in per_seed),
        mean_fuel_l=_mean(fuel),
        std_fuel_l=_std(fuel),
        mean_co2_kg=_mean(fuel) * co2_kg_per_l,
        mean_fuel_travel_l=_mean([r.fuel_travel_ml / ML_TO_L for r in feasible_recs]),
        mean_fuel_stop_l=_mean([r.fuel_stop_ml / ML_TO_L for r in feasible_recs]),
        mean_fuel_compaction_l=_mean(
            [r.fuel_compaction_ml / ML_TO_L for r in feasible_recs]
        ),
        mean_load_term_l=_mean([r.fuel_load_term_ml / ML_TO_L for r in feasible_recs]),
        mean_stops=_mean([float(r.n_visited) for r in feasible_recs]),
        mean_service_s=_mean([float(r.service_time_s) for r in feasible_recs]),
        mean_fill_pct=_mean([r.mean_fill_pct for r in feasible_recs]),
        mean_intra_km=_mean(intra),
        std_intra_km=_std(intra),
        mean_total_km=_mean(total),
        std_total_km=_std(total),
        mean_fixed_km=_mean([r.fixed_distance / 1000 for r in feasible_recs]),
        total_overflow_events=overflow,
        mass_ok=mass_ok,
        mean_shift_util=_mean([r.shift_util_pct for r in feasible_recs]),
    )


# BOLGE-ICI TASARRUF SUPHE ESIGI (M.11 - kategori hatasi duzeltildi)
#
# ONCEKI HALI HATALIYDI: sabit 0.17'ye "yapisal tavan" deniyor ve bolge-ici
# TASARRUF yuzdesiyle karsilastiriliyordu. Oysa %17, F-KPI-a'da olculen
# OPTIMIZE-EDILEBILIR PAY'di (bolge-ici km / toplam km) - bambaska bir
# buyukluk. Iki farkli seyi kiyaslayan kontrol, olcek buyuyunce yanlis alarm
# verdi (B1 %26 > %17). Pay artik %37 (784 nokta); tasarrufla iliskisi yok.
#
# DOGRU esik F-KPI-a'da zaten yaziliydi: bolge-ici tasarrufta %15-25 beklenir,
# "%40+ cikarsa suphelen". H1'in %10-30 bandi MANSET metrige (yakit) uygulanir
# ve o kontrol ayrica kosar (THIRTY_PCT_RULE).
INTRA_SAVING_SUSPECT = 0.40

# %30 KURALI: %30'dan buyuk tasarruf neredeyse her zaman hata isaretidir.
# FAZ 2'de bu kural MANSET metrige (yakit) uygulanir.
THIRTY_PCT_RULE = 0.30

# M.7: yuk baglasimi terimi olculdu, yakitin ~%1'i. Bunu belirgin asarsa ya
# katsayilar degismistir ya da yuk takibinde hata vardir.
LOAD_TERM_EXPECTED_MAX = 0.05  # |yuk terimi| / yakit


def health_checks(kpis: list[SolverKPIs], b0_code: str = "B0") -> list[str]:
    """Otomatik saglik kontrolleri (Bolum 10 + H1). Uyari listesi dondur."""
    warnings: list[str] = []
    by_code = {k.code: k for k in kpis}

    # 1. Kutle korunumu
    for k in kpis:
        if not k.mass_ok:
            warnings.append(f"KUTLE KORUNUMU BOZUK: {k.code} (kod hatasi - Bolum 10 #1)")

    # 2. Tasma = 0 (her cozucu)
    for k in kpis:
        if k.total_overflow_events > 0:
            warnings.append(
                f"TASMA: {k.code} {k.total_overflow_events} olay (0 olmali - Bolum 10 #3)"
            )

    b0 = by_code.get(b0_code)

    # 3. %30 kurali - FAZ 2 MANSET metrigi (yakit) uzerinden (Bolum 10)
    if b0 is not None and b0.mean_fuel_l > 0:
        for k in kpis:
            if k.code == b0_code:
                continue
            saving = (b0.mean_fuel_l - k.mean_fuel_l) / b0.mean_fuel_l
            if saving > THIRTY_PCT_RULE:
                warnings.append(
                    f"%30 KURALI: {k.code} yakit tasarrufu %{saving * 100:.0f} "
                    f"> %{THIRTY_PCT_RULE * 100:.0f}. Literaturdeki gercekci aralik "
                    f"%10-30. Once saglik kontrolleri, sonra rapor. SUPHE BILDIR."
                )

    # 4. Bolge-ici tasarruf suphe esigi (yan kontrol; manset kontrol #3'tur)
    if b0 is not None and b0.mean_intra_km > 0:
        for k in kpis:
            if k.code == b0_code:
                continue
            saving = (b0.mean_intra_km - k.mean_intra_km) / b0.mean_intra_km
            if saving > INTRA_SAVING_SUSPECT:
                warnings.append(
                    f"BOLGE-ICI TASARRUF SUPHELI: {k.code} %{saving * 100:.0f} "
                    f"> %{INTRA_SAVING_SUSPECT * 100:.0f}. F-KPI-a beklentisi %15-25. "
                    f"Senaryo karismis ya da hata olabilir. SUPHE BILDIR."
                )

    # 5. Cozucu basarisizligi: greedy'ye dusulen gunler SESSIZ KALMAZ
    for k in kpis:
        if k.solver_failures:
            pct = 100 * k.solver_failures / max(k.n_days_total, 1)
            warnings.append(
                f"COZUCU COZUM BULAMADI: {k.code} {k.solver_failures} gun "
                f"(%{pct:.1f}) greedy yedegine dustu. O gunler fizibil sayilmaz ve "
                f"ortalamalara girmez. Pay buyukse sure limiti yetersizdir."
            )

    # 6. Yuk baglasimi capasi (M.7): olculen terim yakitin ~%1'i olmali
    for k in kpis:
        if k.mean_fuel_l <= 0:
            continue
        share = abs(k.mean_load_term_l) / k.mean_fuel_l
        if share > LOAD_TERM_EXPECTED_MAX:
            warnings.append(
                f"YUK TERIMI BEKLENENDEN BUYUK: {k.code} |yuk baglasimi| "
                f"yakitin %{share * 100:.1f}'i > %{LOAD_TERM_EXPECTED_MAX * 100:.0f}. "
                f"M.7 bunu ~%1 olcmustu - katsayilar degisti ya da yuk takibi hatali."
            )
    return warnings
