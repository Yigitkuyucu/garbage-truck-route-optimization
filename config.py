"""config.yaml -> dogrulanmis Pydantic modelleri.

Bu modul KABUK'tur: sinif/dataclass serbest.
Cekirdek (domain/) bu modulu gormez; degerler sinir gecisinde diziye cevrilir.

Tum sabitler config.yaml'dadir (sihirli sayi yasagi).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Birim donusumleri - sihirli sayi yasaginin tek istisnasi
M3_TO_LITERS = 1000
SECONDS_PER_HOUR = 3600
LITERS_TO_ML = 1000
METERS_PER_100KM = 100_000
KG_PER_TONNE = 1000

Coord = tuple[float, float]
CommercialType = Literal["low", "mid", "high"]
Weekday = Literal[
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ClipMode = Literal["urban_polygon", "circle"]


class Region(_Base):
    name: str
    center: Coord
    radius_m: int = Field(gt=0)             # OSM indirme yaricapi (cache)
    study_radius_m: int = Field(gt=0)       # calisma dairesi (clip_mode=circle)
    clip_mode: ClipMode = "urban_polygon"   # calisma alani nasil kirpilir
    network_type: str = "drive"

    @model_validator(mode="after")
    def _study_within_download(self) -> Region:
        if self.study_radius_m > self.radius_m:
            raise ValueError("study_radius_m, radius_m'i (indirme) asamaz")
        return self


class NamedPoint(_Base):
    name: str
    coord: Coord


class LevelsRange(_Base):
    min: int = Field(gt=0)
    max: int = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> LevelsRange:
        if self.max < self.min:
            raise ValueError("levels max, min'den kucuk olamaz")
        return self


class LevelsModel(_Base):
    residential: LevelsRange
    commercial: LevelsRange


class BuildingModel(_Base):
    m2_per_person: float = Field(gt=0)
    kg_per_person_day: float = Field(gt=0)
    waste_density_kg_m3: float = Field(gt=0)
    levels: LevelsModel
    levels_sensitivity: list[LevelsRange]   # konut kat dagilimi senaryolari

    @property
    def kg_per_liter(self) -> float:
        """Sikistirilmamis cop: litre -> kg (yakit modeli kutleyle calisir, M.8)."""
        return self.waste_density_kg_m3 / M3_TO_LITERS


class DemandCoefficients(_Base):
    residential: float = Field(gt=0)
    commercial_low: float = Field(gt=0)
    commercial_mid: float = Field(gt=0)
    commercial_high: float = Field(gt=0)


class SensitivityLevel(_Base):
    commercial_low: float = Field(gt=0)
    commercial_mid: float = Field(gt=0)
    commercial_high: float = Field(gt=0)


class SensitivityScenarios(_Base):
    low: SensitivityLevel
    mid: SensitivityLevel
    high: SensitivityLevel


class CommercialPoint(_Base):
    name: str
    coord: Coord
    type: CommercialType
    radius_m: int = Field(gt=0)


class MarketZone(_Base):
    name: str
    day: Weekday
    segments: list[tuple[Coord, Coord]]
    radius_m: int = Field(gt=0)


class Containers(_Base):
    point_spacing_m: int = Field(gt=0)
    volume_l: int = Field(gt=0)               # tek bin hacmi
    target_people_per_bin: int = Field(gt=0)  # belediye yogunlugu (880/44689 ~= 51)
    noise_sigma_margin: float = Field(ge=0)   # provizyon gurultu marji (sigma kati)


class Vehicle(_Base):
    body_m3: int = Field(gt=0)
    compaction_ratio: int = Field(gt=0)
    service_time_sec: int = Field(gt=0)
    shift_seconds: int = Field(gt=0)

    @property
    def effective_capacity_l(self) -> int:
        """Etkin kapasite (L): kasa hacmi * sikistirma orani (m3 -> L)."""
        return self.body_m3 * self.compaction_ratio * M3_TO_LITERS


class Fleet(_Base):
    num_vehicles: int = Field(gt=0)


class FuelCoefficients(_Base):
    """Yakit modelinin ORAN katsayilari (duyarlilik senaryolari bunlari degistirir)."""

    base_l_per_100km: float = Field(gt=0)             # saf surus (dur-kalk HARIC)
    slope_l_per_100km_per_tonne: float = Field(ge=0)  # yuk egimi
    stop_start_ml: int = Field(ge=0)                  # durak basi dur-kalk
    compaction_ml_per_liter: float = Field(ge=0)      # sikistirma cevrimi

    # --- cekirdege gecen turevler: mL/m tabaninda (birim donusumu, Bolum 12 istisnasi) ---

    @property
    def base_ml_per_m(self) -> float:
        """Bos kamyon yakit orani (mL/metre)."""
        return self.base_l_per_100km * LITERS_TO_ML / METERS_PER_100KM

    @property
    def slope_ml_per_m_per_kg(self) -> float:
        """Tasinan kutle basina ek yakit orani (mL/metre/kg)."""
        return (
            self.slope_l_per_100km_per_tonne
            * LITERS_TO_ML
            / METERS_PER_100KM
            / KG_PER_TONNE
        )


class Fuel(FuelCoefficients):
    """FAZ 2 yakit/emisyon modeli.

    Amac fonksiyonu birimi TAM SAYI mL; bacak basina yuvarlanir.
    """

    co2_kg_per_l: float = Field(gt=0)                 # dizel emisyon faktoru
    nominal_load_ratio: float = Field(ge=0, le=1.0)   # skip_penalty + B2 referans yuku
    sensitivity: list[FuelCoefficients]               # katsayi duyarlilik senaryolari

    def full_empty_ratio(self, capacity_kg: float) -> float:
        """rho = dolu/bos yakit orani. Saha olcumu ~1.10 (M.8) - capa kontrolu."""
        return 1.0 + self.slope_ml_per_m_per_kg * capacity_kg / self.base_ml_per_m


class Constraints(_Base):
    hygiene_cap_days: int = Field(gt=0)            # hijyen (koku) tavani
    overflow_predict_sigma_k: float = Field(gt=0)  # doluluk-farkindalikli tahmin marji
    hygiene_cap_sensitivity: list[int]            # duyarlilik: tavan degerleri


class LambdaSweep(_Base):
    start: float = Field(gt=0)
    stop: float = Field(gt=0)
    num: int = Field(gt=1)
    sweep_days: int = Field(gt=0)    # tarama ufku (tam ufka gerek yok)
    sweep_seeds: int = Field(gt=0)   # tarama replikasyonu (istatistik degil, secim)

    @model_validator(mode="after")
    def _check_range(self) -> LambdaSweep:
        if self.stop <= self.start:
            raise ValueError("lambda_sweep.stop, start'tan buyuk olmali")
        return self


class SkipPenalty(_Base):
    insertion_cost_k: int = Field(gt=0)
    lambda_sweep: LambdaSweep


class Simulation(_Base):
    warmup_days: int = Field(ge=0)
    report_days: int = Field(gt=0)
    num_seeds: int = Field(gt=0)
    daily_noise_sigma: float = Field(ge=0)
    weekday_multipliers: dict[Weekday, float]
    market_surge_multiplier: float = Field(gt=0)

    @field_validator("weekday_multipliers")
    @classmethod
    def _all_days(cls, v: dict[str, float]) -> dict[str, float]:
        expected = {
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        }
        if set(v) != expected:
            missing = expected - set(v)
            raise ValueError(f"weekday_multipliers eksik gunler: {missing}")
        return v


class Solvers(_Base):
    b1_threshold: float = Field(gt=0, le=1.0)
    time_limit_long_sec: int = Field(gt=0)      # uzun sim (KPI trendi)
    time_limit_focused_sec: int = Field(gt=0)   # odakli karsilastirma (adil F2)
    focused_days: int = Field(gt=0)


class ABC(_Base):
    colony_size: int = Field(gt=0)
    limit: int = Field(gt=0)
    local_search_iters: int = Field(gt=0)
    local_search_window: int = Field(gt=1)   # kisa 2-opt segment penceresi


class Validation(_Base):
    tuik_population: int = Field(gt=0)
    population_tolerance: float = Field(gt=0)
    municipality_population: int = Field(gt=0)
    municipality_trucks: int = Field(gt=0)
    municipality_containers: int = Field(gt=0)
    anchor_tolerance: float = Field(gt=1.0)

    @property
    def people_per_container(self) -> float:
        """Belediye gercek orani: kisi / konteyner ( or. 880/44689 = 51)."""
        return self.municipality_population / self.municipality_containers

    def expected_trucks(self, study_population: float) -> float:
        """Beklenen kamyon = 7 * (calisma_nufus / tum_sehir_nufus)."""
        return self.municipality_trucks * study_population / self.municipality_population

    def expected_containers(self, study_population: float) -> float:
        """Beklenen konteyner = calisma_nufus / (kisi/konteyner orani)."""
        return study_population / self.people_per_container


class Config(_Base):
    seed: int
    region: Region
    depot: NamedPoint
    dump_site: NamedPoint
    building_model: BuildingModel
    demand_coefficients: DemandCoefficients
    sensitivity_scenarios: SensitivityScenarios
    commercial_points: list[CommercialPoint]
    market_zones: list[MarketZone]
    containers: Containers
    vehicle: Vehicle
    fleet: Fleet
    fuel: Fuel
    constraints: Constraints
    skip_penalty: SkipPenalty
    simulation: Simulation
    solvers: Solvers
    abc: ABC
    validation: Validation

    # Ham YAML metni - config hash'i icin (tekrarlanabilirlik)
    raw_text: str = Field(repr=False)

    @property
    def config_hash(self) -> str:
        """Sonuc klasorune yazilan kisa config hash'i."""
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path = "config.yaml") -> Config:
    """config.yaml'i oku, dogrula, Config dondur."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    data["raw_text"] = text
    return Config.model_validate(data)
