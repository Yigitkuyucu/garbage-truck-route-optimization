"""Streamlit paneli - Dinamik Atik Toplama & Rota Optimizasyonu.

Uc sekme:
  KPI    - deneyin URETTIGI CSV'lerden karsilastirma (B0/B1/B2/X1/X2), Pareto,
           duyarlilik. Panel yeniden hesaplamaz; runs/ ciktilarini okur.
  Harita - SECILEN gun + cozucu icin CANLI solve; B0 (referans) vs akilli cozucu
           yan yana. Atlanan konteynerler soluk - projenin tezi burada gorunur.
  Saglik - saglik raporu + canli capalar (nufus/filo/konteyner).

Rapor gorselleri (harita, KPI grafikleri, Pareto) buradan uretilir (A3: panel
raporun malzemesi). KABUK modulu.

Calistir:  uv run streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/dashboard.py` sys.path'e ui/ ekler, proje kokunu DEGIL.
# Kok ekli olmadan `from config import ...` kirilir; bootstrap ile ekle.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from ui.panel_data import (  # noqa: E402
    SMART_SOLVERS,
    SOLVER_NAMES,
    get_config,
    get_dataset,
    lambda_star,
    list_runs,
    load_health,
    load_hygiene_sensitivity,
    load_kpi,
    load_levels_sensitivity,
    load_pareto,
    replay_solver,
)
from ui.panel_map import build_map, map_html  # noqa: E402

st.set_page_config(page_title="Atik Toplama Rota Optimizasyonu", layout="wide")

B0_DUMMY_LIMIT = 1.0  # B0/B1 solve limiti kullanmaz; cache anahtari sabit kalsin


# ----------------------------------------------------------------------------
# Kenar cubugu - kontroller
# ----------------------------------------------------------------------------

st.sidebar.title("Kontroller")

runs = list_runs()
if not runs:
    st.error("runs/ altinda deney ciktisi yok. Once: `uv run python run_full.py`")
    st.stop()

run_names = [p.name for p in runs]
sel_run_name = st.sidebar.selectbox("Deney kosusu", run_names, index=0)
run = next(p for p in runs if p.name == sel_run_name)

st.sidebar.markdown("### Harita")
smart_code = st.sidebar.selectbox(
    "Akilli cozucu (sag harita)", SMART_SOLVERS,
    index=SMART_SOLVERS.index("B2"),
    format_func=lambda c: SOLVER_NAMES[c],
)
seed = st.sidebar.number_input("Seed (fill gerceklemesi)", 0, 9, 0, 1)
lam_default = lambda_star(run)
lam = st.sidebar.number_input(
    "lambda (skip cezasi)", 0.0, 100.0, float(round(lam_default, 4)), 0.05,
    help="Pareto calisma noktasi (tasma=0 en dusuk mesafe). Varsayilan = lambda*.",
)
horizon = st.sidebar.slider("Harita gun ufku", 5, 30, 14,
                            help="Panel yanit suresi icin kisa (deney 90 gun).")
map_tl = st.sidebar.slider(
    "B2/ABC solve limiti (sn)", 0.5, 10.0, 2.0, 0.5,
    help="Yalniz harita gorseli; adil F2 karsilastirmasi degil.",
)
include_dd = st.sidebar.checkbox("Garaj + dokum dahil gorunum", value=False)

st.sidebar.caption(f"config_hash: `{get_config().config_hash}`")


# ----------------------------------------------------------------------------
# Baslik
# ----------------------------------------------------------------------------

st.title("♻️ Dinamik Atik Toplama ve Rota Optimizasyonu")
st.caption(
    "ARASTIRMA paneli - deney sonuclari ve saglik kontrolleri. Gunluk operasyonel "
    "planlama AYRI uygulamadadir (`uv run uvicorn api.main:app`, FAZ 3 / M.9). "
    "**Iddia:** doluluk-tabanli akilli toplama, sabit rotadan iyidir. Faz 2'de manset "
    "metrik yakit/CO2'dir; yakit biriminde OR-Tools ondedir."
)

tab_kpi, tab_map, tab_health = st.tabs(["📊 KPI", "🗺️ Harita", "✅ Saglik"])


# ----------------------------------------------------------------------------
# SEKME: KPI
# ----------------------------------------------------------------------------

with tab_kpi:
    stage_label = st.radio(
        "Kademe", ["Uzun (90 gun trend)", "Odakli (adil 60sn)"],
        horizontal=True,
    )
    stage = "long" if stage_label.startswith("Uzun") else "focused"
    kpi = load_kpi(run, stage)

    if kpi is None:
        st.warning(f"kpi_{stage}.csv bulunamadi.")
    else:
        b0 = kpi[kpi["kod"] == "B0"].iloc[0]
        sm = kpi[kpi["kod"] == smart_code].iloc[0]

        has_fuel = "yakit_L" in kpi.columns  # eski (Faz 1) kosular yakit icermez

        st.subheader(f"{SOLVER_NAMES[smart_code]} - B0'a gore")
        c1, c2, c3, c4 = st.columns(4)
        stop_save = (b0["durak"] - sm["durak"]) / b0["durak"] * 100
        if has_fuel:
            fuel_save = (b0["yakit_L"] - sm["yakit_L"]) / b0["yakit_L"] * 100
            c1.metric("Yakit / gun", f"{sm['yakit_L']:.1f} L",
                      f"-%{fuel_save:.1f}", delta_color="inverse")
            c2.metric("CO2 / gun", f"{sm['co2_kg']:.0f} kg",
                      f"-{b0['co2_kg'] - sm['co2_kg']:.0f} kg", delta_color="inverse")
        else:
            intra_save = (
                (b0["bolge_ici_km"] - sm["bolge_ici_km"]) / b0["bolge_ici_km"] * 100
            )
            c1.metric("Bolge-ici mesafe", f"{sm['bolge_ici_km']:.2f} km",
                      f"-%{intra_save:.1f}", delta_color="inverse")
            c2.metric("Toplam mesafe", f"{sm['toplam_km']:.2f} km")
        c3.metric("Durak / gun", f"{sm['durak']:.0f}",
                  f"-%{stop_save:.1f}", delta_color="inverse")
        c4.metric("Tasma olayi", f"{int(sm['tasma'])}",
                  "hedef: 0", delta_color="off")

        st.caption(
            "FAZ 2 manseti YAKIT/CO2. Kazanc esas olarak AKILLI "
            "ATLAMADAN ve az duraktan gelir; deadhead (sabit ~16 km) degismez. "
            "Yuk duyarli siralama kaldiraci OLCULDU ve kucuk cikti (yakitin ~%1'i) - "
            "yakit modelinin degeri yeni bir optimizasyon kazanci degil, DOGRU "
            "MUHASEBE BIRIMI olmasidir."
        )

        st.divider()
        # --- Karsilastirma grafikleri ---
        gcol1, gcol2 = st.columns(2)
        order = kpi["kod"].tolist()

        def _bar(df: pd.DataFrame, y: str, title: str, fmt: str = ".1f"):
            base = alt.Chart(df).encode(
                x=alt.X("kod:N", sort=order, title=None),
                y=alt.Y(f"{y}:Q", title=title),
                color=alt.Color("kod:N", legend=None,
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["kod", "cozucu", y],
            )
            text = base.mark_text(dy=-6, fontSize=11).encode(
                text=alt.Text(f"{y}:Q", format=fmt))
            return (base.mark_bar() + text).properties(height=260, title=title)

        if has_fuel:
            gcol1.altair_chart(_bar(kpi, "yakit_L", "Yakit (L/gun)", ".1f"),
                               width="stretch")
            # M.8 #2: kalemler AYRISIK - sikistirma tum cozuculerde ~sabit, birlestirilirse
            # tasarruf yuzdesi yaniltici sekilde kuculur.
            parts = ["yk_seyahat", "yk_durkalk", "yk_sikistirma"]
            fuel_stack = kpi.melt(
                id_vars="kod", value_vars=parts,
                var_name="kalem", value_name="litre",
            )
            gcol2.altair_chart(
                alt.Chart(fuel_stack).mark_bar().encode(
                    x=alt.X("kod:N", sort=order, title=None),
                    y=alt.Y("litre:Q", title="Yakit (L/gun)"),
                    color=alt.Color("kalem:N", title="Kalem",
                                    scale=alt.Scale(
                                        domain=parts,
                                        range=["#1f77b4", "#ff7f0e", "#bbbbbb"])),
                    tooltip=["kod", "kalem", "litre"],
                ).properties(height=260,
                             title="Yakit kalemleri (seyahat / dur-kalk / sikistirma)"),
                width="stretch",
            )
        else:
            gcol1.altair_chart(
                _bar(kpi, "bolge_ici_km", "Bolge-ici mesafe (km/gun)", ".2f"),
                width="stretch")
            gcol2.altair_chart(_bar(kpi, "durak", "Durak sayisi (gun)", ".0f"),
                               width="stretch")
        gcol3, gcol4 = st.columns(2)
        gcol3.altair_chart(_bar(kpi, "durak", "Durak sayisi (gun)", ".0f")
                           if has_fuel else
                           _bar(kpi, "doluluk_%", "Ort. toplama dolulugu (%)"),
                           width="stretch")

        # Toplam km yigilmis: sabit (deadhead) + bolge-ici
        stacked = kpi.melt(
            id_vars="kod", value_vars=["sabit_km", "bolge_ici_km"],
            var_name="bilesen", value_name="km",
        )
        stack_chart = alt.Chart(stacked).mark_bar().encode(
            x=alt.X("kod:N", sort=order, title=None),
            y=alt.Y("km:Q", title="Toplam mesafe (km/gun)"),
            color=alt.Color("bilesen:N", title="Bilesen",
                            scale=alt.Scale(
                                domain=["sabit_km", "bolge_ici_km"],
                                range=["#bbbbbb", "#1f77b4"])),
            tooltip=["kod", "bilesen", "km"],
        ).properties(height=260, title="Toplam = sabit (deadhead) + bolge-ici")
        gcol4.altair_chart(stack_chart, width="stretch")

        st.divider()
        st.markdown("**KPI tablosu**")
        st.dataframe(kpi, width="stretch", hide_index=True)

    # --- Pareto + duyarlilik ---
    st.divider()
    pcol, scol = st.columns([1, 1])
    with pcol:
        st.markdown("**lambda kalibrasyonu - Pareto (mesafe vs tasma)**")
        png = run / "lambda_pareto.png"
        if png.exists():
            st.image(str(png), width="stretch")
        pareto = load_pareto(run)
        if pareto is not None:
            st.caption(f"Calisma noktasi lambda* = {lambda_star(run):.3g} "
                       "(tasma=0 olan en dusuk mesafe).")
    with scol:
        st.markdown("**Duyarlilik - hijyen tavani (3/5/7 gun)**")
        hyg = load_hygiene_sensitivity(run)
        if hyg is not None:
            st.dataframe(hyg, width="stretch", hide_index=True)
        st.markdown("**Duyarlilik - konut kat dagilimi**")
        lv = load_levels_sensitivity(run)
        if lv is not None:
            st.dataframe(lv, width="stretch", hide_index=True)
            # Artefakt uyarisi: B2 bir senaryoda zaman limitine takildiginda
            # durak=0 / tasma sismis olur - gecersiz satir, bozuk sanilmasin.
            bad = lv[lv["B2_durak"] == 0]
            if not bad.empty:
                rows = ", ".join(f"kat {r}" for r in bad["konut_kat"])
                st.caption(
                    f"⚠️ **{rows}** satirindaki B2_durak=0 / tasma sismis degerleri "
                    "**gecersizdir** - bu senaryoda talep filo kapasitesini asar, yani "
                    "senaryo fizibil degildir (cozucu artefakti DEGIL). "
                    "**Ana senaryo kat 2-4'tur.**"
                )


# ----------------------------------------------------------------------------
# SEKME: HARITA
# ----------------------------------------------------------------------------

with tab_map:
    ds = get_dataset()
    b0_states = replay_solver("B0", int(seed), float(lam), int(horizon), B0_DUMMY_LIMIT)
    smart_states = replay_solver(
        smart_code, int(seed), float(lam), int(horizon), float(map_tl)
    )
    n_days = min(len(b0_states), len(smart_states))
    if n_days == 0:
        st.warning("Replay bos dondu.")
        st.stop()

    day = st.slider("Rapor gunu", 0, n_days - 1, 0,
                    help="Warm-up (14 gun) atilmistir; 0 = ilk raporlanan gun.")
    s_b0 = b0_states[day]
    s_sm = smart_states[day]

    st.markdown(
        "**Renkler:** yesil→kirmizi = bos→dolu · gri = **atlandi** · "
        "kesikli = deadhead (sabit) · her arac ayri renk."
    )

    left, right = st.columns(2)
    for col, code, state in (
        (left, "B0", s_b0), (right, smart_code, s_sm)
    ):
        with col:
            m = build_map(ds, state, title=SOLVER_NAMES[code],
                          include_depot_dump=include_dd)
            components.html(map_html(m), height=540)
            r = state.result
            # FAZ 2 manseti: yakit/CO2 one; mesafe yan metrik (M.7)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Yakit", f"{r.fuel_ml / 1000:.1f} L")
            mc2.metric(
                "CO2", f"{r.fuel_ml / 1000 * get_config().fuel.co2_kg_per_l:.0f} kg"
            )
            mc3.metric("Durak", int(state.visited.sum()))
            mc4.metric("Atlanan", r.n_skipped)
            st.caption(
                f"yakit kalemleri: seyahat {r.fuel_travel_ml / 1000:.1f} L · "
                f"dur-kalk {r.fuel_stop_ml / 1000:.1f} L · "
                f"sikistirma {r.fuel_compaction_ml / 1000:.1f} L  |  "
                f"mesafe: bolge-ici {r.intra_distance / 1000:.2f} km / "
                f"toplam {r.total_distance / 1000:.2f} km"
            )

    st.caption(
        f"Gun {day} · seed {seed} · lambda {lam:.3g} · "
        f"B0 vs {SOLVER_NAMES[smart_code]}. Sol referans (hepsini toplar), "
        "sag akilli atlama. Fark = ATLANAN gri konteynerler."
    )


# ----------------------------------------------------------------------------
# SEKME: SAGLIK
# ----------------------------------------------------------------------------

with tab_health:
    cfg = get_config()
    ds = get_dataset()
    val = cfg.validation
    pop = float(ds.residents.sum())        # CALISMA bolgesi nufusu (study daire)
    n_bins = int(ds.n_bins.sum())
    n_points = ds.num_containers
    tol = val.anchor_tolerance             # oran bu araligin disi -> uyar

    def _ok(ratio: float) -> str:
        return "✅" if (1 / tol) <= ratio <= tol else "⚠️"

    st.subheader("Canli capalar - calisma bolgesi")
    a1, a2, a3 = st.columns(3)

    # 1) Filo capasi: girdi vs olcekli beklenti (7 * calisma/tum-sehir)
    exp_trucks = val.expected_trucks(pop)
    tr_ratio = cfg.fleet.num_vehicles / exp_trucks if exp_trucks else 0.0
    a1.metric(f"{_ok(tr_ratio)} Filo (girdi)", f"{cfg.fleet.num_vehicles} arac",
              f"beklenen ~{exp_trucks:.1f} · oran {tr_ratio:.2f}", delta_color="off")

    # 2) Konteyner yogunlugu: model kisi/bin vs referans oran (B5d ikinci capa)
    ppb_model = pop / n_bins if n_bins else 0.0
    ppb_ratio = ppb_model / val.people_per_container if val.people_per_container else 0.0
    a2.metric(f"{_ok(ppb_ratio)} Kisi / konteyner", f"{ppb_model:.0f}",
              f"referans {val.people_per_container:.0f} · oran {ppb_ratio:.2f}",
              delta_color="off")

    # 3) Nufus: olcegin GIRDISI (kirpilmis daire). Tum-sehir kalibrasyonu (0.94,
    #    kat-3) ayri bir capadir (data.build, kirpilmamis kutu) - burada kiyaslanmaz.
    a3.metric("Nufus (calisma bolgesi)", f"{pop:,.0f}",
              f"{n_points} nokta / {n_bins} bin", delta_color="off")

    st.caption(
        "Capalar KATMANLI ve OLCEKLI: filo ve kisi/konteyner, "
        f"calisma bolgesi nufusuna gore beklentiyle karsilastirilir (tolerans {tol:g}x). "
        "Tum-sehir nufus kalibrasyonu (model/TUIK ~0.94, kat-3) veri insa asamasinda, "
        "kirpilmamis indirme kutusunda dogrulanir - bu kirpilmis daireyle kiyaslanmaz."
    )

    st.divider()
    st.subheader("Deney saglik raporu")
    st.code(load_health(run), language="text")
