"""
ATN Dashboard — KI-gestütztes Netzberechnungs-Dashboard
========================================================
Basis: Allgemeine Theorie der Technischen Netze (Strelow, THM Gießen)

Start:
    cd Strelow
    streamlit run dashboard/app.py
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── ATN-Framework und Dashboard-Module einbinden ─────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_atn_path = os.path.join(_here, "..", "atn")
sys.path.insert(0, _atn_path)
sys.path.insert(0, _here)   # für csv_import

from atn.networks.district_heating import DistrictHeatingNetwork
from atn.networks.water import WaterNetwork
from atn.networks.gas import GasNetwork, PressureLevel
from atn.networks.electrical import DCNetwork, ACNetwork
from atn.visualization.plots import (
    plot_network, plot_pressure_profile, plot_temperature_profile
)
from csv_import import (
    detect_columns, missing_required, roles_for_domain,
    build_network, ROLE_LABELS,
)
from geo_networks import BUILDERS as GEO_BUILDERS
from geo_map import build_map as build_geo_map
from streamlit_folium import st_folium

# ── Seitenconfig ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATN Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ATN Dashboard · THM Gießen · Prof. Dr.-Ing. O. Strelow"},
)

# ── Session State initialisieren ──────────────────────────────────────────────
for key, default in [
    ("network",      None),
    ("result",       None),
    ("nodes_df",     None),
    ("edges_df",     None),
    ("coords",       None),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Beispielnetze ─────────────────────────────────────────────────────────────

def build_water_example() -> WaterNetwork:
    net = WaterNetwork("Stadtverteilnetz (Beispiel)")
    net.add_reservoir("HB1", head=45.0, elevation=30.0)
    net.add_pipe("HB1", "K1", 800, 0.20, hazen_williams_c=140, label="P1")
    net.add_pipe("HB1", "K2", 600, 0.15, hazen_williams_c=140, label="P2")
    net.add_pipe("K1",  "K3", 500, 0.15, hazen_williams_c=140, label="P3")
    net.add_pipe("K2",  "K4", 400, 0.10, hazen_williams_c=140, label="P4")
    net.add_pipe("K3",  "K5", 300, 0.10, hazen_williams_c=140, label="P5")
    net.add_demand("K1", flow=0.004, elevation=15.0)
    net.add_demand("K2", flow=0.003, elevation=12.0)
    net.add_demand("K3", flow=0.002, elevation=18.0)
    net.add_demand("K4", flow=0.002, elevation=10.0)
    net.add_demand("K5", flow=0.001, elevation=16.0)
    return net


def build_gas_example(level: str) -> GasNetwork:
    pl = {"ND": PressureLevel.ND, "MD": PressureLevel.MD, "HD": PressureLevel.HD}[level]
    net = GasNetwork("MD-Stadtverteilnetz (Beispiel)", pressure_level=pl)
    p_feed = {"ND": 0.023, "MD": 0.5, "HD": 40.0}[level]
    net.add_feed("UeST", pressure_bar=p_feed)
    net.add_pipe("UeST", "K1", 400, 0.15, label="G1")
    net.add_pipe("UeST", "K2", 300, 0.10, label="G2")
    net.add_pipe("K1",   "K3", 350, 0.10, label="G3")
    net.add_pipe("K2",   "K4", 450, 0.08, label="G4")
    net.add_pipe("K3",   "K4", 200, 0.08, label="G5")
    q = {"ND": [8, 6, 4, 5], "MD": [120, 80, 60, 100], "HD": [2000, 1500, 1000, 1200]}[level]
    net.add_offtake("K1", flow_m3h=q[0])
    net.add_offtake("K2", flow_m3h=q[1])
    net.add_offtake("K3", flow_m3h=q[2])
    net.add_offtake("K4", flow_m3h=q[3])
    return net


def build_heating_example() -> DistrictHeatingNetwork:
    net = DistrictHeatingNetwork("Fernwärmenetz (Beispiel)")
    net.add_heat_source("HW", supply_temp=80.0, mass_flow=5.0)
    net.add_pipe("HW", "K1", length=300, diameter=0.10, label="R1")
    net.add_pipe("HW", "K2", length=500, diameter=0.08, label="R2")
    net.add_pipe("K1", "K3", length=200, diameter=0.06, label="R3")
    net.add_consumer("K1", heat_load=80_000,  return_temp=55.0)
    net.add_consumer("K2", heat_load=120_000, return_temp=55.0)
    net.add_consumer("K3", heat_load=60_000,  return_temp=55.0)
    return net


def build_ac_example() -> ACNetwork:
    net = ACNetwork("Wechselstromnetz (Beispiel)")
    net.add_node("Q1", external_flow=10.0)   # Stromquelle 10 A
    net.add_impedance("Q1", "K1", R=2.0, X=1.5, label="Z1")
    net.add_impedance("Q1", "K2", R=3.0, X=2.0, label="Z2")
    net.add_impedance("K1", "K2", R=1.5, X=1.0, label="Z3")
    net.add_impedance("K2", "K3", R=2.5, X=1.5, label="Z4")
    net.add_node("K1", external_flow=-4.0)
    net.add_node("K2", external_flow=-3.0)
    net.add_node("K3", external_flow=-3.0)
    return net


def build_dc_example() -> DCNetwork:
    net = DCNetwork("Gleichstromnetz (Beispiel)")
    net.add_current_source("Q1", current=10.0)
    net.add_resistor("Q1", "K1", resistance=2.0, label="R1")
    net.add_resistor("Q1", "K2", resistance=3.0, label="R2")
    net.add_resistor("K1", "K2", resistance=1.5, label="R3")
    net.add_resistor("K2", "K3", resistance=2.5, label="R4")
    net.add_node("K1", external_flow=-4.0)
    net.add_node("K2", external_flow=-3.0)
    net.add_node("K3", external_flow=-3.0)
    return net


def network_to_dfs(net) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = [{"Knoten": n, "Ext. Fluss": round(net._external_flows.get(n, 0.0), 6)}
             for n in net._nodes]
    edges = [{"Von": e[0], "Nach": e[1], "Label": e[2],
              "R (init)": round(net._resistances.get(e[2], 0.0), 6)}
             for e in net._edges]
    return pd.DataFrame(nodes), pd.DataFrame(edges)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ATN Dashboard")
    st.markdown("*Allg. Theorie der Technischen Netze*")
    st.markdown("*nach Prof. Dr.-Ing. Olaf Strelow*")
    st.divider()

    domain = st.selectbox(
        "Netzdomäne",
        ["Wasser", "Gas", "Fernwärme", "Strom (DC)", "Strom (AC)"],
        key="domain_select",
    )

    pressure_level_str = "MD"
    if domain == "Gas":
        pressure_level_str = st.selectbox("Druckstufe", ["ND", "MD", "HD"], index=1)

    st.divider()
    st.markdown("**Modulstatus**")
    for label, done in [
        ("Fernwärme", True), ("Wasser", True), ("Gas ND/MD/HD", True),
        ("Strom DC", True), ("Strom AC", True), ("Sektorenkopplung", True),
        ("IO-Parser", False),
    ]:
        icon = "✅" if done else "🔧"
        st.markdown(f"{icon} {label}")

    st.divider()
    st.caption("THM Gießen\nProf. Dr.-Ing. O. Strelow\natn-framework v0.1.0")

# ── Kopfzeile ─────────────────────────────────────────────────────────────────
icons = {"Wasser": "💧", "Gas": "⛽", "Fernwärme": "🔥", "Strom (DC)": "⚡", "Strom (AC)": "〜"}
st.title(f"{icons.get(domain, '🔧')} ATN Dashboard — {domain}")

tab_netz, tab_calc, tab_results, tab_ki = st.tabs([
    "🔧  Netzmodell",
    "▶  Berechnung",
    "📊  Ergebnisse",
    "🤖  KI-Assistent",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NETZMODELL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_netz:
    col_links, col_rechts = st.columns([1, 2])

    with col_links:
        st.subheader("Netz laden")

        if st.button(f"📂 Beispielnetz laden ({domain})",
                     type="primary", use_container_width=True):
            builder = GEO_BUILDERS[domain]
            if domain in ("Gas", "Wasser"):
                net, coords = builder(pressure_level_str)
            else:
                net, coords = builder()

            st.session_state.network = net
            st.session_state.result  = None
            st.session_state.coords  = coords
            st.session_state.nodes_df, st.session_state.edges_df = network_to_dfs(net)
            st.success(f"Geladen: {len(net._nodes)} Knoten, {len(net._edges)} Leitungen")

        st.divider()
        st.subheader("CSV hochladen")
        uploaded = st.file_uploader(
            "Leitungs-CSV (von, nach, label, laenge, durchmesser, …)",
            type=["csv"],
            help="Spalten werden automatisch erkannt. Mapping danach prüfbar.",
        )
        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV-Lesefehler: {e}")
                csv_df = None

            if csv_df is not None and not csv_df.empty:
                st.markdown(f"**{len(csv_df)} Zeilen geladen** — Spalten: "
                            + ", ".join(f"`{c}`" for c in csv_df.columns))

                # ── Automatische Spalten-Erkennung ────────────────────────────
                auto_map = detect_columns(csv_df, domain)
                all_cols = ["—"] + list(csv_df.columns)

                st.markdown("#### Spalten-Mapping")
                st.caption("Automatisch erkannte Zuordnung — bitte prüfen und ggf. korrigieren.")

                roles   = roles_for_domain(domain)
                n_cols  = 3
                ui_cols = st.columns(n_cols)
                col_map: dict[str, str | None] = {}

                for i, role in enumerate(roles):
                    detected = auto_map.get(role)
                    idx = all_cols.index(detected) if detected in all_cols else 0
                    chosen = ui_cols[i % n_cols].selectbox(
                        ROLE_LABELS.get(role, role),
                        all_cols,
                        index=idx,
                        key=f"csvmap_{role}",
                    )
                    col_map[role] = None if chosen == "—" else chosen

                # ── Fehlende Pflichtfelder warnen ────────────────────────────
                missing = missing_required(col_map, domain)
                if missing:
                    labels = [ROLE_LABELS.get(r, r) for r in missing]
                    st.warning(f"Pflichtfelder fehlen: {', '.join(labels)}")
                else:
                    # ── Vorschau ─────────────────────────────────────────────
                    with st.expander("Vorschau (erste 5 Zeilen)"):
                        st.dataframe(csv_df.head(), use_container_width=True)

                    if st.button("🔧 Netz aus CSV aufbauen",
                                 type="primary", use_container_width=True,
                                 key="csv_build"):
                        try:
                            net = build_network(
                                csv_df, col_map, domain,
                                pressure_level=pressure_level_str,
                                net_name=uploaded.name.replace(".csv", ""),
                            )
                            st.session_state.network = net
                            st.session_state.result  = None
                            st.session_state.coords  = None
                            st.session_state.nodes_df, st.session_state.edges_df = \
                                network_to_dfs(net)
                            st.success(
                                f"Netz aufgebaut: {len(net._nodes)} Knoten, "
                                f"{len(net._edges)} Leitungen"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Netz-Aufbau fehlgeschlagen: {e}")

        if st.session_state.network is not None:
            st.divider()
            net = st.session_state.network
            gj  = net.analyze()
            st.subheader("Netzeigenschaften")
            st.metric("Knoten",       len(net._nodes))
            st.metric("Leitungen",    len(net._edges))
            st.metric("Rang",         gj.rank)
            st.metric("Freiheitsgrad", gj.dof)
            st.metric("Maschen",      len(gj.mesh_rows))

    with col_rechts:
        if st.session_state.nodes_df is not None:
            st.subheader("Knoten")
            st.dataframe(st.session_state.nodes_df,
                         use_container_width=True, hide_index=True)
            st.subheader("Leitungen")
            st.dataframe(st.session_state.edges_df,
                         use_container_width=True, hide_index=True)

            # Netz-Topologie visualisieren
            st.subheader("Netz-Topologie")
            net    = st.session_state.network
            coords = st.session_state.coords
            if coords:
                # Geo-Karte mit OSM-Hintergrund
                fmap = build_geo_map(net, coords, domain)
                st_folium(fmap, use_container_width=True, height=480,
                          returned_objects=[])
            else:
                # Fallback: schematischer Topologie-Plot (CSV-Import)
                try:
                    fig = plot_network(net, title=f"Topologie: {net.name}")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.caption(f"Topologie-Plot nicht verfügbar: {e}")
        else:
            st.info("Bitte zuerst ein Netz laden.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BERECHNUNG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_calc:
    if st.session_state.network is None:
        st.warning("Bitte zuerst im Tab **Netzmodell** ein Netz laden.")
    else:
        net = st.session_state.network

        st.subheader("Berechnungsparameter")
        col1, col2 = st.columns(2)

        with col1:
            max_iter = st.slider("Max. Iterationen", 10, 500, 100)
            tol = st.select_slider(
                "Konvergenzschwelle",
                options=[1e-4, 1e-5, 1e-6, 1e-7],
                value=1e-6,
                format_func=lambda x: f"{x:.0e}",
            )

        with col2:
            st.markdown("**Aktuelles Netz**")
            st.markdown(f"- Domäne: **{domain}**")
            st.markdown(f"- Knoten: **{len(net._nodes)}**")
            st.markdown(f"- Leitungen: **{len(net._edges)}**")
            if domain == "Gas":
                st.markdown(f"- Druckstufe: **{pressure_level_str}**")

        st.divider()

        if st.button("▶ Berechnung starten", type="primary", use_container_width=True):
            with st.spinner("ATN-Berechnung läuft..."):
                try:
                    if domain == "Wasser":
                        res = net.solve_hydraulic(max_iter=max_iter, tol=tol)
                        st.session_state.result = ("water", res)

                    elif domain == "Gas":
                        res = net.solve_hydraulic(max_iter=max_iter, tol=tol)
                        st.session_state.result = ("gas", res)

                    elif domain == "Fernwärme":
                        hyd  = net.solve_hydraulic(max_iter=max_iter, tol=tol)
                        therm = net.solve_thermal_stationary(hyd)
                        st.session_state.result = ("heating", hyd, therm)

                    elif domain == "Strom (AC)":
                        res = net.solve_ac()
                        st.session_state.result = ("ac", res)

                    else:  # Strom DC
                        res = net.solve()
                        st.session_state.result = ("dc", res)

                    st.success("✅ Berechnung abgeschlossen — Ergebnisse im Tab **Ergebnisse**.")
                except Exception as e:
                    st.error(f"Fehler: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ERGEBNISSE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if st.session_state.result is None:
        st.info("Bitte zuerst eine Berechnung durchführen.")
    else:
        kind = st.session_state.result[0]

        # ── Wasser ────────────────────────────────────────────────────────────
        if kind == "water":
            res = st.session_state.result[1]
            st.subheader("Wasserversorgungsnetz — Ergebnisse")

            ok = len(res.pressure_violations) == 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Iterationen", res.iterations)
            c2.metric("Konvergenz",  "✓ ja" if res.converged else "✗ nein")
            c3.metric("DVGW W 303",
                      "✅ OK" if ok else f"⚠ {len(res.pressure_violations)} Verletzung(en)")
            c4.metric("Min. Druck",  f"{min(res.pressures.values()):.3f} bar")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Knoten — Druck**")
                df = pd.DataFrame([
                    {"Knoten": n,
                     "Druckhöhe [m]": f"{res.heads[n]:.2f}",
                     "Druck [bar]":   f"{res.pressures[n]:.3f}",
                     "Status": "✓" if n not in res.pressure_violations else "⚠"}
                    for n in sorted(res.heads)
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            with col_b:
                st.markdown("**Leitungen — Fluss**")
                df = pd.DataFrame([
                    {"Leitung": l,
                     "Q [l/s]":  f"{q*1000:.2f}",
                     "v [m/s]":  f"{res.velocities.get(l,0):.3f}",
                     "dh [m]":   f"{res.head_losses.get(l,0):.3f}"}
                    for l, q in sorted(res.flows.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            if st.session_state.coords:
                st.markdown("**Karte — Versorgungsdruck**")
                fmap = build_geo_map(st.session_state.network,
                                     st.session_state.coords, domain,
                                     result=st.session_state.result)
                st_folium(fmap, use_container_width=True, height=420,
                          returned_objects=[])

            st.markdown("**Druckprofil**")
            fig, ax = plt.subplots(figsize=(9, 3))
            ns = sorted(res.pressures)
            ps = [res.pressures[n] for n in ns]
            cs = ["#27AE60" if n not in res.pressure_violations else "#E74C3C" for n in ns]
            ax.bar(ns, ps, color=cs, zorder=3)
            ax.axhline(2.0, color="orange", linestyle="--", linewidth=1.2, label="Min. 2 bar")
            ax.axhline(8.0, color="red",    linestyle="--", linewidth=1.2, label="Max. 8 bar")
            ax.set_ylabel("Druck [bar]")
            ax.set_title("Versorgungsdruck je Knoten (DVGW W 303)")
            ax.legend(); ax.grid(axis="y", alpha=0.4)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Gas ───────────────────────────────────────────────────────────────
        elif kind == "gas":
            res = st.session_state.result[1]
            st.subheader(f"Gasnetz ({res.pressure_level}) — Ergebnisse")

            ok = len(res.pressure_violations) == 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Iterationen", res.iterations)
            c2.metric("Konvergenz",  "✓ ja" if res.converged else "✗ nein")
            c3.metric("DVGW G 600",
                      "✅ OK" if ok else f"⚠ {len(res.pressure_violations)} Verletzung(en)")
            c4.metric("Min. Druck",  f"{min(res.pressures_bar.values()):.4f} bar")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Knoten — Druck**")
                df = pd.DataFrame([
                    {"Knoten": n,
                     "p [bar]":  f"{res.pressures_bar[n]:.4f}",
                     "p [mbar]": f"{res.pressures[n]/100:.1f}",
                     "Status": "✓" if n not in res.pressure_violations else "⚠"}
                    for n in sorted(res.pressures)
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            with col_b:
                st.markdown("**Leitungen — Fluss**")
                df = pd.DataFrame([
                    {"Leitung": l,
                     "Q [Nm³/h]": f"{q:.1f}",
                     "v [m/s]":   f"{res.velocities.get(l,0):.3f}",
                     "dp [mbar]": f"{res.pressure_drops.get(l,0):.3f}"}
                    for l, q in sorted(res.flows_norm.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            if st.session_state.coords:
                st.markdown("**Karte — Versorgungsdruck**")
                fmap = build_geo_map(st.session_state.network,
                                     st.session_state.coords, domain,
                                     result=st.session_state.result)
                st_folium(fmap, use_container_width=True, height=420,
                          returned_objects=[])

            st.markdown("**Druckprofil**")
            fig, ax = plt.subplots(figsize=(9, 3))
            ns = sorted(res.pressures_bar)
            ps = [res.pressures_bar[n] for n in ns]
            cs = ["#27AE60" if n not in res.pressure_violations else "#E74C3C" for n in ns]
            ax.bar(ns, ps, color=cs, zorder=3)
            ax.set_ylabel("Druck [bar]"); ax.grid(axis="y", alpha=0.4)
            ax.set_title(f"Versorgungsdruck je Knoten (DVGW G 600 — {res.pressure_level})")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            net = st.session_state.network
            lp  = net.calc_linepack(res)
            if lp:
                total = sum(l.volume_norm_m3 for l in lp)
                st.metric("Linepack gesamt", f"{total:.1f} Nm³",
                          help="Gasinhalt der Leitungen bei Betriebsdruck")

        # ── Fernwärme ─────────────────────────────────────────────────────────
        elif kind == "heating":
            hyd, therm = st.session_state.result[1], st.session_state.result[2]
            st.subheader("Fernwärmenetz — Ergebnisse")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Iterationen",    hyd.iterations)
            c2.metric("Konvergenz",     "✓ ja" if hyd.converged else "✗ nein")
            c3.metric("Wärmeverluste",  f"{sum(therm.heat_losses.values())/1000:.1f} kW")
            c4.metric("Min. Vorlauf",   f"{min(therm.temperatures.values()):.1f} °C")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Hydraulik — Massenstrom**")
                df = pd.DataFrame([
                    {"Leitung": l, "ṁ [kg/s]": f"{m:.4f}",
                     "Regime": hyd.flow_regimes.get(l, "-")}
                    for l, m in sorted(hyd.mass_flows.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            with col_b:
                st.markdown("**Thermik — Temperatur**")
                df = pd.DataFrame([
                    {"Knoten": n, "Vorlauf [°C]": f"{t:.2f}",
                     "Verlust [W]": f"{therm.heat_losses.get(n, 0):.0f}"}
                    for n, t in sorted(therm.temperatures.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            if st.session_state.coords:
                st.markdown("**Karte — Vorlauftemperaturen**")
                fmap = build_geo_map(st.session_state.network,
                                     st.session_state.coords, domain,
                                     result=st.session_state.result)
                st_folium(fmap, use_container_width=True, height=420,
                          returned_objects=[])

            st.markdown("**Temperaturprofil**")
            try:
                net = st.session_state.network
                fig = plot_temperature_profile(net, therm,
                                               title="Vorlauftemperaturen")
                st.pyplot(fig); plt.close(fig)
            except Exception:
                fig, ax = plt.subplots(figsize=(9, 3))
                ns = sorted(therm.temperatures)
                ax.bar(ns, [therm.temperatures[n] for n in ns], color="#2E86AB")
                ax.set_ylabel("Temperatur [°C]")
                ax.set_title("Vorlauftemperatur je Knoten")
                ax.grid(axis="y", alpha=0.4)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Strom AC ──────────────────────────────────────────────────────────
        elif kind == "ac":
            res = st.session_state.result[1]
            st.subheader("Wechselstromnetz — Ergebnisse")

            voltages = res["voltages"]
            active   = res["active_power"]
            reactive = res["reactive_power"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Knoten", len(voltages))
            c2.metric("Max |U|", f"{max(abs(v) for v in voltages.values()):.3f} V")
            c3.metric("P gesamt", f"{sum(active.values()):.2f} W")

            import cmath
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Knotenspannungen**")
                df = [{"Knoten": n,
                       "|U| [V]": f"{abs(u):.4f}",
                       "Re [V]":  f"{u.real:.4f}",
                       "Im [V]":  f"{u.imag:.4f}",
                       "φ [°]":   f"{cmath.phase(u)*180/3.14159:.2f}" if abs(u) > 1e-9 else "—"}
                      for n, u in sorted(voltages.items())]
                st.dataframe(pd.DataFrame(df), hide_index=True, use_container_width=True)

            with col_b:
                st.markdown("**Leistungen je Knoten**")
                df2 = [{"Knoten": n,
                        "P [W]":   f"{active.get(n, 0):.3f}",
                        "Q [var]": f"{reactive.get(n, 0):.3f}",
                        "|S| [VA]": f"{abs(active.get(n,0) + 1j*reactive.get(n,0)):.3f}"}
                       for n in sorted(voltages)]
                st.dataframe(pd.DataFrame(df2), hide_index=True, use_container_width=True)

            st.markdown("**Spannungsbetrag je Knoten**")
            fig, ax = plt.subplots(figsize=(9, 3))
            ns = sorted(voltages)
            us = [abs(voltages[n]) for n in ns]
            ax.bar(ns, us, color="#2E86AB")
            ax.set_ylabel("|U| [V]"); ax.grid(axis="y", alpha=0.4)
            ax.set_title("Knotenspannungen (Betrag)")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Strom DC ──────────────────────────────────────────────────────────
        elif kind == "dc":
            res = st.session_state.result[1]
            st.subheader("Gleichstromnetz — Ergebnisse")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Knotenpotenziale [V]**")
                df = pd.DataFrame([
                    {"Knoten": n, "Spannung [V]": f"{u:.4f}"}
                    for n, u in sorted(res.potentials.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            with c2:
                st.markdown("**Leitungsströme [A]**")
                df = pd.DataFrame([
                    {"Leitung": l, "Strom [A]": f"{i:.4f}"}
                    for l, i in sorted(res.flows.items())
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)

            st.markdown("**Spannungsprofil**")
            fig, ax = plt.subplots(figsize=(9, 3))
            ns = sorted(res.potentials)
            ax.bar(ns, [res.potentials[n] for n in ns], color="#2E86AB")
            ax.set_ylabel("Spannung [V]"); ax.grid(axis="y", alpha=0.4)
            ax.set_title("Knotenpotenziale")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KI-ASSISTENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ki:
    st.subheader("🤖 KI-Assistent")
    st.caption("Fragen zu Berechnungsergebnissen, DVGW-Normen und ATN-Methodik. "
               "Claude API-Anbindung folgt im nächsten Schritt.")

    # Demo-Antworten für häufige Fragen
    DEMO = {
        "verletzung": (
            "Eine **DVGW-Druckverletzung** bedeutet, dass der Versorgungsdruck an einem "
            "Knoten unter dem Mindestwert liegt (Wasser: < 2 bar, Gas ND: < 17 mbar).\n\n"
            "**Mögliche Ursachen:** Zu kleiner Rohrdurchmesser, zu hohe Entnahme, "
            "zu große Leitungslänge.\n\n"
            "**Empfehlung:** Rohrdurchmesser der Zuleitung vergrößern oder zusätzliche "
            "Einspeisung (Pumpe / Druckregler) einplanen."
        ),
        "weymouth": (
            "Die **Weymouth-Formel** berechnet den Druckabfall in Hochdruck-Gasleitungen:\n\n"
            "```\nΔ(p²) = λ_W · L/D · ρ_N · p_N · Q_N²/A²\n"
            "λ_W = 0.009407 · D^(-1/3)  (Weymouth-Reibungszahl)\n```\n\n"
            "Sie ist eine Variante der Darcy-Weisbach-Formel mit einer empirischen "
            "Reibungszahl speziell für kompressibles Gas in HD-Leitungen."
        ),
        "hazen": (
            "Der **Hazen-Williams-Beiwert C** beschreibt die hydraulische Güte einer Wasserleitung:\n\n"
            "| Werkstoff | C-Wert |\n|---|---|\n"
            "| PE/PVC (neu) | 140 |\n| Stahl (neu) | 130 |\n"
            "| Stahl (alt) | 100–120 |\n| Gusseisen (alt) | 80–100 |\n\n"
            "Je höher C, desto geringer der Druckverlust."
        ),
        "dvgw": (
            "Relevante **DVGW-Normen** im ATN-Dashboard:\n\n"
            "- **W 303** — Wasserversorgung: Versorgungsdruck 2–8 bar\n"
            "- **W 400** — Planung und Bau von Wasserleitungen\n"
            "- **G 462** — Gasleitungen aus Stahl bis 16 bar\n"
            "- **G 600** — Technische Regeln Gasinstallation (ND/MD)\n\n"
            "Mindestdruck ND-Gas: 17 mbar (G 600, Abschnitt 8)."
        ),
        "atn": (
            "Die **Allgemeine Theorie der Technischen Netze (ATN)** von Prof. Strelow "
            "beschreibt Strom-, Gas-, Wärme- und Wassernetze mit denselben drei Gleichungen:\n\n"
            "```\nKnotensatz:        K · I + I_ext = 0\n"
            "Maschensatz:       ΔU = Kᵀ · U\n"
            "Widerstandsgesetz: I = −R⁻¹ · ΔU\n```\n\n"
            "Das macht domänenübergreifende Sektorenkopplung in einem einheitlichen "
            "Matrizenkalkül möglich."
        ),
    }

    def get_response(prompt: str) -> str:
        pl = prompt.lower()
        for key, ans in DEMO.items():
            if key in pl:
                return ans

        # Kontext-bewusste Antwort wenn Ergebnis vorhanden
        if st.session_state.result and ("ergebnis" in pl or "resultat" in pl
                                         or "berechnung" in pl or "status" in pl):
            kind = st.session_state.result[0]
            if kind == "water":
                res = st.session_state.result[1]
                ok  = len(res.pressure_violations) == 0
                return (
                    f"**Ergebnis des Wassernetzes:**\n\n"
                    f"- Konvergenz nach {res.iterations} Iterationen: "
                    f"{'✓ ja' if res.converged else '✗ nein'}\n"
                    f"- DVGW W 303: {'✅ Alle Knoten OK' if ok else f'⚠ {len(res.pressure_violations)} Verletzung(en)'}\n"
                    f"- Minimaler Versorgungsdruck: {min(res.pressures.values())/1e5:.3f} bar\n\n"
                    f"{'Alle Drücke liegen im Bereich 2–8 bar.' if ok else 'Empfehlung: Druckverletzungen in Tab **Ergebnisse** prüfen.'}"
                )
            elif kind == "gas":
                res = st.session_state.result[1]
                ok  = len(res.pressure_violations) == 0
                return (
                    f"**Ergebnis des Gasnetzes ({res.pressure_level}):**\n\n"
                    f"- Konvergenz nach {res.iterations} Iterationen: "
                    f"{'✓ ja' if res.converged else '✗ nein'}\n"
                    f"- DVGW G 600: {'✅ Alle Knoten OK' if ok else f'⚠ {len(res.pressure_violations)} Verletzung(en)'}\n"
                    f"- Minimaldruck: {min(res.pressures_bar.values()):.4f} bar"
                )

        return (
            "Ich bin der **ATN-Assistent** und helfe bei:\n\n"
            "- 📋 Erklärung von Berechnungsergebnissen (fragen Sie: *'Erkläre das Ergebnis'*)\n"
            "- 📐 DVGW-Normen (fragen Sie: *'Was sagt DVGW?'*, *'Was bedeutet Verletzung?'*)\n"
            "- 🔬 ATN-Methodik (fragen Sie: *'Was ist ATN?'*, *'Was ist Weymouth?'*)\n\n"
            "---\n"
            "⚙️ **Vollständige Claude API-Anbindung** (Anthropic) ist der nächste Entwicklungsschritt — "
            "dann beantworte ich beliebige Fachfragen, ergänze fehlende Parameter aus Normen "
            "und generiere automatisch Berechnungsberichte."
        )

    # Chat-Verlauf anzeigen
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Eingabe
    if prompt := st.chat_input("Frage zum Netz, Ergebnis oder Normwert ..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = get_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
