"""
Beispiel 6: Gasnetz — Druckberechnung fuer alle DVGW-Druckstufen
=================================================================

Demonstriert die Anwendung des ATN-Frameworks auf Gasversorgungsnetze.
Alle drei Druckstufen (ND, MD, HD) werden mit je einem Beispielnetz gezeigt.

Teil A: Niederdruck-Verteilnetz (ND, <= 100 mbar)
        Typisch: Hausanschlussgebiet, Vorstadtnetz
        Formel: Renouard-ND, Potenzial = p [Pa]

Teil B: Mitteldruck-Versorgungsnetz (MD, <= 1 bar)
        Typisch: Stadtverteilnetz, Gewerbegebiet
        Formel: Renouard-MD, Potenzial = p^2 [Pa^2], linearisiert

Teil C: Hochdruck-Transportnetz (HD, > 1 bar)
        Typisch: Fernleitungsstrang, interkommunale Verbindung
        Formel: Weymouth, Potenzial = p^2 [Pa^2], linearisiert

Teil D: Sektorenkopplung — Gasnetz trifft CouplingModel
        Gasverfuegbarkeit aus Netzberechnung als Input fuer KWK-Optimierung
"""

import sys
sys.path.insert(0, "..")
from atn.networks.gas import GasNetwork, PressureLevel
from atn.coupling.model import CouplingModel


# ============================================================
# TEIL A: NIEDERDRUCK-VERTEILNETZ
# ============================================================
print("=" * 65)
print("TEIL A: NIEDERDRUCK-VERTEILNETZ (ND, <= 100 mbar)")
print("Formel: Renouard-ND  |  Potenzial: p [Pa]")
print("=" * 65)

# Typisches Wohngebiets-ND-Netz:
#   [DR] --G1-- [K1] --G3-- [K3]
#          |
#          G2
#          |
#         [K2]
# DR = Druckregler (Eingang aus MD-Netz, Ausgang 23 mbar)

nd = GasNetwork("ND-Wohngebiet", pressure_level=PressureLevel.ND)

nd.add_feed("DR", pressure_bar=0.023)               # 23 mbar Einspeisedruck
nd.add_pipe("DR", "K1", length=80,  diameter=0.063, label="G1")   # PE, DN 63
nd.add_pipe("DR", "K2", length=60,  diameter=0.05,  label="G2")   # PE, DN 50
nd.add_pipe("K1", "K3", length=100, diameter=0.05,  label="G3")   # PE, DN 50

nd.add_offtake("K1", flow_m3h=8.0,  min_pressure_bar=0.017)   # 8 Nm3/h
nd.add_offtake("K2", flow_m3h=6.0,  min_pressure_bar=0.017)
nd.add_offtake("K3", flow_m3h=4.0,  min_pressure_bar=0.017)

nd_result = nd.solve_hydraulic()
print(f"\nKonvergenz: {'ja' if nd_result.converged else 'NEIN'} "
      f"({nd_result.iterations} Iterationen)")

print("\nDrucke:")
print(f"  {'Knoten':<8} {'p [mbar]':>12}  Status")
print("  " + "-" * 30)
for node, p_pa in sorted(nd_result.pressures.items()):
    p_mbar = p_pa / 100.0
    ok = "OK" if node not in nd_result.pressure_violations else "! VERLETZUNG"
    print(f"  {node:<8} {p_mbar:>12.2f}  {ok}")

print("\nRohrleitungen:")
print(f"  {'Leit.':<8} {'Q [Nm3/h]':>12} {'v [m/s]':>10} {'dp [mbar]':>12}")
print("  " + "-" * 46)
for label in sorted(nd_result.flows_norm):
    q  = nd_result.flows_norm[label]
    v  = nd_result.velocities.get(label, 0)
    dp = nd_result.pressure_drops.get(label, 0)
    print(f"  {label:<8} {q:>12.2f} {v:>10.3f} {dp:>12.3f}")

dvgw_nd = nd.check_dvgw(nd_result)
print(f"\n{dvgw_nd['summary']}")


# ============================================================
# TEIL B: MITTELDRUCK-VERSORGUNGSNETZ
# ============================================================
print("\n" + "=" * 65)
print("TEIL B: MITTELDRUCK-VERSORGUNGSNETZ (MD, <= 1 bar)")
print("Formel: Renouard-MD  |  Potenzial: p^2 [Pa^2], linearisiert")
print("=" * 65)

# Stadtverteilnetz mit Druckreglerstation (UeST) als Einspeisung:
#
#   [UeST] --G1-- [K1] --G3-- [K3]
#     |                         |
#     G2                        G5
#     |                         |
#   [K2] ----G4-----------  [K4]
#
# UeST = Uebergabestation aus HD-Netz, Ausgangsdruck 0.5 bar

md = GasNetwork("MD-Stadtverteilnetz", pressure_level=PressureLevel.MD)

md.add_feed("UeST", pressure_bar=0.5)    # 500 mbar Einspeisedruck

md.add_pipe("UeST", "K1", length=400, diameter=0.15, label="G1")
md.add_pipe("UeST", "K2", length=300, diameter=0.10, label="G2")
md.add_pipe("K1",   "K3", length=350, diameter=0.10, label="G3")
md.add_pipe("K2",   "K4", length=450, diameter=0.08, label="G4")
md.add_pipe("K3",   "K4", length=200, diameter=0.08, label="G5")  # Ringschluss

md.add_offtake("K1", flow_m3h=120.0, min_pressure_bar=0.1)   # Gewerbe
md.add_offtake("K2", flow_m3h=80.0,  min_pressure_bar=0.1)   # Wohnen
md.add_offtake("K3", flow_m3h=60.0,  min_pressure_bar=0.1)   # Wohnen
md.add_offtake("K4", flow_m3h=100.0, min_pressure_bar=0.1)   # Industrie

md_result = md.solve_hydraulic(max_iter=200, tol=1e-6)
print(f"\nKonvergenz: {'ja' if md_result.converged else 'NEIN'} "
      f"({md_result.iterations} Iterationen)")

print("\nDrucke:")
print(f"  {'Knoten':<8} {'p [bar]':>10} {'p [mbar]':>12}  Status")
print("  " + "-" * 38)
for node, p_pa in sorted(md_result.pressures.items()):
    ok = "OK" if node not in md_result.pressure_violations else "! VERLETZUNG"
    print(f"  {node:<8} {p_pa/1e5:>10.4f} {p_pa/100:>12.1f}  {ok}")

print("\nRohrleitungen:")
print(f"  {'Leit.':<8} {'Q [Nm3/h]':>12} {'v [m/s]':>10} {'dp [mbar]':>12}")
print("  " + "-" * 46)
for label in sorted(md_result.flows_norm):
    q  = md_result.flows_norm[label]
    v  = md_result.velocities.get(label, 0)
    dp = md_result.pressure_drops.get(label, 0)
    print(f"  {label:<8} {q:>12.1f} {v:>10.3f} {dp:>12.3f}")

dvgw_md = md.check_dvgw(md_result)
print(f"\n{dvgw_md['summary']}")

# Linienspeicher (Linepack)
linepack = md.calc_linepack(md_result)
total_lp = sum(lp.volume_norm_m3 for lp in linepack)
print(f"\nLinepack (Gasinhalt der Leitungen): {total_lp:.1f} Nm3 gesamt")
for lp in linepack[:3]:
    print(f"  {lp.pipe_label}: {lp.volume_norm_m3:.1f} Nm3 "
          f"bei {lp.pressure_pa/1e5:.3f} bar")


# ============================================================
# TEIL C: HOCHDRUCK-TRANSPORTNETZ
# ============================================================
print("\n" + "=" * 65)
print("TEIL C: HOCHDRUCK-TRANSPORTNETZ (HD, > 1 bar)")
print("Formel: Weymouth  |  Potenzial: p^2 [Pa^2], linearisiert")
print("=" * 65)

# Interkommunales Transportnetz:
#   [EIN] --G1-- [K1] --G2-- [K2] --G3-- [K3]
# EIN = Einspeisepunkt (z.B. Fernleitungsanbindung, 40 bar)

hd = GasNetwork("HD-Transportleitung", pressure_level=PressureLevel.HD)
hd.set_gas_properties(rho_rel=0.625, z_factor=0.94, temperature_k=283.15)

hd.add_feed("EIN", pressure_bar=40.0)

hd.add_pipe("EIN", "K1", length=5000, diameter=0.30, label="G1")   # DN 300
hd.add_pipe("K1",  "K2", length=4000, diameter=0.25, label="G2")   # DN 250
hd.add_pipe("K2",  "K3", length=3000, diameter=0.20, label="G3")   # DN 200

hd.add_offtake("K1", flow_m3h=2000.0, min_pressure_bar=5.0)
hd.add_offtake("K2", flow_m3h=3000.0, min_pressure_bar=5.0)
hd.add_offtake("K3", flow_m3h=1500.0, min_pressure_bar=5.0)

hd_result = hd.solve_hydraulic()
print(f"\nKonvergenz: {'ja' if hd_result.converged else 'NEIN'} "
      f"({hd_result.iterations} Iterationen)")

print("\nDrucke:")
print(f"  {'Knoten':<8} {'p [bar]':>12}  Status")
print("  " + "-" * 30)
for node, p_pa in sorted(hd_result.pressures.items()):
    ok = "OK" if node not in hd_result.pressure_violations else "! VERLETZUNG"
    print(f"  {node:<8} {p_pa/1e5:>12.2f}  {ok}")

print("\nRohrleitungen:")
print(f"  {'Leit.':<8} {'Q [Nm3/h]':>12} {'v [m/s]':>10} {'dp [bar]':>10}")
print("  " + "-" * 44)
for label in sorted(hd_result.flows_norm):
    q  = hd_result.flows_norm[label]
    v  = hd_result.velocities.get(label, 0)
    dp = hd_result.pressure_drops.get(label, 0) / 10.0  # mbar -> bar
    print(f"  {label:<8} {q:>12.0f} {v:>10.2f} {dp:>10.4f}")

dvgw_hd = hd.check_dvgw(hd_result)
print(f"\n{dvgw_hd['summary']}")


# ============================================================
# TEIL D: SEKTORENKOPPLUNG — GASNETZ + COUPLING MODEL
# ============================================================
print("\n" + "=" * 65)
print("TEIL D: SEKTORENKOPPLUNG — GASNETZ + KWK-OPTIMIERUNG")
print("Gasverfuegbarkeit aus MD-Netzberechnung als KWK-Input")
print("=" * 65)

# Aus dem MD-Netz: verfuegbarer Gasstrom am Industrieknoten K4
q_gas_verfuegbar = abs(md_result.flows_norm.get("G4", 0))  # Nm3/h
# Heizwert Erdgas H (DVGW G 260): 10.0 kWh/Nm3
HHV = 10.0  # kWh/Nm3
P_gas_verfuegbar = q_gas_verfuegbar * HHV   # kW (thermisch)

print(f"\nVerfuegbare Gasleistung an K4: {q_gas_verfuegbar:.1f} Nm3/h "
      f"= {P_gas_verfuegbar:.0f} kW")

# KWK-Anlage: BHKW + Spitzenlastkessel (wie Beispiel 03, aber Gas-gekoppelt)
# Parameter so gewaehlt, dass Gasbedarf <= verfuegbarer Gasstrom aus Netz
P_el_bedarf  = 80.0    # kW elektr. Bedarf Industrieknoten
Q_hz_bedarf  = 250.0   # kW Waermebedarf
eta_el       = 0.38    # elektr. Wirkungsgrad BHKW
eta_hz       = 0.50    # therm. Wirkungsgrad BHKW
eta_kessel   = 0.92    # Kesselwirkungsgrad

kwk = CouplingModel("KWK-Anlage an Knoten K4")

kwk.add_variable("P_BHKW",    "kW", lower=20,  upper=200,
                 description="BHKW elektr. Leistung")
kwk.add_variable("Q_Kessel",  "kW", lower=0,   upper=200,
                 description="Spitzenlastkessel Waerme")
kwk.add_variable("P_Netz",    "kW", lower=-100, upper=100,
                 description="Strombezug aus Netz (pos.) / Einspeisung (neg.)")
kwk.add_variable("V_Gas",     "kW", lower=0,   upper=P_gas_verfuegbar,
                 description="Gesamtgasleistung (aus Netzberechnung begrenzt)")

# Strom-Bilanz: BHKW + Netzbezug = Bedarf
kwk.add_balance("Elektrizitaet",
                {"P_BHKW": 1.0, "P_Netz": 1.0},
                rhs=P_el_bedarf)

# Waerme-Bilanz: BHKW-Waerme + Kessel = Bedarf
kwk.add_balance("Waerme",
                {"P_BHKW": eta_hz / eta_el, "Q_Kessel": 1.0},
                rhs=Q_hz_bedarf)

# Gas-Bilanz: BHKW-Gas + Kessel-Gas = Verfuegbar
kwk.add_balance("Erdgas",
                {"P_BHKW": 1.0 / eta_el, "Q_Kessel": 1.0 / eta_kessel,
                 "V_Gas": -1.0},
                rhs=0.0)

analysis = kwk.analyze()
print(f"\nKopplungsmodell: Rang={analysis.rank}, Freiheitsgrad={analysis.dof}")
print(f"Entscheidungsgroessen: {analysis.decision_variables}")

# Kostenoptimierung: minimiere Betriebskosten
costs = {
    "P_Netz":   0.25,     # EUR/kWh Strombezugspreis
    "V_Gas":    0.05,     # EUR/kWh Gaspreis
    "Q_Kessel": 0.0,      # indirekt ueber V_Gas erfasst
    "P_BHKW":   0.0,
}

opt = kwk.optimize(costs, minimize_obj=True)
print(f"\nKostenoptimierung (min. Betriebskosten):")
print(f"  Status: {opt.status}")
if opt.status == 'optimal':
    print(f"  Betriebskosten: {opt.objective_value:.2f} EUR/h")
    for k, v in opt.solution.items():
        tag = " [E]" if k in opt.decision_values else " [F]"
        print(f"    {k:<20}: {v:8.1f} kW{tag}")
