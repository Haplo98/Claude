"""
Beispiel 4: Visualisierungsmodul
=================================
Demonstriert alle Visualisierungsfunktionen des ATN-Frameworks.

Erzeugte Abbildungen (in examples/figures/):
  01_network.png          - DC-Netz Topologie (Strelow Band 3)
  02_coupling_matrix.png  - Kopplungsmatrix Drei Heizwerke
  03_decision_space.png   - Entscheidungsraum Hexagon (Strelow 2024, Abb. 2)
  04_pressure.png         - Druckprofil Fernwärmenetz
  05_temperature.png      - Stationäres Temperaturprofil
  06_dynamic.png          - Dynamische Wärmeausbreitung
  07_validation.png       - Messdaten-Validierung mit Ausreißer
  08_dashboard.png        - Kombinierter Fernwärme-Dashboard
"""
import sys
import os

# Matplotlib auf nicht-interaktives Backend setzen (kein Display nötig)
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, "..")
import numpy as np

from atn.networks.electrical import DCNetwork
from atn.networks.district_heating import DistrictHeatingNetwork
from atn.coupling.model import CouplingModel
from atn.validation.measurements import SystemValidator
from atn.visualization.plots import (
    plot_network,
    plot_decision_space,
    plot_coupling_matrix,
    plot_pressure_profile,
    plot_temperature_profile,
    plot_dynamic_temperatures,
    plot_validation_residuals,
    plot_district_heating_dashboard,
)

# Output-Verzeichnis
os.makedirs("figures", exist_ok=True)


# =========================================================================
# 1. DC-NETZ TOPOLOGIE (Strelow Band 3, Abb. 1-2)
# =========================================================================
print("=" * 55)
print("1. DC-Netz Topologie")
print("=" * 55)

dc_net = DCNetwork("Beispiel Band 3")
dc_net.add_edge("K1", "K2", resistance=2.0, label="L1")
dc_net.add_edge("K2", "K3", resistance=3.0, label="L2")
dc_net.add_edge("K3", "K4", resistance=1.0, label="L3")
dc_net.add_edge("K4", "K5", resistance=2.0, label="L4")
dc_net.add_edge("K5", "K1", resistance=1.5, label="L5")
dc_net.add_edge("K1", "K4", resistance=4.0, label="L6")
dc_net.add_edge("K2", "K4", resistance=2.5, label="L7")
dc_net.set_external_flow("K1",  10.0)
dc_net.set_external_flow("K3",  -4.0)
dc_net.set_external_flow("K5",  -6.0)

dc_result = dc_net.solve(reference_node="K5")
print(f"Bilanzkontrolle: {dc_result.balance_error:.2e} A")

# Manuelle Positionen (angelehnt an Abb. 1 in Band 3)
pos_dc = {
    "K1": (-1.0,  0.5),
    "K2": ( 0.0,  1.0),
    "K3": ( 1.0,  0.5),
    "K4": ( 0.5, -0.5),
    "K5": (-0.5, -0.5),
}

fig1 = plot_network(
    dc_net, dc_result, pos=pos_dc,
    potential_label="U", potential_unit="V",
    flow_label="I", flow_unit="A",
    title="Gleichstromnetz — Strelow Band 3 (2017)\n5 Knoten, 7 Leitungen, 2 Maschen",
)
fig1.savefig("figures/01_network.png", bbox_inches='tight', dpi=120)
print("  -> figures/01_network.png gespeichert")


# =========================================================================
# 2. KOPPLUNGSMATRIX: DREI HEIZWERKE (Strelow 2024, Abb. 1)
# =========================================================================
print("\n" + "=" * 55)
print("2. Kopplungsmatrix")
print("=" * 55)

Q_bedarf = 800.0
model_hw = CouplingModel("Drei Heizwerke")
model_hw.add_variable("Q1_Heizwerk1", "kW", lower=0, upper=500)
model_hw.add_variable("Q2_Heizwerk2", "kW", lower=0, upper=500)
model_hw.add_variable("Q3_Heizwerk3", "kW", lower=0, upper=500)
model_hw.add_balance(
    "Waerme",
    {"Q1_Heizwerk1": 1.0, "Q2_Heizwerk2": 1.0, "Q3_Heizwerk3": 1.0},
    rhs=Q_bedarf,
)

analysis_hw = model_hw.analyze()
print(f"  Rang={analysis_hw.rank}, dof={analysis_hw.dof}")
print(f"  Entscheidungsgrossen: {analysis_hw.decision_variables}")
print(f"  Folgegrossen:         {analysis_hw.dependent_variables}")

fig2 = plot_coupling_matrix(
    model_hw,
    title="Kopplungsmatrix — Drei Heizwerke (Strelow 2024)",
    figsize=(8, 4),
)
fig2.savefig("figures/02_coupling_matrix.png", bbox_inches='tight', dpi=120)
print("  -> figures/02_coupling_matrix.png gespeichert")


# =========================================================================
# 3. ENTSCHEIDUNGSRAUM (Strelow 2024, Abb. 2 — Hexagon)
# =========================================================================
print("\n" + "=" * 55)
print("3. Entscheidungsraum")
print("=" * 55)

# Optimalen Punkt berechnen
costs = {"Q1_Heizwerk1": 0.08, "Q2_Heizwerk2": 0.06, "Q3_Heizwerk3": 0.04}
opt = model_hw.optimize(costs)
print(f"  Optimum: {opt.solution}  Kosten: {opt.objective_value:.2f} EUR/h")

opt_point = {
    "Q2_Heizwerk2": opt.solution["Q2_Heizwerk2"],
    "Q3_Heizwerk3": opt.solution["Q3_Heizwerk3"],
}
test_pts = [
    {"Q2_Heizwerk2": 200.0, "Q3_Heizwerk3": 150.0},  # zulässig
    {"Q2_Heizwerk2": 400.0, "Q3_Heizwerk3": 300.0},  # unzulässig (Q1 < 0)
    {"Q2_Heizwerk2": 100.0, "Q3_Heizwerk3": 100.0},  # zulässig
]

fig3 = plot_decision_space(
    model_hw,
    x_var="Q2_Heizwerk2",
    y_var="Q3_Heizwerk3",
    n_samples=60,
    optimal_point=opt_point,
    test_points=test_pts,
    title="Entscheidungsraum — Drei Heizwerke\n(Strelow 2024, Abb. 2)",
)
fig3.savefig("figures/03_decision_space.png", bbox_inches='tight', dpi=120)
print("  -> figures/03_decision_space.png gespeichert")


# =========================================================================
# 4-6. FERNWÄRMENETZ: DRUCK + TEMPERATUR + DYNAMIK
# =========================================================================
print("\n" + "=" * 55)
print("4-6. Fernwärmenetz")
print("=" * 55)

dh = DistrictHeatingNetwork("Strangnetz Beispiel")
dh.add_heat_source("HW", supply_temp=85.0, mass_flow=20.0)
dh.add_consumer("V1", heat_load=500_000, return_temp=55.0)
dh.add_consumer("V2", heat_load=800_000, return_temp=55.0)
dh.add_consumer("V3", heat_load=400_000, return_temp=55.0)
dh.add_pipe("HW", "V1", length=200, diameter=0.150, label="P1")
dh.add_pipe("V1", "V2", length=150, diameter=0.100, label="P2")
dh.add_pipe("V2", "V3", length=100, diameter=0.080, label="P3")

hyd = dh.solve_hydraulic(max_iter=30, tol=1e-5)
print(f"  Hydraulik: {hyd.iterations} Iterationen, konvergiert={hyd.converged}")

therm = dh.solve_thermal_stationary(hyd)
print(f"  Temperaturen: {', '.join(f'{n}={T:.1f}C' for n, T in therm.temperatures.items())}")

# Dynamik: Kaltstart → Wärmewelle
T_init = {node: 70.0 for node in dh._nodes}

def source_profile(t):
    return 85.0 if t >= 600 else 70.0

dynamic = dh.solve_thermal_dynamic(
    hyd,
    T_initial=T_init,
    dt=60.0,
    t_end=3600.0,
    source_temp_profile={"HW": source_profile},
)

node_order = ["HW", "V1", "V2", "V3"]

# Plot 4: Druckprofil
fig4 = plot_pressure_profile(
    dh, hyd, node_order=node_order,
    title="Druckprofil — Strangnetz (Strelow & Kouka 2025)",
)
fig4.savefig("figures/04_pressure.png", bbox_inches='tight', dpi=120)
print("  -> figures/04_pressure.png gespeichert")

# Plot 5: Stationäres Temperaturprofil
fig5 = plot_temperature_profile(
    dh, therm, node_order=node_order,
    title="Stationäres Temperaturprofil — Strangnetz",
)
fig5.savefig("figures/05_temperature.png", bbox_inches='tight', dpi=120)
print("  -> figures/05_temperature.png gespeichert")

# Plot 6: Dynamik
fig6 = plot_dynamic_temperatures(
    dynamic,
    nodes=["HW", "V1", "V2", "V3"],
    time_unit='min',
    title="Wärmeausbreitung — Lastwechsel HW: 70->85 C (nach 10 min)",
)
fig6.savefig("figures/06_dynamic.png", bbox_inches='tight', dpi=120)
print("  -> figures/06_dynamic.png gespeichert")


# =========================================================================
# 7. MESSDATEN-VALIDIERUNG (Strelow & Dawitz 2020)
# =========================================================================
print("\n" + "=" * 55)
print("7. Messdaten-Validierung")
print("=" * 55)

# Bilanzmatrix: zwei Mischknoten in einem Wärmenetz
# x = [Q_in1, Q_in2, Q_out1, Q_out2, Q_out3]
# Knoten A: Q_in1 - Q_out1 - Q_out2 = 0
# Knoten B: Q_in2 - Q_out3 = 0
K_val = np.array([
    [ 1,  0, -1, -1,  0],
    [ 0,  1,  0,  0, -1],
])
b_val = np.zeros(2)

validator = SystemValidator(K_val, b_val)

# Messungen (ohne Sensor 4 — nicht gemessen)
# Wahre Werte: 100, 60, 40, 60, 60
# Messwerte: leicht fehlerbehaftet, Sensor 2 hat groben Fehler
true_vals = [100.0, 60.0, 40.0, 60.0, 60.0]
noise     = [ 0.5,  15.0,  0.3,  0.4,  0.2]   # Sensor 2: grober Fehler

validator.add_measurement(0, true_vals[0] + noise[0], weight=10.0, unit="kW", sensor_id="Q_in1")
validator.add_measurement(1, true_vals[1] + noise[1], weight=10.0, unit="kW", sensor_id="Q_in2")
validator.add_measurement(2, true_vals[2] + noise[2], weight=10.0, unit="kW", sensor_id="Q_out1")
validator.add_measurement(3, true_vals[3] + noise[3], weight=10.0, unit="kW", sensor_id="Q_out2")
validator.add_measurement(4, true_vals[4] + noise[4], weight=10.0, unit="kW", sensor_id="Q_out3")

val_result = validator.validate()
print(f"  Validierungsstatus: {val_result.status}")
print(f"  RMS Korrektur: {val_result.rms_correction:.4f} kW")
print(f"  Max Korrektur: {val_result.max_correction:.4f} kW")

faulty = validator.detect_faulty_sensors(threshold_sigma=1.5)
if faulty:
    print(f"  Fehlerverdaechtige Sensoren: {[s.sensor_id for s in faulty]}")

fig7 = plot_validation_residuals(
    validator, val_result,
    title="Messdaten-Validierung — Wärmenetz (Strelow & Dawitz 2020)",
)
fig7.savefig("figures/07_validation.png", bbox_inches='tight', dpi=120)
print("  -> figures/07_validation.png gespeichert")


# =========================================================================
# 8. KOMBINIERTER DASHBOARD
# =========================================================================
print("\n" + "=" * 55)
print("8. Fernwärme-Dashboard")
print("=" * 55)

fig8 = plot_district_heating_dashboard(
    dh, hyd, thermal_result=therm, dynamic_result=dynamic,
    node_order=node_order,
    title="Fernwärme-Dashboard — Strangnetz (Strelow & Kouka 2025)",
)
fig8.savefig("figures/08_dashboard.png", bbox_inches='tight', dpi=120)
print("  -> figures/08_dashboard.png gespeichert")


# =========================================================================
# ZUSAMMENFASSUNG
# =========================================================================
print("\n" + "=" * 55)
print("ERGEBNIS: Alle Abbildungen in examples/figures/")
print("=" * 55)
figs = sorted(os.listdir("figures"))
for f in figs:
    path = os.path.join("figures", f)
    size_kb = os.path.getsize(path) // 1024
    print(f"  {f:<30} {size_kb:4d} kB")
