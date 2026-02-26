"""
Beispiel 2: Fernwärmenetz
==========================
Strangnetz mit einem Heizwerk und drei Verbrauchern.
Demonstriert hydraulische + stationäre + dynamische thermische Berechnung
(Strelow & Kouka, Band 35, 2025).

Topologie:
    HW (Heizwerk) ─── V1 ─── V2 ─── V3
                  Vorlauf DN150  DN100  DN80

Rücklauf (vereinfacht symmetrisch, hier nicht modelliert).
"""
import sys
sys.path.insert(0, "..")
import numpy as np
from atn.networks.district_heating import DistrictHeatingNetwork

net = DistrictHeatingNetwork("Strangnetz Beispiel")

# Heizwerk: 85°C Vorlauf, 20 kg/s Gesamtmassenstrom
net.add_heat_source("HW", supply_temp=85.0, mass_flow=20.0)

# Verbraucher
net.add_consumer("V1", heat_load=500_000, return_temp=55.0)   # 500 kW
net.add_consumer("V2", heat_load=800_000, return_temp=55.0)   # 800 kW
net.add_consumer("V3", heat_load=400_000, return_temp=55.0)   # 400 kW

# Rohrleitungen (Vorlauf)
net.add_pipe("HW", "V1", length=200, diameter=0.150, label="P1")
net.add_pipe("V1", "V2", length=150, diameter=0.100, label="P2")
net.add_pipe("V2", "V3", length=100, diameter=0.080, label="P3")

# ── Hydraulische Berechnung ──────────────────────────────────────────
print("=" * 55)
print("HYDRAULISCHE BERECHNUNG")
print("=" * 55)
hyd = net.solve_hydraulic(max_iter=30, tol=1e-5)

print(f"Konvergenz nach {hyd.iterations} Iterationen: {hyd.converged}")
print("\nDrücke [bar]:")
for node, p in hyd.pressures.items():
    print(f"  {node}: {p/1e5:.3f} bar")

print("\nMassenströme [kg/s]:")
for pipe, m in hyd.mass_flows.items():
    regime = hyd.flow_regimes.get(pipe, "?")
    print(f"  {pipe}: {m:.3f} kg/s  ({regime})")

# ── Stationäres Temperaturprofil ─────────────────────────────────────
print("\n" + "=" * 55)
print("STATIONÄRES TEMPERATURPROFIL")
print("=" * 55)
therm = net.solve_thermal_stationary(hyd)

print("\nVorlauftemperaturen [°C]:")
for node, T in therm.temperatures.items():
    print(f"  {node}: {T:.2f} °C")

print("\nWärmeverluste [kW]:")
total_loss = 0
for pipe, Q_loss in therm.heat_losses.items():
    print(f"  {pipe}: {Q_loss/1000:.2f} kW")
    total_loss += Q_loss
print(f"  Gesamt: {total_loss/1000:.2f} kW")

# ── Dynamisches Temperaturprofil ─────────────────────────────────────
print("\n" + "=" * 55)
print("DYNAMISCHES TEMPERATURPROFIL (Lastwechsel)")
print("=" * 55)

# Startzustand: 70°C überall (Kaltstart nach Abschaltung)
T_init = {node: 70.0 for node in net._nodes}

# Quelltemperatur springt nach 10 min auf 85°C
def source_profile(t):
    return 85.0 if t >= 600 else 70.0

dynamic = net.solve_thermal_dynamic(
    hyd,
    T_initial=T_init,
    dt=60.0,          # 1-Minuten-Schritte
    t_end=3600.0,     # 1 Stunde
    source_temp_profile={"HW": source_profile},
)

# Ankunftszeit der Wärmewelle bei V3
T_hist_V3 = dynamic.temperature_history["V3"]
arrival_idx = np.argmax(T_hist_V3 > 75.0)
if arrival_idx > 0:
    arrival_time = dynamic.time_axis[arrival_idx]
    print(f"\nWärmewelle erreicht V3 nach {arrival_time/60:.1f} Minuten")

print("\nTemperaturverlauf bei V2 (Stundenwerte):")
T_hist_V2 = dynamic.temperature_history["V2"]
for i in [0, 10, 20, 30, 40, 50, 60]:
    print(f"  t={i} min: {T_hist_V2[i]:.1f} °C")
