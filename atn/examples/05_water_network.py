"""
Beispiel 5: Wasserversorgungsnetz — Druckberechnung und Leckagedetektion
=========================================================================

Demonstriert die Anwendung des ATN-Frameworks auf Trinkwassernetze.

Netzstruktur (vereinfachtes Stadtverteilnetz):

    [HB1] ──P1── [K1] ──P3── [K3]
      │                        │
      P2                       P5
      │                        │
    [K2] ──P4── [K4] ──────  [K5]

    HB1 : Hochbehälter (Druckhöhe 45 m ü. NHN, Geländehöhe 30 m)
    K1–K5: Verbraucherknoten mit Entnahme und Geländehöhe
    P1–P5: PE-100-Rohrleitungen (C = 140)

Teil A: Hydraulische Berechnung (Druckhöhen, Volumenströme, Fließgeschwindigkeiten)
Teil B: DVGW W 303 Druckprüfung
Teil C: Leckagedetektion (simuliertes Messzenario mit Leckage in P3)
"""

import sys
sys.path.insert(0, "..")
from atn.networks.water import WaterNetwork

# ─── NETZ AUFBAUEN ────────────────────────────────────────────────────────────

net = WaterNetwork("Stadtverteilnetz Beispiel")

# Hochbehälter als Druckrandbedingung
net.add_reservoir("HB1", head=45.0, elevation=30.0)

# Rohrleitungen (PE 100, Hazen-Williams C = 140)
net.add_pipe("HB1", "K1", length=800, diameter=0.20, hazen_williams_c=140, label="P1")
net.add_pipe("HB1", "K2", length=600, diameter=0.15, hazen_williams_c=140, label="P2")
net.add_pipe("K1",  "K3", length=500, diameter=0.15, hazen_williams_c=140, label="P3")
net.add_pipe("K2",  "K4", length=400, diameter=0.10, hazen_williams_c=140, label="P4")
net.add_pipe("K3",  "K5", length=300, diameter=0.10, hazen_williams_c=140, label="P5")

# Verbraucher (Geländehöhe [m ü. NHN], Entnahme [m³/s])
net.add_demand("K1", flow=0.004, elevation=15.0)   # 4,0 l/s  Gewerbegebiet
net.add_demand("K2", flow=0.003, elevation=12.0)   # 3,0 l/s  Wohngebiet Süd
net.add_demand("K3", flow=0.002, elevation=18.0)   # 2,0 l/s  Wohngebiet Nord
net.add_demand("K4", flow=0.002, elevation=10.0)   # 2,0 l/s  Wohngebiet West
net.add_demand("K5", flow=0.001, elevation=16.0)   # 1,0 l/s  Einzelgebäude

# ─── TEIL A: HYDRAULISCHE BERECHNUNG ─────────────────────────────────────────

print("=" * 65)
print("TEIL A: HYDRAULISCHE BERECHNUNG")
print("=" * 65)

result = net.solve_hydraulic()

print(f"\nKonvergenz: {'ja' if result.converged else 'NEIN'} "
      f"({result.iterations} Iterationen)")

print("\nDruckhöhen und Versorgungsdrücke:")
print(f"  {'Knoten':<8} {'Druckhöhe [m]':>15} {'Druck [bar]':>12}  Status")
print("  " + "-" * 50)
for node in sorted(result.heads):
    h = result.heads[node]
    p = result.pressures[node]
    ok = "OK" if node not in result.pressure_violations else "! VERLETZUNG"
    print(f"  {node:<8} {h:>15.2f} {p:>12.3f}  {ok}")

print("\nRohrleitungen:")
print(f"  {'Leitung':<10} {'Q [l/s]':>10} {'v [m/s]':>10} {'dh [m WS]':>12}")
print("  " + "-" * 46)
for label in sorted(result.flows):
    q_ls = result.flows[label] * 1000       # m³/s -> l/s
    v    = result.velocities.get(label, 0)
    dh   = result.head_losses.get(label, 0)
    print(f"  {label:<10} {q_ls:>10.2f} {v:>10.3f} {dh:>12.3f}")

# ─── TEIL B: DVGW W 303 ───────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TEIL B: DVGW W 303 — VERSORGUNGSDRUCKPRÜFUNG")
print("=" * 65)

dvgw = net.check_dvgw_w303(result)
print(f"\n{dvgw['summary']}\n")
for node, info in sorted(dvgw['details'].items()):
    print(f"  {node}: {info['pressure_bar']:.3f} bar  ->  {info['status']}")

# ─── TEIL C: LECKAGEDETEKTION ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TEIL C: LECKAGEDETEKTION")
print("=" * 65)

# Simuliertes Szenario: Leitung P3 hat Leckage (28 % Volumenstromverlust)
# In der Praxis: Messwerte aus Durchflussmessern (AMR-System)
measured = {
    "P1": result.flows["P1"] * 0.98,    # ±2 % Messungenauigkeit
    "P2": result.flows["P2"] * 1.01,
    "P3": result.flows["P3"] * 0.72,    # ← 28 % Verlust durch Leckage!
    "P4": result.flows["P4"] * 0.99,
    "P5": result.flows["P5"] * 0.97,
}

print("\nSimuliertes Szenario: Leckage in Leitung P3 (28 % Volumenstromverlust)")
print("Eingangsdaten: Messwerte aus Netz-AMR-System\n")

candidates = net.detect_leaks(measured, threshold=0.0005)

if candidates:
    print(f"{len(candidates)} Leckage-Kandidat(en) identifiziert:")
    print(f"  {'Leitung':<10} {'Residual [l/s]':>16} {'Score':>8}")
    print("  " + "-" * 36)
    for c in candidates:
        print(f"  {c.pipe_label:<10} {c.residual * 1000:>16.2f} {c.probability:>8.2f}")
    print(f"\n-> Empfehlung: Leitung '{candidates[0].pipe_label}' zuerst inspizieren.")
else:
    print("Keine signifikanten Leckagen detektiert.")
