"""
Beispiel 3: Sektorenkopplung — Kopplungsbilanzen
=================================================
Zeigt die Methode der unterbestimmten Kopplungsbilanzen (Strelow 2024).

Teil A: Drei Heizwerke — Strelow 2024, Kap. 2.1, Abb. 1-4
    Drei Heizwerke versorgen einen Gebäudekomplex über ein Wärmenetz.
    -> 1 Bilanz, 3 Variablen, dof=2, hexagonaler Entscheidungsraum

Teil B: KWK-Anlage mit zwei Energieträgern — erweitertes Beispiel
    Gas-BHKW + Spitzenlastkessel versorgen Strom- und Wärmenetz.
    -> 2 Bilanzen, 3 Variablen, dof=1
"""
import sys
sys.path.insert(0, "..")
from atn.coupling.model import CouplingModel

# ????????????????????????????????????????????????????????????????????????
# TEIL A: Drei Heizwerke (Strelow 2024, Kap. 2.1)
# ????????????????????????????????????????????????????????????????????????
print("=" * 65)
print("TEIL A: DREI HEIZWERKE")
print("Strelow 2024 — Kap. 2.1, Abb. 1")
print("=" * 65)

Q_bedarf = 800.0   # kW — meteorologisch vorgegebener Gesamtbedarf

model_A = CouplingModel("Drei Heizwerke")

model_A.add_variable("Q1_Heizwerk1", "kW", lower=0, upper=500)
model_A.add_variable("Q2_Heizwerk2", "kW", lower=0, upper=500)
model_A.add_variable("Q3_Heizwerk3", "kW", lower=0, upper=500)

# Einzige Bilanz: Wärmebedarf (Gl. 1 in Strelow 2024)
model_A.add_balance(
    "Wärme",
    {"Q1_Heizwerk1": 1.0, "Q2_Heizwerk2": 1.0, "Q3_Heizwerk3": 1.0},
    rhs=Q_bedarf,
)

model_A.print_matrix()
analysis_A = model_A.analyze()
print("\n" + model_A.summary())
# -> dof=2: Q2 und Q3 sind Entscheidungsgrößen, Q1 ist Folgegröße

# Betriebspunkt-Berechnung
print("\nBetriebspunkte (Strelow 2024, Abb. 2 — Entscheidungsraum):")
test_points = [
    {"Q2_Heizwerk2": 200.0, "Q3_Heizwerk3": 150.0},
    {"Q2_Heizwerk2": 400.0, "Q3_Heizwerk3": 300.0},  # Q1 würde negativ -> unzulässig
    {"Q2_Heizwerk2": 100.0, "Q3_Heizwerk3": 100.0},
]
for dec in test_points:
    sol = model_A.solve(dec)
    ok = model_A.is_feasible(sol)
    q1 = sol['Q1_Heizwerk1']
    print(f"  Q2={dec['Q2_Heizwerk2']:5.0f}, Q3={dec['Q3_Heizwerk3']:5.0f} -> "
          f"Q1={q1:6.1f} kW  {'OK' if ok else 'UNZULAESSIG'}")

# Kostenoptimierung (Heizwerk 3 = Wärmepumpe, günstigster Betrieb)
print("\nKostenoptimierung (spezifische Kosten [€/kWh]):")
costs_A = {"Q1_Heizwerk1": 0.08,   # Gaskessel
           "Q2_Heizwerk2": 0.06,   # Biogas
           "Q3_Heizwerk3": 0.04}   # Wärmepumpe (günstigst)
opt_A = model_A.optimize(costs_A)
print(f"  Status: {opt_A.status}")
print(f"  Minimale Kosten: {opt_A.objective_value:.2f} €/h")
for k, v in opt_A.solution.items():
    marker = " [Entscheidungsgroesse]" if k in opt_A.decision_values else " [Folgegroesse]"
    print(f"    {k:<22}: {v:7.1f} kW{marker}")

# Entscheidungsraum
space_A = model_A.decision_space_2d("Q2_Heizwerk2", "Q3_Heizwerk3", n_samples=50)
print(f"\nEntscheidungsraum: {space_A['n_feasible']} von {50*50} "
      f"Rasterpunkten zulässig")
print("  (entspricht dem farbigen Sechseck in Strelow 2024, Abb. 2)")
print("  Begrenzung durch: Q_min, Q_max jeder Anlage + Bilanzbedingung")

# ????????????????????????????????????????????????????????????????????????
# TEIL B: KWK-System — Strom + Wärme gekoppelt
# ????????????????????????????????????????????????????????????????????????
print("\n" + "=" * 65)
print("TEIL B: KWK-SYSTEM (Strom + Wärme, dof=1)")
print("=" * 65)

# Systemparameter
P_el_bedarf = 80.0   # kW Strombedarf
Q_hz_bedarf = 300.0  # kW Wärmebedarf
eta_el = 0.35        # elektr. Wirkungsgrad BHKW
eta_hz = 0.55        # therm. Wirkungsgrad BHKW

model_B = CouplingModel("KWK + Spitzenlastkessel")

# Variablen
model_B.add_variable("P_BHKW",    "kW", lower=20,  upper=150,
                     description="BHKW elektr. Leistung")
model_B.add_variable("Q_Kessel",  "kW", lower=0,   upper=200,
                     description="Spitzenlastkessel Wärme")
model_B.add_variable("P_Netz",    "kW", lower=-50, upper=50,
                     description="Netzbezug/Einspeisung (±)")

# Strom-Bilanz: P_BHKW + P_Netz = P_el_bedarf
model_B.add_balance(
    "Elektrizität",
    {"P_BHKW": 1.0, "P_Netz": 1.0},
    rhs=P_el_bedarf,
)

# Wärme-Bilanz: BHKW-Wärme + Kessel = Q_hz_bedarf
# Q_BHKW = P_BHKW * (eta_hz/eta_el)
model_B.add_balance(
    "Wärme",
    {"P_BHKW": eta_hz/eta_el, "Q_Kessel": 1.0},
    rhs=Q_hz_bedarf,
)

model_B.print_matrix()
print()
analysis_B = model_B.analyze()
print(model_B.summary())

# Einzelne Betriebspunkte (P_Netz = Entscheidungsgröße: neg. = Einspeisung, pos. = Bezug)
print("\nBetriebspunkte [P_Netz als Entscheidungsgröße]:")
for p_netz in [-50, -20, 0, 16, 20]:
    sol = model_B.solve({"P_Netz": float(p_netz)})
    ok = model_B.is_feasible(sol)
    print(f"  P_Netz={p_netz:+5.0f} kW -> "
          f"P_BHKW={sol['P_BHKW']:6.1f} kW, "
          f"Q_Kessel={sol['Q_Kessel']:6.1f} kW  "
          f"{'OK' if ok else 'UNZULAESSIG'}")

# Optimierung: maximale Eigenerzeugung (minimaler Netzbezug)
print("\nOptimierung: minimaler Netzbezug:")
opt_B = model_B.optimize({"P_Netz": 1.0}, minimize_obj=True)
print(f"  Status: {opt_B.status}")
if opt_B.status == 'optimal':
    print(f"  Optimaler Netzbezug: {opt_B.solution.get('P_Netz', '?'):.1f} kW")
    for k, v in opt_B.solution.items():
        print(f"    {k:<15}: {v:7.1f} kW")
