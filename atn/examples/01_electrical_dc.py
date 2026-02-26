"""
Beispiel 1: Gleichstromnetz
============================
Reproduziert das Beispiel aus Strelow, Band 3 (2017), Abb. 1-2:
5 Knoten, 7 Leitungen.
"""
import sys
sys.path.insert(0, "..")

from atn.networks.electrical import DCNetwork

net = DCNetwork("Beispiel Band 3")

# Netz aufbauen (Topologie wie Abb. 1 in Band 3)
net.add_edge("K1", "K2", resistance=2.0, label="L1")
net.add_edge("K2", "K3", resistance=3.0, label="L2")
net.add_edge("K3", "K4", resistance=1.0, label="L3")
net.add_edge("K4", "K5", resistance=2.0, label="L4")
net.add_edge("K5", "K1", resistance=1.5, label="L5")
net.add_edge("K1", "K4", resistance=4.0, label="L6")  # Maschen-Leitungen
net.add_edge("K2", "K4", resistance=2.5, label="L7")

# Einspeisung und Entnahme
net.set_external_flow("K1",  10.0)   # Einspeisung [A]
net.set_external_flow("K3",  -4.0)   # Entnahme
net.set_external_flow("K5",  -6.0)   # Entnahme

# Gauß-Jordan-Analyse
gj = net.analyze()
print(net.summary())
print(f"Freiheitsgrad (= Anzahl Maschen): {gj.dof}")
print(f"Maschen-Zeilen in K_J:            {gj.mesh_rows}")

# Lösung
result = net.solve(reference_node="K5")
print("\nSpannungen [V]:")
for node, U in result.potentials.items():
    print(f"  {node}: {U:.3f} V")

print("\nStröme [A]:")
for edge, I in result.flows.items():
    print(f"  {edge}: {I:.3f} A")

print(f"\nBilanzkontrolle (max. Fehler): {result.balance_error:.2e} A")
