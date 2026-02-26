"""
Elektrisches Netz — Gleichstrom, Wechselstrom, Dreiphasen-Wechselstrom.

Implementierung nach Strelow, THM-Hochschulschriften Band 3 (2017).
"""

from __future__ import annotations
import numpy as np
from ..core.network import ATNNetwork, NetworkResult


class DCNetwork(ATNNetwork):
    """
    Gleichstromnetz.
    Potenzial = Spannung [V], Fluss = Strom [A], Widerstand = Ohm [Ω].
    """

    def add_resistor(self, from_node: str, to_node: str,
                     resistance: float, label: str | None = None) -> DCNetwork:
        """Ohmschen Widerstand hinzufügen."""
        return self.add_edge(from_node, to_node, resistance, label)

    def add_voltage_source(self, node: str, voltage: float) -> DCNetwork:
        """Spannungsquelle (Einspeiseknoten)."""
        self.add_node(node, external_flow=0.0)
        # Spannung wird als Potenzialvorgabe behandelt (Referenzknoten-Methode)
        self._voltage_sources = getattr(self, '_voltage_sources', {})
        self._voltage_sources[node] = voltage
        return self

    def add_current_source(self, node: str, current: float) -> DCNetwork:
        """Stromquelle an Knoten (positiv = Einspeisung)."""
        return self.add_node(node, external_flow=current)


class ACNetwork(ATNNetwork):
    """
    Wechselstromnetz mit komplexen Impedanzen.
    Potenzial = komplexe Spannung [V], Fluss = komplexer Strom [A].

    Intern werden Real- und Imaginärteil getrennt verarbeitet
    (Strelow Band 3, Gl. 2-5).
    """

    def add_impedance(self, from_node: str, to_node: str,
                      R: float, X: float = 0.0,
                      label: str | None = None) -> ACNetwork:
        """
        Impedanz Z = R + jX hinzufügen.
        R: Wirkwiderstand [Ω], X: Blindwiderstand [Ω].
        """
        self._impedances = getattr(self, '_impedances', {})
        if label is None:
            label = f"Z{len(self._edges) + 1}"
        self._impedances[label] = complex(R, X)
        # Für die Basisklasse: |Z| als Näherungswiderstand
        self.add_edge(from_node, to_node, abs(complex(R, X)), label)
        return self

    def solve_ac(self, reference_node: str | None = None) -> dict:
        """
        Wechselstromlösung mit komplexer Admittanzmatrix.
        Gibt komplexe Spannungen, Ströme, Schein-/Wirk-/Blindleistungen zurück.
        """
        if self._dirty:
            self.build_matrices()

        n_nodes = len(self._nodes)
        ref_idx = (self._nodes.index(reference_node)
                   if reference_node else n_nodes - 1)

        # Komplexe Admittanzmatrix Y_c
        impedances = getattr(self, '_impedances', {})
        edge_labels = [e[2] for e in self._edges]
        Y_diag = np.array([1.0 / impedances.get(l, complex(self._resistances[l]))
                          for l in edge_labels], dtype=complex)
        R_inv_c = np.diag(Y_diag)

        Y_c = self.K.astype(complex) @ R_inv_c @ self.K.T.astype(complex)

        keep = [i for i in range(n_nodes) if i != ref_idx]
        Y_red = Y_c[np.ix_(keep, keep)]

        # Externe komplexe Ströme (hier vereinfacht: reell)
        I_ext_c = np.zeros(n_nodes, dtype=complex)
        for i, node in enumerate(self._nodes):
            I_ext_c[i] = self._external_flows.get(node, 0.0)

        I_red = -I_ext_c[keep]
        U_red = np.linalg.solve(Y_red, I_red)

        U_c = np.zeros(n_nodes, dtype=complex)
        for i, orig_i in enumerate(keep):
            U_c[orig_i] = U_red[i]

        I_flows_c = -R_inv_c @ (self.K.T.astype(complex) @ U_c)

        # Leistungen
        S = {self._nodes[i]: U_c[i] * np.conj(I_ext_c[i])
             for i in range(n_nodes)}

        return {
            'voltages': dict(zip(self._nodes, U_c)),
            'currents': dict(zip(edge_labels, I_flows_c)),
            'apparent_power': S,
            'active_power': {k: v.real for k, v in S.items()},
            'reactive_power': {k: v.imag for k, v in S.items()},
        }


class ThreePhaseNetwork(ACNetwork):
    """
    Dreiphasen-Wechselstromnetz.
    Erweiterung des AC-Netzes um Phasensymmetrie und verkettete Spannungen.
    (Strelow Band 3, Kapitel 4)
    """
    PHASE_SHIFT = np.exp(1j * 2 * np.pi / 3)  # 120°-Versatz

    def solve_three_phase(self, reference_node: str | None = None) -> dict:
        """
        Dreiphasenlösung unter Annahme symmetrischer Last.
        Gibt Strang- und verkettete Spannungen zurück.
        """
        result = self.solve_ac(reference_node)
        voltages = result['voltages']

        # Verkettete Spannungen (Phasen-Phasen)
        nodes = list(voltages.keys())
        line_voltages = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                key = f"U_{nodes[i]}-{nodes[j]}"
                line_voltages[key] = voltages[nodes[i]] - voltages[nodes[j]]

        result['line_voltages'] = line_voltages
        result['phase_shift'] = self.PHASE_SHIFT
        return result
