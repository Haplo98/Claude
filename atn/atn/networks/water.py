"""
Wasserversorgungsnetz — hydraulische Berechnung und Leckagedetektion.

Erweiterung des ATN-Frameworks auf Trinkwasser- und Brauchwassernetze.
Implementiert nach dem Vorbild des Fernwärmemodul (Strelow & Kouka 2025),
angepasst für Wasserverteilnetze (DVGW W 303, W 400).

Physikalisches Modell:
  Potenzial  : Hydraulische Druckhöhe [m WS]
  Fluss      : Volumenstrom [m³/s]
  Widerstand : Rohrwiderstand nach Hazen-Williams (linearisiert)

Besonderheiten gegenüber Fernwärme:
  - Einrohr-System (keine Rücklaufleitung)
  - Druckzonen mit mehreren Reservoiren (unterschiedliche Druckhöhen)
  - Pumpen als aktive Elemente (Druckerhöhung)
  - Leckagedetektion durch Messdatenabgleich (Residual-Analyse)
  - DVGW W 303 Druckprüfung (2–8 bar Versorgungsdruck)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.network import ATNNetwork

# Fluid-Eigenschaften Trinkwasser (15 °C)
WATER_RHO = 999.1    # kg/m³
WATER_MU  = 1.14e-3  # Pa·s
GRAVITY   = 9.81     # m/s²

# DVGW W 303 Richtwerte
P_MIN_BAR = 2.0      # bar  Mindestversorgungsdruck
P_MAX_BAR = 8.0      # bar  Maximaldruck


@dataclass
class WaterHydraulicResult:
    heads: dict[str, float]           # Druckhöhe [m WS] je Knoten
    pressures: dict[str, float]       # Versorgungsdruck [bar] je Knoten
    flows: dict[str, float]           # Volumenstrom [m³/s] je Leitung
    velocities: dict[str, float]      # Fließgeschwindigkeit [m/s] je Leitung
    head_losses: dict[str, float]     # Druckhöhenverlust [m WS] je Leitung
    pressure_violations: list[str]    # Knoten außerhalb DVGW-Richtwert
    iterations: int
    converged: bool


@dataclass
class LeakCandidate:
    """Leckage-Kandidat aus der Residual-Analyse."""
    pipe_label: str
    residual: float      # |Q_gemessen - Q_berechnet| [m³/s]
    probability: float   # heuristischer Score 0–1


class WaterNetwork(ATNNetwork):
    """
    Wasserversorgungsnetz nach ATN-Methode (Strelow).

    Modelliert Trinkwassernetze mit Reservoiren, Rohrleitungen, Pumpen
    und Verbrauchern. Hydraulische Berechnung durch iterative Linearisierung
    der Hazen-Williams-Formel (analog zu Strelow & Kouka 2025).

    Typische Anwendung:
        net = WaterNetwork("Stadtwerke Musterstadt")
        net.add_reservoir("HB1", head=45.0, elevation=30.0)
        net.add_pipe("HB1", "K1", length=500, diameter=0.2)
        net.add_demand("K1", flow=0.005, elevation=15.0)
        result = net.solve_hydraulic()
        print(net.check_dvgw_w303(result)['summary'])
    """

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._pipe_params: dict[str, dict] = {}
        self._pump_heads: dict[str, float] = {}       # label → Förderhöhe [m WS]
        self._reservoir_heads: dict[str, float] = {}  # node  → Druckhöhe [m WS]
        self._elevation: dict[str, float] = {}        # node  → Geländehöhe [m ü. NHN]

    # ------------------------------------------------------------------
    # Netz aufbauen
    # ------------------------------------------------------------------

    def add_reservoir(self, node: str,
                      head: float,
                      elevation: float = 0.0) -> WaterNetwork:
        """
        Reservoir / Hochbehälter mit fester Druckhöhe (Dirichlet-Randbedingung).

        Args:
            head     : Wasserstand / hydraulische Druckhöhe [m WS]
            elevation: Geländehöhe des Knotens [m ü. NHN]
        """
        self._reservoir_heads[node] = head
        self._elevation[node] = elevation
        self.add_node(node, external_flow=0.0)
        return self

    def add_pipe(self,
                 from_node: str, to_node: str,
                 length: float,
                 diameter: float,
                 hazen_williams_c: float = 130.0,
                 label: str | None = None) -> WaterNetwork:
        """
        Rohrleitung hinzufügen.

        Args:
            length           : Länge [m]
            diameter         : Innendurchmesser [m]
            hazen_williams_c : Hazen-Williams-Beiwert C (DVGW-Tabellen):
                               Gusseisen alt ~80, Stahl verzinkt ~120,
                               Grauguss ~100, PVC/PE neu ~140
        """
        if label is None:
            label = f"P{len(self._edges) + 1}"
        self._pipe_params[label] = {
            'length': length,
            'diameter': diameter,
            'hazen_williams_c': hazen_williams_c,
        }
        R_init = self._hw_resistance(label, flow=1e-3)
        self.add_edge(from_node, to_node, R_init, label)
        return self

    def add_pump(self,
                 from_node: str, to_node: str,
                 pump_head: float,
                 label: str | None = None) -> WaterNetwork:
        """
        Kreiselpumpe (Druckerhöhung pump_head [m WS]).

        Modelliert als Leitung mit sehr geringem Widerstand und
        zusätzlicher externer Druckhöhenquelle an den Endknoten.
        """
        if label is None:
            label = f"PU{len(self._edges) + 1}"
        self._pump_heads[label] = pump_head
        self._pipe_params[label] = {
            'length': 1.0, 'diameter': 0.3, 'hazen_williams_c': 150.0,
        }
        self.add_edge(from_node, to_node, 0.01, label)  # vernachlässigbarer R
        return self

    def add_demand(self, node: str,
                   flow: float,
                   elevation: float = 0.0) -> WaterNetwork:
        """
        Verbraucher-Knoten.

        Args:
            flow     : Entnahme-Volumenstrom [m³/s] (positiv = Entnahme)
            elevation: Geländehöhe [m ü. NHN] (für Druckberechnung)
        """
        self._elevation[node] = elevation
        self.add_node(node, external_flow=-flow)
        return self

    # ------------------------------------------------------------------
    # Widerstandsberechnung nach Hazen-Williams
    # ------------------------------------------------------------------

    def _hw_resistance(self, label: str, flow: float) -> float:
        """
        Linearisierter hydraulischer Widerstand R_lin [m WS/(m³/s)].

        Hazen-Williams (SI):
            Δh = 10.67 · L · Q^1.852 / (C^1.852 · D^4.87)

        Linearisierung um Arbeitspunkt Q₀:
            R_lin = dΔh/dQ |_{Q₀}
                  = 10.67 · L · 1.852 · Q₀^0.852 / (C^1.852 · D^4.87)
        """
        p = self._pipe_params[label]
        L, D, C = p['length'], p['diameter'], p['hazen_williams_c']
        Q0 = max(abs(flow), 1e-6)
        return float(10.67 * L * 1.852 * Q0**0.852 / (C**1.852 * D**4.87))

    def _hw_head_loss(self, label: str, flow: float) -> float:
        """Exakter Hazen-Williams-Druckhöhenverlust [m WS]."""
        p = self._pipe_params[label]
        L, D, C = p['length'], p['diameter'], p['hazen_williams_c']
        Q = abs(flow)
        if Q < 1e-9:
            return 0.0
        return float(10.67 * L * Q**1.852 / (C**1.852 * D**4.87))

    def _velocity(self, label: str, flow: float) -> float:
        """Mittlere Fließgeschwindigkeit [m/s]."""
        D = self._pipe_params[label]['diameter']
        A = np.pi * D**2 / 4
        return abs(flow) / A

    # ------------------------------------------------------------------
    # Hydraulische Netzberechnung
    # ------------------------------------------------------------------

    def solve_hydraulic(self,
                        max_iter: int = 50,
                        tol: float = 1e-7) -> WaterHydraulicResult:
        """
        Iterative hydraulische Netzberechnung (Hazen-Williams).

        Algorithmus (analog Strelow & Kouka 2025, Kap. 3.2):
          1. Initiallösung mit geschätzten Linearisierungswiderständen
          2. Hazen-Williams-Widerstände aus aktuellem Volumenstrom aktualisieren
          3. Wiederholen bis ||ΔQ||_∞ < tol

        Reservoir-Knoten dienen als Referenzknoten (feste Druckhöhe).
        Pumpen erhöhen die Druckhöhe am Zielknoten um pump_head.
        """
        if self._dirty:
            self.build_matrices()

        flows = {e[2]: 1e-3 for e in self._edges}
        converged = False
        result = None

        for iteration in range(max_iter):
            for label in [e[2] for e in self._edges]:
                if label in self._pipe_params and label not in self._pump_heads:
                    self._resistances[label] = self._hw_resistance(
                        label, flows.get(label, 1e-3))

            self.build_matrices()

            ref_node = (list(self._reservoir_heads.keys())[0]
                        if self._reservoir_heads else None)
            result = self.solve(reference_node=ref_node)
            flows_new = result.flows

            max_change = max(
                abs(flows_new.get(l, 0) - flows.get(l, 0))
                for l in flows_new
            )
            flows = flows_new
            if max_change < tol:
                converged = True
                break

        # Druckhöhen aus ATN-Potenzialen + Reservoir-Anker
        h_offset = 0.0
        if self._reservoir_heads and result is not None:
            ref = list(self._reservoir_heads.keys())[0]
            h_offset = self._reservoir_heads[ref] - result.potentials.get(ref, 0.0)

        heads = {n: result.potentials[n] + h_offset for n in self._nodes}

        # Pumpenhöhe auf Zielknoten aufaddieren
        for label, dh in self._pump_heads.items():
            edge = next((e for e in self._edges if e[2] == label), None)
            if edge:
                to_node = edge[1]
                heads[to_node] = heads.get(to_node, 0.0) + dh

        pressures = {
            n: (heads[n] - self._elevation.get(n, 0.0)) * WATER_RHO * GRAVITY / 1e5
            for n in self._nodes
        }
        head_losses = {
            e[2]: self._hw_head_loss(e[2], flows.get(e[2], 0))
            for e in self._edges if e[2] in self._pipe_params
        }
        velocities = {
            e[2]: self._velocity(e[2], flows.get(e[2], 0))
            for e in self._edges if e[2] in self._pipe_params
        }
        violations = [
            n for n, p in pressures.items()
            if p < P_MIN_BAR or p > P_MAX_BAR
        ]

        return WaterHydraulicResult(
            heads=heads,
            pressures=pressures,
            flows=flows,
            velocities=velocities,
            head_losses=head_losses,
            pressure_violations=violations,
            iterations=iteration + 1,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Leckagedetektion
    # ------------------------------------------------------------------

    def detect_leaks(self,
                     measured_flows: dict[str, float],
                     threshold: float = 0.001) -> list[LeakCandidate]:
        """
        Leckagedetektion durch Residual-Analyse.

        Vergleicht gemessene Volumenströme mit dem berechneten Netzmodell.
        Leitungen mit |Q_gemessen - Q_berechnet| > threshold gelten als
        Leckage-Kandidaten (Strelow: Validierung durch Messdatenabgleich).

        Args:
            measured_flows: {Leitungslabel → gemessener Volumenstrom [m³/s]}
            threshold     : Mindestschwelle für Leckage-Verdacht [m³/s]
                            Standard: 1 l/s

        Returns:
            Sortierte Liste von LeakCandidate (größtes Residual zuerst)
        """
        calc = self.solve_hydraulic().flows
        total = sum(abs(v) for v in measured_flows.values()) + 1e-12

        candidates = []
        for label, q_meas in measured_flows.items():
            residual = abs(q_meas - calc.get(label, 0.0))
            if residual > threshold:
                candidates.append(LeakCandidate(
                    pipe_label=label,
                    residual=residual,
                    probability=min(residual / total * 10, 1.0),
                ))
        return sorted(candidates, key=lambda c: c.residual, reverse=True)

    # ------------------------------------------------------------------
    # DVGW-Prüfung
    # ------------------------------------------------------------------

    def check_dvgw_w303(self, result: WaterHydraulicResult) -> dict:
        """
        DVGW W 303: Versorgungsdruckprüfung an allen Verbraucherknoten.

        Richtwerte: 2 bar (Mindest) bis 8 bar (Maximum).

        Returns:
            dict mit 'ok', 'violations', 'details', 'summary'
        """
        ok = len(result.pressure_violations) == 0
        details = {
            n: {
                'pressure_bar': result.pressures[n],
                'status': ('OK'
                           if P_MIN_BAR <= result.pressures[n] <= P_MAX_BAR
                           else 'VERLETZUNG'),
            }
            for n in result.pressures
        }
        v = len(result.pressure_violations)
        return {
            'ok': ok,
            'violations': result.pressure_violations,
            'details': details,
            'summary': (
                f"DVGW W 303: {'Alle Knoten OK' if ok else f'{v} Verletzung(en)'}"
                f" (Richtwert {P_MIN_BAR}–{P_MAX_BAR} bar)"
            ),
        }
