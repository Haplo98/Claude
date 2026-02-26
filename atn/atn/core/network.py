"""
Basisklasse ATNNetwork — allgemeines technisches Netz.

Vereinheitlicht elektrische, hydraulische, thermische und wirtschaftliche Netze
über dasselbe mathematische Gerüst (Strelow, ATN-Framework):

    Knotensatz:        K · I  + I_ext = 0          (Erhaltungsgesetz)
    Maschensatz:       ΔU     = K^T · U             (Potenzialverlauf)
    Widerstandsgesetz: I      = -R^(-1) · ΔU        (Fluss-Potential)
    Kombiniert:        I_ext  = -K · R^(-1) · K^T · U
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .gauss_jordan import partial_gauss_jordan, GaussJordanResult


@dataclass
class NetworkResult:
    """Ergebnis einer Netzberechnung."""
    potentials: dict[str, float]   # Knotenpotenziale (Spannung, Druck, Temperatur)
    flows: dict[str, float]        # Leitungsflüsse (Strom, Massenstrom, Wärme)
    external_flows: dict[str, float]  # Externe Ströme an Knoten
    balance_error: float           # Max. Bilanzverletzung (Qualitätskontrolle)


class ATNNetwork:
    """
    Allgemeines technisches Netz (ATN) nach Strelow.

    Domänen-unabhängige Basisklasse. Spezifische Netze (elektrisch, hydraulisch,
    thermisch) erben von dieser Klasse und überschreiben nur die physikalischen
    Interpretationen.

    Physikalische Analogien:
    ┌──────────────┬────────────────┬────────────────┬────────────────┐
    │ Domäne       │ Potenzial U    │ Fluss I        │ Widerstand R   │
    ├──────────────┼────────────────┼────────────────┼────────────────┤
    │ Elektrisch   │ Spannung [V]   │ Strom [A]      │ Ohm [Ω]        │
    │ Hydraulisch  │ Druck [Pa]     │ Massenstrom    │ hydraul. Wid.  │
    │              │                │ [kg/s]         │ [Pa·s/kg]      │
    │ Thermisch    │ Temperatur [K] │ Wärmestrom [W] │ therm. Wid.    │
    │              │                │                │ [K/W]          │
    │ Wirtschaft   │ Preis [€]      │ Mengenstrom    │ Handelswid.    │
    └──────────────┴────────────────┴────────────────┴────────────────┘
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._nodes: list[str] = []
        self._edges: list[tuple[str, str, str]] = []  # (from, to, label)
        self._resistances: dict[str, float] = {}
        self._external_flows: dict[str, float] = {}

        # Matrizen (werden bei Bedarf aufgebaut)
        self.K: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self._gj: GaussJordanResult | None = None
        self._dirty = True  # Neu bauen wenn Topologie geändert

    # ------------------------------------------------------------------
    # Netz aufbauen
    # ------------------------------------------------------------------

    def add_node(self, name: str, external_flow: float = 0.0) -> ATNNetwork:
        """Knoten hinzufügen. external_flow > 0: Einspeisung, < 0: Entnahme."""
        if name in self._nodes:
            self._external_flows[name] = external_flow
            return self
        self._nodes.append(name)
        self._external_flows[name] = external_flow
        self._dirty = True
        return self

    def add_edge(self, from_node: str, to_node: str,
                 resistance: float, label: str | None = None) -> ATNNetwork:
        """Leitung/Verbindung hinzufügen."""
        if label is None:
            label = f"L{len(self._edges) + 1}"
        # Knoten automatisch anlegen falls noch nicht vorhanden
        for n in (from_node, to_node):
            if n not in self._nodes:
                self.add_node(n)
        self._edges.append((from_node, to_node, label))
        self._resistances[label] = resistance
        self._dirty = True
        return self

    def set_external_flow(self, node: str, value: float) -> ATNNetwork:
        """Externen Fluss an einem Knoten setzen (ohne Neuaufbau der Topologie)."""
        self._external_flows[node] = value
        return self

    # ------------------------------------------------------------------
    # Matrizen
    # ------------------------------------------------------------------

    def build_matrices(self) -> None:
        """Kopplungsmatrix K und Widerstandsmatrix R aufbauen."""
        n = len(self._nodes)
        m = len(self._edges)
        node_idx = {name: i for i, name in enumerate(self._nodes)}

        K = np.zeros((n, m))
        for j, (from_n, to_n, label) in enumerate(self._edges):
            K[node_idx[from_n], j] = -1  # Ausgang
            K[node_idx[to_n], j] = +1    # Eingang

        self.K = K
        self.R = np.diag([self._resistances[e[2]] for e in self._edges])
        self._gj = None
        self._dirty = False

    @property
    def I_ext(self) -> np.ndarray:
        """Externer Flussvektor I^io."""
        return np.array([self._external_flows[n] for n in self._nodes])

    # ------------------------------------------------------------------
    # Gauß-Jordan-Analyse
    # ------------------------------------------------------------------

    def analyze(self) -> GaussJordanResult:
        """
        Gauß-Jordan-Analyse der Kopplungsmatrix.
        Liefert Rang, Freiheitsgrad, Maschen und Kausalitätstrennung.
        """
        if self._dirty:
            self.build_matrices()
        if self._gj is None:
            self._gj = partial_gauss_jordan(self.K)
        return self._gj

    @property
    def dof(self) -> int:
        """Freiheitsgrad des Netzes (Anzahl unabhängiger Leitungen)."""
        return self.analyze().dof

    @property
    def meshes(self) -> list[list[str]]:
        """Liste der Maschen als Kantenlabels."""
        gj = self.analyze()
        edge_labels = [e[2] for e in self._edges]
        result = []
        for row_idx in gj.mesh_rows:
            # Maschen-Zeile in J identifiziert die beteiligten Leitungen
            mesh_edges = [edge_labels[j]
                         for j in range(len(self._edges))
                         if abs(gj.K_J[row_idx, j]) < 1e-10
                         and abs(gj.J[row_idx].sum()) > 1e-10]
            result.append(mesh_edges)
        return result

    # ------------------------------------------------------------------
    # Lösung
    # ------------------------------------------------------------------

    def solve(self, reference_node: str | None = None) -> NetworkResult:
        """
        Löst das Netz für alle Potenziale und Flüsse.

        Verwendet die Knotenadmittanzmatrix:
            Y = K · R^(-1) · K^T
            Y · U = -I_ext   (reduziert um Referenzknoten)

        Args:
            reference_node: Knoten mit U=0. Standard: letzter Knoten.

        Returns:
            NetworkResult mit Potenzialen, Flüssen, externer Überprüfung.
        """
        if self._dirty:
            self.build_matrices()

        n_nodes = len(self._nodes)
        ref_idx = (self._nodes.index(reference_node)
                   if reference_node else n_nodes - 1)

        R_inv = np.diag(1.0 / np.diag(self.R))
        Y = self.K @ R_inv @ self.K.T  # Admittanzmatrix

        # Referenzknoten entfernen
        keep = [i for i in range(n_nodes) if i != ref_idx]
        Y_red = Y[np.ix_(keep, keep)]
        I_red = self.I_ext[keep]   # Y·U = I_ext (aus Knotensatz + Widerstandsgesetz)

        # Lineare Gleichungslösung
        U_red = np.linalg.solve(Y_red, I_red)

        # Vollständiger Potenzialvektor
        U = np.zeros(n_nodes)
        for i, orig_i in enumerate(keep):
            U[orig_i] = U_red[i]

        # Flüsse: I = -R^(-1) · ΔU = -R^(-1) · K^T · U
        delta_U = self.K.T @ U
        I_flows = -R_inv @ delta_U

        # Bilanzkontrolle
        balance = self.K @ I_flows + self.I_ext
        balance_error = float(np.max(np.abs(balance)))

        edge_labels = [e[2] for e in self._edges]
        return NetworkResult(
            potentials=dict(zip(self._nodes, U)),
            flows=dict(zip(edge_labels, I_flows)),
            external_flows=self._external_flows.copy(),
            balance_error=balance_error,
        )

    # ------------------------------------------------------------------
    # Darstellung
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Kurzbeschreibung des Netzes."""
        gj = self.analyze()
        return (
            f"ATN-Netz: '{self.name}'\n"
            f"  Knoten:       {len(self._nodes)}\n"
            f"  Leitungen:    {len(self._edges)}\n"
            f"  Rang:         {gj.rank}\n"
            f"  Freiheitsgrad:{gj.dof}\n"
            f"  Maschen:      {len(gj.mesh_rows)}\n"
        )

    def __repr__(self) -> str:
        return (f"ATNNetwork('{self.name}', "
                f"{len(self._nodes)} Knoten, {len(self._edges)} Leitungen)")
