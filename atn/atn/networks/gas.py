"""
Gasnetz — hydraulische Berechnung für Nieder-, Mittel- und Hochdrucknetze.

Implementierung nach DVGW G 462/G 600 und Strelow ATN-Framework.
Struktur analog zu DistrictHeatingNetwork und WaterNetwork.

Physikalisches Modell (ATN-Analogie):
  ┌─────────────┬───────────────────┬───────────────────┬────────────────────┐
  │ Druckstufe  │ Potenzial U       │ Fluss I           │ Widerstand R       │
  ├─────────────┼───────────────────┼───────────────────┼────────────────────┤
  │ ND (≤100mbar│ Druck p [Pa]      │ Q_N [Nm³/s]       │ Renouard-ND (lin.) │
  │ MD (≤1 bar) │ Druck² p² [Pa²]   │ Q_N [Nm³/s]       │ Renouard-MD (lin.) │
  │ HD (>1 bar) │ Druck² p² [Pa²]   │ Q_N [Nm³/s]       │ Weymouth    (lin.) │
  └─────────────┴───────────────────┴───────────────────┴────────────────────┘

Linearisierung (analog zu Strelow & Kouka 2025 für Fernwärme):
  Alle Formeln sind nichtlinear in Q_N. Der iterative ATN-Solver
  linearisiert um den aktuellen Arbeitspunkt Q₀ und aktualisiert
  die Widerstände bis zur Konvergenz.

Normzustand (DIN 1343): T_N = 273.15 K, p_N = 101325 Pa.
Volumenströme sind stets auf Normzustand bezogen (Nm³/s).

Formeln:
  Renouard ND:  Δp       = 232    · ρ_rel · L · Q_N^1.82 / D^4.82
  Renouard MD:  Δ(p²)    = 48600  · ρ_rel · L · Q_N^1.82 / D^4.82
  Weymouth:     Δ(p²)    = R_W(D) · L · Q_N²
                R_W(D)   = 8·λ_W·ρ_N·T·Z·R_m / (π²·D^5·p_N·T_N)
                λ_W      = 0.009407 · D^(-1/3)   [Weymouth-Reibungszahl]

Quellen:
  DVGW G 462    Gasleitungen aus Stahl
  DVGW G 600    Technische Regeln Gasinstallation
  Renouard 1952 Formules de calcul des réseaux de distribution de gaz
  Weymouth 1912 Transmission of natural gas
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.network import ATNNetwork

# ── Normzustand (DIN 1343) ─────────────────────────────────────────────────
T_NORM = 273.15    # K
P_NORM = 101325.0  # Pa
R_M    = 8.314     # J/(mol·K)  universelle Gaskonstante

# ── Erdgas H (Standardgas, DVGW G 260) ────────────────────────────────────
GAS_RHO_REL  = 0.625   # relative Dichte gegen Luft (Erdgas H, typisch)
GAS_RHO_NORM = 0.785   # kg/Nm³  Normdichte Erdgas H
GAS_MU       = 1.1e-5  # Pa·s  dynamische Viskosität Erdgas (20 °C)
GAS_Z        = 0.998   # Realgasfaktor (≈1 bei ND/MD)
GAS_T_SOIL   = 283.15  # K  mittlere Bodentemperatur (10 °C)
AIR_RHO_NORM = 1.293   # kg/Nm³

# ── DVGW G 600: Druckstufengrenzen ────────────────────────────────────────
P_ND_MAX = 100e2   # Pa  ND ≤ 100 mbar
P_MD_MAX = 1e5     # Pa  MD ≤  1 bar
# HD: > 1 bar (bis max. Betriebsdruck der Leitung)


class PressureLevel:
    ND = "ND"   # Niederdruck    ≤ 100 mbar
    MD = "MD"   # Mitteldruck    ≤ 1 bar
    HD = "HD"   # Hochdruck      > 1 bar


@dataclass
class GasHydraulicResult:
    pressures: dict[str, float]       # Druck [Pa] je Knoten
    pressures_bar: dict[str, float]   # Druck [bar] je Knoten
    flows_norm: dict[str, float]      # Normvolumenstrom [Nm³/h] je Leitung
    velocities: dict[str, float]      # Strömungsgeschwindigkeit [m/s] je Leitung
    pressure_drops: dict[str, float]  # Druckverlust [mbar] je Leitung
    pressure_level: str               # 'ND' | 'MD' | 'HD'
    pressure_violations: list[str]    # Knoten außerhalb DVGW-Richtwert
    iterations: int
    converged: bool


@dataclass
class LinepackResult:
    """Linienspeicher-Kapazität des Netzes (stationär)."""
    pipe_label: str
    volume_norm_m3: float    # Gasinhalt bei Betriebsdruck [Nm³]
    pressure_pa: float       # mittlerer Betriebsdruck [Pa]


class GasNetwork(ATNNetwork):
    """
    Gasnetz nach ATN-Methode (Strelow), DVGW G 462/G 600.

    Unterstützt alle drei DVGW-Druckstufen mit automatischer Formelwahl:
      ND (≤ 100 mbar): Renouard-ND       → Linearisierung in p [Pa]
      MD (≤ 1 bar)   : Renouard-MD       → Linearisierung in p² [Pa²]
      HD (> 1 bar)   : Weymouth          → Linearisierung in p² [Pa²]

    Typische Anwendung (MD-Verteilnetz):
        net = GasNetwork("Stadtwerk Musterstadt MD-Netz")
        net.add_feed("ÜST", pressure_bar=0.5)
        net.add_pipe("ÜST", "K1", length=500, diameter=0.1)
        net.add_offtake("K1", flow_m3h=50.0)
        result = net.solve_hydraulic()
        print(net.check_dvgw(result)['summary'])

    Gasparameter können über set_gas_properties() überschrieben werden
    (z.B. für Biogas, Wasserstoff-Beimischung).
    """

    def __init__(self, name: str = "",
                 pressure_level: str = PressureLevel.MD):
        super().__init__(name)
        self._pressure_level = pressure_level
        self._pipe_params: dict[str, dict] = {}
        self._compressor_ratios: dict[str, float] = {}
        self._feed_pressures: dict[str, float] = {}      # Knoten → Einspeisedruck [Pa]
        self._min_pressures: dict[str, float] = {}       # Knoten → DVGW-Mindestdruck [Pa]

        # Gasparameter (überschreibbar für Biogas, H₂-Blend, ...)
        self._rho_rel = GAS_RHO_REL
        self._rho_n   = GAS_RHO_NORM
        self._z       = GAS_Z
        self._temp_k  = GAS_T_SOIL

    # ------------------------------------------------------------------
    # Konfiguration
    # ------------------------------------------------------------------

    def set_gas_properties(self,
                           rho_rel: float = GAS_RHO_REL,
                           rho_norm: float = GAS_RHO_NORM,
                           z_factor: float = GAS_Z,
                           temperature_k: float = GAS_T_SOIL) -> GasNetwork:
        """
        Gaseigenschaften überschreiben (für Biogas, H₂-Blend etc.).

        Args:
            rho_rel      : relative Dichte gegen Luft (Erdgas H: 0.625, Biogas: ~0.87)
            rho_norm     : Normdichte [kg/Nm³]
            z_factor     : Realgasfaktor (ND/MD: ≈ 1.0, HD-Erdgas: 0.90–0.99)
            temperature_k: mittlere Gastemperatur [K]
        """
        self._rho_rel = rho_rel
        self._rho_n   = rho_norm
        self._z       = z_factor
        self._temp_k  = temperature_k
        return self

    # ------------------------------------------------------------------
    # Netz aufbauen
    # ------------------------------------------------------------------

    def add_feed(self, node: str,
                 pressure_bar: float,
                 flow_m3h: float = 0.0,
                 min_pressure_bar: float | None = None) -> GasNetwork:
        """
        Einspeiseknoten (Übergabestation, Verdichterstation, Druckregler).

        Args:
            pressure_bar    : Einspeisedruck [bar] (Dirichlet-Randbedingung)
            flow_m3h        : Einspeisung [Nm³/h], 0 = aus Druck geregelt
            min_pressure_bar: DVGW-Mindestdruck [bar] (Standard: Druckstufe)
        """
        p_pa = pressure_bar * 1e5
        self._feed_pressures[node] = p_pa
        if min_pressure_bar is not None:
            self._min_pressures[node] = min_pressure_bar * 1e5
        self.add_node(node, external_flow=flow_m3h / 3600.0)  # Nm³/h → Nm³/s
        return self

    def add_pipe(self,
                 from_node: str, to_node: str,
                 length: float,
                 diameter: float,
                 roughness: float = 1.0e-4,
                 label: str | None = None,
                 material: str = "Stahl") -> GasNetwork:
        """
        Gasleitung hinzufügen.

        Args:
            length   : Länge [m]
            diameter : Innendurchmesser [m]
            roughness: Wandrauheit ε [m] (Stahl neu: 0.05 mm, alt: 0.1 mm,
                       PE: 0.007 mm)
            material : Info für Dokumentation (kein Einfluss auf Berechnung)
        """
        if label is None:
            label = f"G{len(self._edges) + 1}"
        self._pipe_params[label] = {
            'length': length,
            'diameter': diameter,
            'roughness': roughness,
            'material': material,
        }
        R_init = self._gas_resistance(label, flow=1e-3)
        self.add_edge(from_node, to_node, R_init, label)
        return self

    def add_compressor(self, from_node: str, to_node: str,
                       pressure_ratio: float,
                       label: str | None = None) -> GasNetwork:
        """
        Verdichter (Druckerhöhung um Faktor pressure_ratio).

        Modelliert als Leitung mit vernachlässigbarem Widerstand plus
        externe Druckerhöhung am Ausgangsknoten.

        Args:
            pressure_ratio: p_aus / p_ein [-] (z.B. 1.5 = 50% Druckerhöhung)
        """
        if label is None:
            label = f"C{len(self._edges) + 1}"
        self._compressor_ratios[label] = pressure_ratio
        self._pipe_params[label] = {
            'length': 1.0, 'diameter': 0.5, 'roughness': 1e-5, 'material': 'Verdichter'
        }
        self.add_edge(from_node, to_node, 1e-6, label)
        return self

    def add_offtake(self, node: str,
                    flow_m3h: float,
                    min_pressure_bar: float | None = None) -> GasNetwork:
        """
        Entnahmeknoten (Hausanschluss, Industrie, Kraftwerk).

        Args:
            flow_m3h        : Entnahme [Nm³/h] (positiv = Entnahme)
            min_pressure_bar: DVGW-Mindestdruck [bar] an diesem Knoten
        """
        if min_pressure_bar is not None:
            self._min_pressures[node] = min_pressure_bar * 1e5
        self.add_node(node, external_flow=-flow_m3h / 3600.0)  # negativ = Entnahme
        return self

    # ------------------------------------------------------------------
    # Widerstandsberechnung (Darcy-Weisbach, analog Fernwärme)
    # ------------------------------------------------------------------

    def _moody_friction(self, label: str, flow: float,
                        p_mean: float) -> float:
        """
        Moody-Reibungszahl für Gas (analog Strelow & Kouka 2025, Kap. 2.2).

        Betriebs-Reynolds-Zahl:
            Re = ρ_B · v_B · D / μ
            ρ_B = ρ_N · p_B/p_N,  v_B = Q_N · p_N/(p_B · A)
            → Re = ρ_N · Q_N · p_N / (A · μ · p_B) × D    [p-abhängig]

        Für ND (p_B ≈ p_N): Re ≈ ρ_N · v_N · D / μ

        Regime:  Re < 2300  → laminar:   f = 64/Re
                 Re ≥ 2300  → turbulent: Swamee-Jain (explizite Colebrook-Näherung)
        """
        p_data = self._pipe_params[label]
        D, eps = p_data['diameter'], p_data['roughness']
        A = np.pi * D**2 / 4

        Q_N = max(abs(flow), 1e-9)
        p_B = max(p_mean, 1e3)

        # Betriebsgeschwindigkeit: v_B = Q_N · p_N/p_B / A
        v_B = Q_N * P_NORM / (p_B * A)
        rho_B = self._rho_n * p_B / P_NORM
        Re = rho_B * v_B * D / GAS_MU

        if Re < 2300:
            return 64.0 / max(Re, 1e-3)
        else:
            return 0.25 / (np.log10(eps / (3.7 * D) + 5.74 / Re**0.9))**2

    def _gas_resistance(self, label: str, flow: float) -> float:
        """
        Linearisierter hydraulischer Widerstand [Pa·s/Nm³].

        Darcy-Weisbach für kompressibles Gas:

          ND  (p_B ≈ p_N, inkompressibel):
              Δp  ≈ f · L/D · ρ_N · Q_N² / (2A²)
              R_lin = f · L · ρ_N · Q₀ / (D · A²)

          MD/HD (kompressibel, linearisiert über mittleren Druck p_m):
              Δ(p²) ≈ f · L/D · ρ_N · p_N · Q_N² / A²
              R_lin = f · L · ρ_N · p_N · Q₀ / (D · A² · p_m)

          HD:  Weymouth-Reibungszahl λ_W = 0.009407 · D^(-1/3)
               anstelle des Moody-f (bewährte Näherung für HD-Gas, Strelow)
        """
        p_data = self._pipe_params[label]
        L, D = p_data['length'], p_data['diameter']
        A = np.pi * D**2 / 4
        Q0 = max(abs(flow), 1e-9)

        p_m = self._typical_pressure() or P_NORM

        if self._pressure_level == PressureLevel.ND:
            f = self._moody_friction(label, Q0, p_m)
            return float(f * L * self._rho_n * Q0 / (D * A**2))

        elif self._pressure_level == PressureLevel.MD:
            f = self._moody_friction(label, Q0, p_m)
            return float(f * L * self._rho_n * P_NORM * Q0 / (D * A**2 * p_m))

        else:  # HD — Weymouth-Reibungszahl
            lam_w = 0.009407 * D**(-1/3)
            return float(lam_w * L * self._rho_n * P_NORM * Q0 / (D * A**2 * p_m))

    def _typical_pressure(self) -> float | None:
        """Mittleren Einspeisedruck für Linearisierung schätzen."""
        if self._feed_pressures:
            return float(np.mean(list(self._feed_pressures.values())))
        return None

    def _pressure_drop(self, label: str, flow: float,
                       p_mean: float) -> float:
        """Druckverlust [Pa] nach Darcy-Weisbach."""
        p_data = self._pipe_params[label]
        L, D = p_data['length'], p_data['diameter']
        A = np.pi * D**2 / 4
        Q = abs(flow)
        if Q < 1e-9:
            return 0.0
        f = self._moody_friction(label, Q, p_mean)
        v_B = Q * P_NORM / (max(p_mean, 1e3) * A)
        rho_B = self._rho_n * max(p_mean, 1e3) / P_NORM
        return float(f * L / D * rho_B * v_B**2 / 2.0)

    def _velocity(self, label: str, flow: float,
                  pressure_pa: float) -> float:
        """Betriebsgeschwindigkeit [m/s] bei gegebenem Betriebsdruck."""
        p_data = self._pipe_params[label]
        D = p_data['diameter']
        A = np.pi * D**2 / 4
        Q_B = abs(flow) * P_NORM / max(pressure_pa, P_NORM)
        return Q_B / A

    # ------------------------------------------------------------------
    # Hydraulische Berechnung
    # ------------------------------------------------------------------

    def solve_hydraulic(self,
                        max_iter: int = 50,
                        tol: float = 1e-7) -> GasHydraulicResult:
        """
        Iterative hydraulische Netzberechnung.

        Algorithmus (analog Strelow & Kouka 2025, Kap. 3.2):
          1. Initiallösung mit geschätzten linearisierten Widerständen
          2. Widerstände aus aktuellem Normvolumenstrom aktualisieren
          3. Wiederholen bis ||ΔQ_N||_∞ < tol [Nm³/s]

        Einspeiseknoten = Referenzknoten mit vorgegebenem Druck.
        Verdichter erhöhen den Druck am Ausgangsknoten.
        """
        if self._dirty:
            self.build_matrices()

        flows = {e[2]: 1e-3 for e in self._edges}
        converged = False
        result = None

        for iteration in range(max_iter):
            for label in [e[2] for e in self._edges]:
                if label in self._pipe_params and label not in self._compressor_ratios:
                    self._resistances[label] = self._gas_resistance(
                        label, flows.get(label, 1e-3))

            self.build_matrices()

            ref_node = (list(self._feed_pressures.keys())[0]
                        if self._feed_pressures else None)
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

        # Drücke aus ATN-Potenzialen + Einspeise-Anker
        p_offset = 0.0
        if self._feed_pressures and result is not None:
            ref = list(self._feed_pressures.keys())[0]
            p_offset = self._feed_pressures[ref] - result.potentials.get(ref, 0.0)

        pressures = {n: max(result.potentials[n] + p_offset, 0.0)
                     for n in self._nodes}

        # Verdichter-Druckerhöhung
        for label, ratio in self._compressor_ratios.items():
            edge = next((e for e in self._edges if e[2] == label), None)
            if edge:
                to_node = edge[1]
                pressures[to_node] = pressures.get(to_node, 0.0) * ratio

        pressures_bar = {n: p / 1e5 for n, p in pressures.items()}

        # Mitteldruck pro Leitung für Geschwindigkeit und Druckverlust
        pressure_drops = {}
        velocities = {}
        for e in self._edges:
            if e[2] not in self._pipe_params:
                continue
            p_from = pressures.get(e[0], P_NORM)
            p_to   = pressures.get(e[1], P_NORM)
            p_mid  = max((p_from + p_to) / 2.0, P_NORM)
            pressure_drops[e[2]] = self._pressure_drop(e[2], flows.get(e[2], 0), p_mid)
            velocities[e[2]] = self._velocity(e[2], flows.get(e[2], 0), p_mid)

        # DVGW-Druckverletzungen
        violations = []
        for n, p_pa in pressures.items():
            min_p = self._min_pressures.get(n, self._default_min_pressure())
            if p_pa < min_p:
                violations.append(n)

        # Nm³/s → Nm³/h für Ausgabe
        flows_m3h = {k: v * 3600.0 for k, v in flows.items()}

        return GasHydraulicResult(
            pressures=pressures,
            pressures_bar=pressures_bar,
            flows_norm=flows_m3h,
            velocities=velocities,
            pressure_drops={k: v / 100.0 for k, v in pressure_drops.items()},  # Pa → mbar
            pressure_level=self._pressure_level,
            pressure_violations=violations,
            iterations=iteration + 1,
            converged=converged,
        )

    def _default_min_pressure(self) -> float:
        """DVGW-Mindestdrücke nach Druckstufe."""
        return {
            PressureLevel.ND: 17e2,   # 17 mbar (DVGW G 600)
            PressureLevel.MD: 20e2,   # 20 mbar (ND-Abgabedruck)
            PressureLevel.HD: 1e5,    # 1 bar
        }.get(self._pressure_level, 1e3)

    # ------------------------------------------------------------------
    # Linienspeicher (Linepack)
    # ------------------------------------------------------------------

    def calc_linepack(self, result: GasHydraulicResult) -> list[LinepackResult]:
        """
        Gasinhalt der Leitungen bei Betriebsdruck (stationär).

        Linepack [Nm³] = V_geo · p_B / p_N
        Relevant für Kurzzeitspeicher und Lastausgleich.
        """
        linepacks = []
        for e in self._edges:
            label = e[2]
            if label not in self._pipe_params:
                continue
            p_data = self._pipe_params[label]
            D, L = p_data['diameter'], p_data['length']
            V_geo = np.pi * D**2 / 4 * L

            p_from = result.pressures.get(e[0], P_NORM)
            p_to   = result.pressures.get(e[1], P_NORM)
            p_mean = (p_from + p_to) / 2.0

            vol_norm = V_geo * p_mean / P_NORM
            linepacks.append(LinepackResult(
                pipe_label=label,
                volume_norm_m3=float(vol_norm),
                pressure_pa=float(p_mean),
            ))
        return sorted(linepacks, key=lambda x: x.volume_norm_m3, reverse=True)

    # ------------------------------------------------------------------
    # DVGW-Prüfung
    # ------------------------------------------------------------------

    def check_dvgw(self, result: GasHydraulicResult) -> dict:
        """
        DVGW G 600 / G 462: Versorgungsdruckprüfung an allen Knoten.

        Mindestdrücke:
          ND: 17 mbar Abgabedruck (DVGW G 600, Abschnitt 8)
          MD: 20 mbar (Abgabedruck auf ND-Seite des Druckreglers)
          HD: 1 bar   (typischer Netzbetriebsdruck)

        Returns:
            dict mit 'ok', 'violations', 'details', 'summary'
        """
        ok = len(result.pressure_violations) == 0
        details = {}
        for n, p_pa in result.pressures.items():
            min_p = self._min_pressures.get(n, self._default_min_pressure())
            details[n] = {
                'pressure_bar': p_pa / 1e5,
                'pressure_mbar': p_pa / 100.0,
                'min_mbar': min_p / 100.0,
                'status': 'OK' if p_pa >= min_p else 'VERLETZUNG',
            }
        v = len(result.pressure_violations)
        return {
            'ok': ok,
            'violations': result.pressure_violations,
            'details': details,
            'summary': (
                f"DVGW {result.pressure_level}-Netz: "
                f"{'Alle Knoten OK' if ok else f'{v} Verletzung(en)'}"
            ),
        }
