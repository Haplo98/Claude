"""
Fernwärmenetz — hydraulische und thermische Berechnung.

Implementierung nach Strelow & Kouka, THM-Hochschulschriften Band 35 (2025).

Zwei gekoppelte Teilprobleme:
  1. Hydraulik:  Druckprofil + Massestromverteilung (nichtlinear → iterativ linearisiert)
  2. Thermik:    Temperaturprofile stationär + dynamisch (Wärmewelle)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.network import ATNNetwork


@dataclass
class HydraulicResult:
    pressures: dict[str, float]      # Pa
    mass_flows: dict[str, float]     # kg/s
    flow_regimes: dict[str, str]     # 'laminar' | 'turbulent' | 'creeping'
    iterations: int
    converged: bool


@dataclass
class ThermalResult:
    temperatures: dict[str, float]   # °C
    heat_losses: dict[str, float]    # W pro Leitung


@dataclass
class DynamicThermalResult:
    time_axis: np.ndarray                    # s
    temperature_history: dict[str, np.ndarray]  # Knoten → Zeitreihe [°C]


# Fluid-Eigenschaften Heizwasser (ca. 80°C Vorlauf)
WATER_RHO = 972.0     # kg/m³
WATER_CP  = 4196.0    # J/(kg·K)
WATER_MU  = 3.55e-4   # Pa·s (dynamische Viskosität)
LAMBDA_INS = 0.035    # W/(m·K) Rohrdämmung (PUR)
T_GROUND  = 10.0      # °C Erdreichtemperatur


class DistrictHeatingNetwork(ATNNetwork):
    """
    Fernwärmenetz nach Strelow & Kouka (2025).

    Modelliert das Vorlaufnetz. Das Rücklaufnetz wird symmetrisch behandelt.
    Physikalisches Potenzial: Druck [Pa] (hydraulisch) / Temperatur [°C] (thermisch).
    """

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._pipe_params: dict[str, dict] = {}   # label → Rohrparameter
        self._source_temps: dict[str, float] = {} # Knoten → Einspeistemp. [°C]
        self._heat_loads: dict[str, float] = {}   # Knoten → Wärmelast [W]
        self._return_temps: dict[str, float] = {} # Knoten → Rücklauftemp. [°C]

    # ------------------------------------------------------------------
    # Netz aufbauen
    # ------------------------------------------------------------------

    def add_pipe(self,
                 from_node: str, to_node: str,
                 length: float,
                 diameter: float,
                 roughness: float = 1.0e-4,
                 insulation_thickness: float = 0.05,
                 label: str | None = None) -> DistrictHeatingNetwork:
        """
        Rohrleitung hinzufügen.

        Args:
            length              : Länge [m]
            diameter            : Innendurchmesser [m]
            roughness           : Wandrauheit [m] (Standard: 0.1 mm)
            insulation_thickness: Dämmdicke [m]
        """
        if label is None:
            label = f"R{len(self._edges) + 1}"

        self._pipe_params[label] = {
            'length': length,
            'diameter': diameter,
            'roughness': roughness,
            'insulation_thickness': insulation_thickness,
        }

        # Initialen hydraulischen Widerstand schätzen (bei v=1 m/s)
        R_init = self._pipe_resistance(label, mass_flow=1.0)
        self.add_edge(from_node, to_node, R_init, label)
        return self

    def add_heat_source(self, node: str,
                        supply_temp: float,
                        mass_flow: float) -> DistrictHeatingNetwork:
        """
        Wärmequelle (Heizwerk, BHKW, Geothermie...).

        Args:
            supply_temp: Vorlauftemperatur [°C]
            mass_flow  : Massenstrom [kg/s] (positiv = Einspeisung)
        """
        self._source_temps[node] = supply_temp
        self.add_node(node, external_flow=mass_flow)
        return self

    def add_consumer(self, node: str,
                     heat_load: float,
                     return_temp: float = 55.0) -> DistrictHeatingNetwork:
        """
        Verbraucher (Hausanschluss, Industrieabnehmer...).

        Args:
            heat_load  : Wärmelast [W]
            return_temp: Rücklauftemperatur [°C]
        """
        self._heat_loads[node] = heat_load
        self._return_temps[node] = return_temp
        # Massenstrom aus Wärmelast und Temperaturspreizung berechnen
        # ṁ = Q̇ / (cp · ΔT) — initiale Schätzung
        T_supply = list(self._source_temps.values())[0] if self._source_temps else 80.0
        delta_T = max(T_supply - return_temp, 1.0)
        m_dot = -heat_load / (WATER_CP * delta_T)  # negativ = Entnahme
        self.add_node(node, external_flow=m_dot)
        return self

    # ------------------------------------------------------------------
    # Hydraulische Berechnung
    # ------------------------------------------------------------------

    def _pipe_resistance(self, label: str, mass_flow: float) -> float:
        """
        Hydraulischer Widerstand R_hyd [Pa/(kg/s)] einer Rohrleitung.

        Druckverlustformel: Δp = R_hyd · ṁ
        (linearisiert um Arbeitspunkt mass_flow für ATN-Framework)

        Strömungsregime nach Strelow & Kouka (2025), Kap. 2.2:
          Re < 100   : Kriechströmung  → f = 64/Re (Stokes)
          Re < 2300  : laminare Strömung → f = 64/Re
          Re ≥ 2300  : turbulente Strömung → Swamee-Jain-Näherung
        """
        p = self._pipe_params[label]
        L, D, eps = p['length'], p['diameter'], p['roughness']
        A = np.pi * D**2 / 4

        m_dot = max(abs(mass_flow), 1e-6)
        v = m_dot / (WATER_RHO * A)
        Re = WATER_RHO * v * D / WATER_MU

        if Re < 2300:
            f = 64 / max(Re, 1e-3)
            regime = 'laminar' if Re >= 100 else 'creeping'
        else:
            # Swamee-Jain (explizite Näherung der Colebrook-White-Gleichung)
            f = 0.25 / (np.log10(eps / (3.7 * D) + 5.74 / Re**0.9))**2
            regime = 'turbulent'

        # Linearisierter Druckverlust: Δp ≈ f·L/D · ρv²/2 ≈ R_hyd · ṁ
        # R_hyd = f·L/(D·A²·2·ρ) · ṁ  → bei Iteration aktualisiert
        R_hyd = f * L / (D * A**2 * 2 * WATER_RHO)
        return float(R_hyd * m_dot)  # linearisiert um Arbeitspunkt

    def _flow_regime(self, label: str, mass_flow: float) -> str:
        p = self._pipe_params[label]
        D, A = p['diameter'], np.pi * p['diameter']**2 / 4
        v = abs(mass_flow) / (WATER_RHO * A) if abs(mass_flow) > 1e-9 else 0
        Re = WATER_RHO * v * D / WATER_MU
        if Re < 100:
            return 'creeping'
        elif Re < 2300:
            return 'laminar'
        return 'turbulent'

    def solve_hydraulic(self,
                        max_iter: int = 30,
                        tol: float = 1e-5) -> HydraulicResult:
        """
        Iterative hydraulische Netzberechnung.

        Algorithmus (Strelow & Kouka 2025, Kap. 3.2):
          1. Initiallösung mit geschätzten Widerständen
          2. Widerstände aus aktuellen Massenströmen aktualisieren
          3. Wiederholen bis Konvergenz (||Δṁ||_∞ < tol)

        Behandelt Strangnetze, gering und stark vermaschte Netze einheitlich.
        """
        if self._dirty:
            self.build_matrices()

        flows = {e[2]: 1.0 for e in self._edges}  # Startwert
        converged = False

        for iteration in range(max_iter):
            # Widerstände aktualisieren
            for label in [e[2] for e in self._edges]:
                if label in self._pipe_params:
                    self._resistances[label] = self._pipe_resistance(
                        label, flows.get(label, 1.0))

            self.build_matrices()
            result = self.solve()
            flows_new = result.flows

            # Konvergenzprüfung
            max_change = max(
                abs(flows_new.get(l, 0) - flows.get(l, 0))
                for l in flows_new
            )
            flows = flows_new

            if max_change < tol:
                converged = True
                break

        regimes = {
            label: self._flow_regime(label, flows.get(label, 0))
            for label in [e[2] for e in self._edges]
            if label in self._pipe_params
        }

        return HydraulicResult(
            pressures=result.potentials,
            mass_flows=flows,
            flow_regimes=regimes,
            iterations=iteration + 1,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Stationäres Temperaturprofil
    # ------------------------------------------------------------------

    def solve_thermal_stationary(self,
                                  hyd: HydraulicResult) -> ThermalResult:
        """
        Stationäres Temperaturprofil (Strelow & Kouka 2025, Kap. 4.3).

        Temperaturabfall entlang Rohr:
            T(L) = T_Erde + (T_Ein - T_Erde) · exp(-k_L · L / (ṁ · cp))

        k_L [W/(m·K)]: Wärmeverlustkoeffizient pro Meter (aus Dämmgeometrie)
        """
        node_temps: dict[str, float] = dict(self._source_temps)
        heat_losses: dict[str, float] = {}

        # Topologische Sortierung: von Quellen zu Verbrauchern
        visited = set(self._source_temps.keys())
        queue = list(self._source_temps.keys())

        while queue:
            current = queue.pop(0)
            T_in = node_temps.get(current, T_GROUND)

            for from_n, to_n, label in self._edges:
                if from_n != current or label not in self._pipe_params:
                    continue
                if to_n in visited:
                    continue

                p = self._pipe_params[label]
                m_dot = abs(hyd.mass_flows.get(label, 0))

                if m_dot < 1e-9:
                    node_temps[to_n] = T_GROUND
                    heat_losses[label] = 0.0
                    visited.add(to_n)
                    queue.append(to_n)
                    continue

                # Wärmeverlustkoeffizient k_L [W/(m·K)]
                r_i = p['diameter'] / 2
                r_o = r_i + p['insulation_thickness']
                R_ins = np.log(r_o / r_i) / (2 * np.pi * LAMBDA_INS)
                k_L = 1.0 / R_ins

                # Temperaturabfall
                T_out = (T_GROUND
                         + (T_in - T_GROUND)
                         * np.exp(-k_L * p['length'] / (m_dot * WATER_CP)))

                node_temps[to_n] = T_out
                heat_losses[label] = m_dot * WATER_CP * (T_in - T_out)

                visited.add(to_n)
                queue.append(to_n)

        return ThermalResult(temperatures=node_temps, heat_losses=heat_losses)

    # ------------------------------------------------------------------
    # Dynamisches Temperaturprofil (Wärmewelle)
    # ------------------------------------------------------------------

    def solve_thermal_dynamic(self,
                               hyd: HydraulicResult,
                               T_initial: dict[str, float],
                               dt: float,
                               t_end: float,
                               source_temp_profile: dict[str, callable] | None = None
                               ) -> DynamicThermalResult:
        """
        Instationäres Temperaturprofil (Strelow & Kouka 2025, Kap. 4.4).

        Modelliert die Wärmewelle bei Lastwechsel oder Quellenänderung.
        Verwendet explizite Euler-Diskretisierung mit CFL-Stabilitätsprüfung.

        Args:
            hyd              : Hydraulisches Ergebnis (Massenströme)
            T_initial        : Anfangstemperaturen [°C] je Knoten
            dt               : Zeitschritt [s]
            t_end            : Simulationsende [s]
            source_temp_profile: Optionale Zeitfunktionen für Quelltemperaturen
                                 {node: callable(t) → °C}
        """
        time_steps = int(t_end / dt)
        T = {node: T_initial.get(node, T_GROUND) for node in self._nodes}
        history = {node: [T[node]] for node in self._nodes}

        for step in range(time_steps):
            t = step * dt
            T_new = dict(T)

            # Quelltemperaturen aktualisieren
            if source_temp_profile:
                for node, func in source_temp_profile.items():
                    T_new[node] = func(t)
            else:
                for node, T_src in self._source_temps.items():
                    T_new[node] = T_src

            for from_n, to_n, label in self._edges:
                if label not in self._pipe_params:
                    continue

                p = self._pipe_params[label]
                m_dot = hyd.mass_flows.get(label, 0)

                if abs(m_dot) < 1e-9:
                    continue

                A = np.pi * p['diameter']**2 / 4
                v = abs(m_dot) / (WATER_RHO * A)
                courant = v * dt / p['length']  # CFL-Zahl

                T_upstream = T.get(from_n, T_GROUND) if m_dot > 0 else T.get(to_n, T_GROUND)
                T_downstream_node = to_n if m_dot > 0 else from_n

                if courant >= 1.0:
                    # Volle Advektion: Temperatur vollständig transportiert
                    T_new[T_downstream_node] = T_upstream
                else:
                    # Teiladvektion
                    T_old_down = T.get(T_downstream_node, T_GROUND)
                    T_new[T_downstream_node] = ((1 - courant) * T_old_down
                                                + courant * T_upstream)

            T = T_new
            for node in self._nodes:
                history[node].append(T[node])

        return DynamicThermalResult(
            time_axis=np.arange(time_steps + 1) * dt,
            temperature_history={node: np.array(vals)
                                  for node, vals in history.items()},
        )
