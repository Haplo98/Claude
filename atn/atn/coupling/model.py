"""
Kopplungsmodell für Sektorenkopplung — die innovativste Komponente des ATN-Frameworks.

Implementierung nach Strelow (2024): "Kopplungsmodelle — Mathematischer Hintergrund
der Sektorenkopplung", Institut für Thermodynamik, THM.

Kernidee: Sektorengekoppelte Energiesysteme sind inhärent unterbestimmt (d > 0).
Die Methode der unterbestimmten Kopplungsbilanzen:
  1. Formuliert das System als K · V + B = 0
  2. Trennt per Gauß-Jordan in Entscheidungsgrößen V_e und Folgegrößen V_f
  3. Macht den Entscheidungsraum geometrisch sichtbar
  4. Ermöglicht Optimierung innerhalb dieses Raums

Beispiel (Strelow 2024, Abb. 5-6): 5 Kleinkraftwerke, 4 Energieträger (Abwärme,
Heizöl, Erdgas, Elektrizität, Nahwärme) → 2 Freiheitsgrade → hexagonaler
Entscheidungsraum (Abb. 2).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linprog, minimize
from ..core.gauss_jordan import partial_gauss_jordan, GaussJordanResult


@dataclass
class Variable:
    """Systemvariable in einem Kopplungsmodell."""
    name: str
    unit: str = ""
    lower: float = 0.0
    upper: float = float('inf')
    description: str = ""


@dataclass
class Balance:
    """Bilanzgleichung (eine Zeile der Kopplungsmatrix K)."""
    name: str
    coefficients: dict[str, float]  # variable_name → Koeffizient
    rhs: float = 0.0                # rechte Seite (= -B_i)
    description: str = ""


@dataclass
class CouplingAnalysis:
    """Ergebnis der Gauß-Jordan-Analyse eines Kopplungsmodells."""
    rank: int
    dof: int
    decision_variables: list[str]   # V_e — frei wählbar
    dependent_variables: list[str]  # V_f — folgen aus V_e
    gj: GaussJordanResult


@dataclass
class OptimizationResult:
    status: str                      # 'optimal' | 'infeasible' | 'unbounded'
    solution: dict[str, float]
    objective_value: float
    decision_values: dict[str, float]
    dependent_values: dict[str, float]


class CouplingModel:
    """
    Kopplungsmodell für sektorengekoppelte Energieversorgungssysteme.

    Erlaubt die simultane Modellierung von Strom, Wärme, Gas und anderen
    Energieträgern in einem einheitlichen Matrixkalkül.

    Typische Anwendung (kommunales Versorgungssystem):
        model = CouplingModel("Stadtwerk Musterstadt")

        # Systemvariablen (eine pro Anlage)
        model.add_variable("P_Kleinkraftwerk", "kW", lower=0, upper=500)
        model.add_variable("Q_Oel_BHKW",       "kW", lower=0, upper=800)
        ...

        # Bilanzen (eine pro Energieträger)
        model.add_balance("Strom",   {"P_Kleinkraftwerk": 1, "Q_Oel_BHKW": eta_el, ...}, rhs=-P_netz)
        model.add_balance("Wärme",   {"Q_Oel_BHKW": eta_hz, ...}, rhs=-Q_bedarf)
        model.add_balance("Erdgas",  {...}, rhs=-V_gas_budget)

        analysis = model.analyze()
        # → dof=2: zwei Entscheidungsgrößen, Rest folgt kausal

        result = model.optimize({"Brennstoffkosten": costs})
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._variables: list[Variable] = []
        self._balances: list[Balance] = []
        self._analysis: CouplingAnalysis | None = None

    # ------------------------------------------------------------------
    # Modell aufbauen
    # ------------------------------------------------------------------

    def add_variable(self, name: str, unit: str = "",
                     lower: float = 0.0, upper: float = float('inf'),
                     description: str = "") -> CouplingModel:
        """Systemvariable hinzufügen (eine Spalte in K)."""
        self._variables.append(Variable(name, unit, lower, upper, description))
        self._analysis = None  # Cache invalidieren
        return self

    def add_balance(self, name: str,
                    coefficients: dict[str, float],
                    rhs: float = 0.0,
                    description: str = "") -> CouplingModel:
        """
        Bilanzgleichung hinzufügen (eine Zeile in K).

        coefficients: {Variablenname → Koeffizient in der Bilanz}
        rhs: rechte Seite (Vorgabewert, z.B. Bedarf oder Kontingent)

        Konvention (wie Strelow): K · V = -B, also rhs entspricht B_i.
        """
        self._balances.append(Balance(name, coefficients, rhs, description))
        self._analysis = None
        return self

    # ------------------------------------------------------------------
    # Matrizen aufbauen
    # ------------------------------------------------------------------

    def _build(self) -> tuple[np.ndarray, np.ndarray]:
        """Kopplungsmatrix K und Bilanzvektors B aufbauen."""
        n_bal = len(self._balances)
        n_var = len(self._variables)
        var_idx = {v.name: i for i, v in enumerate(self._variables)}

        K = np.zeros((n_bal, n_var))
        B = np.zeros(n_bal)

        for i, bal in enumerate(self._balances):
            for var_name, coeff in bal.coefficients.items():
                if var_name not in var_idx:
                    raise KeyError(f"Variable '{var_name}' nicht bekannt. "
                                   f"Bekannte Variablen: {list(var_idx.keys())}")
                K[i, var_idx[var_name]] = coeff
            # Konvention: K·V + B = 0, also B = -rhs
            # (rhs ist die rechte Seite der natürlichen Bilanzgleichung K·V = rhs)
            B[i] = -bal.rhs

        return K, B

    # ------------------------------------------------------------------
    # Gauß-Jordan-Analyse
    # ------------------------------------------------------------------

    def analyze(self) -> CouplingAnalysis:
        """
        Gauß-Jordan-Analyse: Trennung in Entscheidungs- und Folgegrößen.

        Gibt zurück:
          - Rang r des Systems
          - Freiheitsgrad d = n_variablen - r
          - Liste der Entscheidungsgrößen V_e (frei wählbar)
          - Liste der Folgegrößen V_f (kausal abhängig)

        Die Pivot-Spalten entsprechen den Folgegrößen (V_f),
        die freien Spalten den Entscheidungsgrößen (V_e).
        """
        if self._analysis is not None:
            return self._analysis

        K, B = self._build()
        gj = partial_gauss_jordan(K)

        var_names = [v.name for v in self._variables]
        self._analysis = CouplingAnalysis(
            rank=gj.rank,
            dof=gj.dof,
            decision_variables=[var_names[i] for i in gj.free_cols],
            dependent_variables=[var_names[i] for i in gj.pivot_cols],
            gj=gj,
        )
        return self._analysis

    # ------------------------------------------------------------------
    # Lösung für gegebene Entscheidungsgrößen
    # ------------------------------------------------------------------

    def solve(self, decision_values: dict[str, float]) -> dict[str, float]:
        """
        Berechnet alle Folgegrößen für vorgegebene Entscheidungsgrößen.

        Gleichungssystem (Strelow 2024, Gl. 14):
            V_f = -J · B - R_e · V_e

        wobei J die Jordan-Matrix und R_e die Restmatrix aus K_J ist.

        Args:
            decision_values: {Variablenname → Wert} für alle V_e

        Returns:
            Vollständiger Lösungsvektor {alle Variablennamen → Wert}
        """
        analysis = self.analyze()
        K, B = self._build()
        gj = analysis.gj

        n_f = len(gj.pivot_cols)
        n_e = len(gj.free_cols)
        var_names = [v.name for v in self._variables]

        # Entscheidungsvektor aufbauen
        V_e = np.array([decision_values[var_names[i]] for i in gj.free_cols])

        # Aus K_J: [E | R_e] · [V_f | V_e]^T = -J · B
        K_J_f = gj.K_J[:n_f, :][:, gj.pivot_cols]   # Einheitsmatrix-Teil
        K_J_e = gj.K_J[:n_f, :][:, gj.free_cols]    # Rest-Teil
        rhs = -gj.J[:n_f] @ B - K_J_e @ V_e

        V_f = np.linalg.solve(K_J_f, rhs)

        # Vollständigen Lösungsvektor zusammensetzen
        result = {}
        for i, idx in enumerate(gj.pivot_cols):
            result[var_names[idx]] = float(V_f[i])
        for i, idx in enumerate(gj.free_cols):
            result[var_names[idx]] = float(V_e[i])

        return result

    def is_feasible(self, solution: dict[str, float]) -> bool:
        """Prüft ob eine Lösung alle Variablengrenzen einhält."""
        for v in self._variables:
            val = solution.get(v.name)
            if val is None:
                return False
            if val < v.lower - 1e-6 or val > v.upper + 1e-6:
                return False
        return True

    # ------------------------------------------------------------------
    # Optimierung
    # ------------------------------------------------------------------

    def optimize(self,
                 objective: dict[str, float],
                 minimize_obj: bool = True) -> OptimizationResult:
        """
        Optimierung innerhalb des Entscheidungsraums.

        Minimiert (oder maximiert) eine lineare Zielfunktion über den
        Entscheidungsgrößen V_e, unter Einhaltung aller Variablengrenzen.

        Args:
            objective    : {Variablenname → Kostenkoeffizient}
            minimize_obj : True = Minimierung, False = Maximierung

        Algorithmus:
          - Lineare Zielfunktion → scipy.optimize.linprog (HiGHS-Solver)
          - Nichtlineare Zielfunktion → Erweiterung auf Hoek-Suche / evolutionär
            (Strelow 2024, Kap. 4)
        """
        analysis = self.analyze()
        gj = analysis.gj
        var_names = [v.name for v in self._variables]

        # Grenzen der Entscheidungsgrößen
        bounds_e = [(self._variables[idx].lower,
                     self._variables[idx].upper if self._variables[idx].upper != float('inf') else None)
                    for idx in gj.free_cols]

        # Ungleichungsrestriktionen aus Grenzen der Folgegrößen
        # V_f = A_e · V_e + b_f  (aus GJ-Zerlegung)
        K, B = self._build()
        n_f = len(gj.pivot_cols)
        K_J_f = gj.K_J[:n_f, :][:, gj.pivot_cols]
        K_J_e = gj.K_J[:n_f, :][:, gj.free_cols]
        b_f = -np.linalg.solve(K_J_f, gj.J[:n_f] @ B)
        A_e = -np.linalg.solve(K_J_f, K_J_e)

        # Reduzierte Kosten: c_e_red = c_f · A_e + c_e
        # Folgegrößen-Kosten auf Entscheidungsgrößen zurückrechnen (korrekte Sensitivität)
        c_full = np.array([objective.get(var_names[i], 0.0)
                           for i in range(len(self._variables))])
        c_f = c_full[gj.pivot_cols]
        c_e = c_full[gj.free_cols]
        c_e_red = c_f @ A_e + c_e
        if not minimize_obj:
            c_e_red = -c_e_red

        A_ub_list, b_ub_list = [], []
        for i, idx in enumerate(gj.pivot_cols):
            v = self._variables[idx]
            # upper: A_e[i] · V_e ≤ upper - b_f[i]
            if v.upper != float('inf'):
                A_ub_list.append(A_e[i])
                b_ub_list.append(v.upper - b_f[i])
            # lower: -A_e[i] · V_e ≤ -(lower - b_f[i]) = b_f[i] - lower
            if v.lower != -float('inf'):
                A_ub_list.append(-A_e[i])
                b_ub_list.append(b_f[i] - v.lower)

        A_ub = np.array(A_ub_list) if A_ub_list else None
        b_ub = np.array(b_ub_list) if b_ub_list else None

        lp_result = linprog(c_e_red, A_ub=A_ub, b_ub=b_ub, bounds=bounds_e,
                            method='highs')

        if not lp_result.success:
            return OptimizationResult(
                status='infeasible', solution={}, objective_value=float('inf'),
                decision_values={}, dependent_values={})

        decision_values = {var_names[gj.free_cols[i]]: float(lp_result.x[i])
                          for i in range(len(gj.free_cols))}
        full_solution = self.solve(decision_values)
        obj_val = sum(objective.get(k, 0) * v for k, v in full_solution.items())

        return OptimizationResult(
            status='optimal',
            solution=full_solution,
            objective_value=float(obj_val),
            decision_values=decision_values,
            dependent_values={var_names[idx]: full_solution[var_names[idx]]
                              for idx in gj.pivot_cols},
        )

    # ------------------------------------------------------------------
    # Entscheidungsraum visualisieren
    # ------------------------------------------------------------------

    def decision_space_2d(self,
                          x_var: str, y_var: str,
                          n_samples: int = 200
                          ) -> dict:
        """
        Berechnet den 2D-Entscheidungsraum für zwei Entscheidungsgrößen.

        Gibt die Eckpunkte des zulässigen Polygons zurück (wie Strelow 2024, Abb. 2).
        Nur sinnvoll wenn dof ≥ 2.

        Args:
            x_var, y_var: Namen der beiden Entscheidungsgrößen
            n_samples   : Auflösung der Grenzlinien-Berechnung

        Returns:
            dict mit 'feasible_points', 'boundary_x', 'boundary_y'
        """
        analysis = self.analyze()
        var_names = [v.name for v in self._variables]

        if x_var not in analysis.decision_variables:
            raise ValueError(f"'{x_var}' ist keine Entscheidungsgröße. "
                             f"Entscheidungsgrößen: {analysis.decision_variables}")

        x_v = next(v for v in self._variables if v.name == x_var)
        y_v = next(v for v in self._variables if v.name == y_var)

        x_range = np.linspace(x_v.lower, min(x_v.upper, x_v.lower + 1e6), n_samples)
        y_range = np.linspace(y_v.lower, min(y_v.upper, y_v.lower + 1e6), n_samples)

        # Feste Werte für weitere Entscheidungsgrößen (Mittelpunkt ihrer Grenzen)
        other_decisions = {}
        for name in analysis.decision_variables:
            if name not in (x_var, y_var):
                v = next(vv for vv in self._variables if vv.name == name)
                mid = (v.lower + min(v.upper, v.lower + 1e3)) / 2
                other_decisions[name] = mid

        feasible_x, feasible_y = [], []
        for x in x_range:
            for y in y_range:
                sol = self.solve({x_var: x, y_var: y, **other_decisions})
                if self.is_feasible(sol):
                    feasible_x.append(x)
                    feasible_y.append(y)

        return {
            'feasible_x': feasible_x,
            'feasible_y': feasible_y,
            'x_label': f"{x_var} [{x_v.unit}]",
            'y_label': f"{y_var} [{y_v.unit}]",
            'n_feasible': len(feasible_x),
        }

    # ------------------------------------------------------------------
    # Darstellung
    # ------------------------------------------------------------------

    def summary(self) -> str:
        analysis = self.analyze()
        lines = [
            f"Kopplungsmodell: '{self.name}'",
            f"  Variablen:          {len(self._variables)}",
            f"  Bilanzen:           {len(self._balances)}",
            f"  Rang:               {analysis.rank}",
            f"  Freiheitsgrad d:    {analysis.dof}",
            f"  Entscheidungsgrößen: {analysis.decision_variables}",
            f"  Folgegrößen:        {analysis.dependent_variables}",
        ]
        return "\n".join(lines)

    def print_matrix(self) -> None:
        """Kopplungsmatrix K formatiert ausgeben."""
        K, B = self._build()
        var_names = [v.name for v in self._variables]
        bal_names = [b.name for b in self._balances]

        col_w = max(len(n) for n in var_names + ["Bilanz"]) + 2
        header = f"{'Bilanz':<{col_w}}" + "".join(f"{n:>{col_w}}" for n in var_names) + f"{'| RHS':>{col_w}}"
        print(header)
        print("-" * len(header))
        for i, bal_name in enumerate(bal_names):
            row = f"{bal_name:<{col_w}}"
            for j in range(len(var_names)):
                val = K[i, j]
                row += f"{val:>{col_w}.3g}" if val != 0 else f"{'·':>{col_w}}"
            row += f"{self._balances[i].rhs:>{col_w}.3g}"
            print(row)
