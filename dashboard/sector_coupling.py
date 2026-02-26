"""
Sektorenkopplung-Tab für das ATN Dashboard.

Demonstriert das CouplingModel anhand eines kommunalen Energiesystems:
  BHKW + Wärmepumpe + Spitzenlastkessel + Stromnetz

5 Variablen, 3 Bilanzen → Freiheitsgrad d = 2
→ 2D-Entscheidungsraum (Hexagon, Strelow 2024 Abb. 2) visualisierbar
"""
from __future__ import annotations
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atn"))
from atn.coupling.model import CouplingModel


# ── Farbschema ────────────────────────────────────────────────────────────────
C_FEASIBLE  = "#3498DB"
C_OPTIMAL   = "#E74C3C"
C_BOUND_VAR = "#E74C3C"
C_BOUND_DEP = "#7B2D8B"
C_GRID      = "#DEE2E6"


def build_model(
    P_el_bedarf: float = 150.0,
    Q_hz_bedarf: float = 350.0,
    eta_el:      float = 0.38,
    eta_hz:      float = 0.50,
    P_bhkw_max:  float = 300.0,
    COP:         float = 3.5,
    P_wp_max:    float = 150.0,
    eta_kessel:  float = 0.92,
    Q_k_max:     float = 400.0,
    P_netz_max:  float = 200.0,
) -> CouplingModel:
    """
    Kopplungsmodell: BHKW + Wärmepumpe + Spitzenlastkessel + Stromnetz.

    Bilanzgleichungen:
      Strom:   P_BHKW  − P_WP + P_Netz                        = P_el_bedarf
      Wärme:   P_BHKW·(η_hz/η_el) + P_WP·COP + Q_Kessel      = Q_hz_bedarf
      Erdgas:  P_BHKW/η_el + Q_Kessel/η_kessel − V_Gas        = 0

    5 Variablen − 3 Bilanzen → Freiheitsgrad d = 2
    """
    m = CouplingModel("Kommunales Energiesystem Gießen")

    m.add_variable("P_BHKW",   "kW", lower=0,          upper=P_bhkw_max,
                   description="BHKW elektr. Leistung")
    m.add_variable("P_WP",     "kW", lower=0,          upper=P_wp_max,
                   description="Wärmepumpe elektr. Leistung")
    m.add_variable("Q_Kessel", "kW", lower=0,          upper=Q_k_max,
                   description="Spitzenlastkessel Wärmeleistung")
    m.add_variable("P_Netz",   "kW", lower=-P_netz_max, upper=P_netz_max,
                   description="Strombezug (+) / Einspeisung (−)")
    m.add_variable("V_Gas",    "kW", lower=0,          upper=float("inf"),
                   description="Gesamtgasleistung (thermisch)")

    m.add_balance("Elektrizität",
                  {"P_BHKW": 1.0, "P_WP": -1.0, "P_Netz": 1.0},
                  rhs=P_el_bedarf,
                  description="Strombilanz")

    m.add_balance("Wärme",
                  {"P_BHKW": eta_hz / eta_el, "P_WP": COP, "Q_Kessel": 1.0},
                  rhs=Q_hz_bedarf,
                  description="Wärmebilanz")

    m.add_balance("Erdgas",
                  {"P_BHKW": 1.0 / eta_el, "Q_Kessel": 1.0 / eta_kessel,
                   "V_Gas": -1.0},
                  rhs=0.0,
                  description="Gasbilanz")

    return m


def plot_decision_space(
    model: CouplingModel,
    opt_result=None,
    n_samples: int = 55,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    2D-Entscheidungsraum (Strelow 2024, Abb. 2).

    Zeigt den zulässigen Bereich im Raum der beiden Entscheidungsgrößen
    (P_BHKW × P_WP) als gefülltes Polygon + optimalen Betriebspunkt.
    """
    analysis = model.analyze()
    if analysis.dof < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"dof = {analysis.dof} < 2\n(kein 2D-Raum darstellbar)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    x_var = analysis.decision_variables[0]
    y_var = analysis.decision_variables[1]

    # ── Zulässige Punkte berechnen ────────────────────────────────────────────
    space = model.decision_space_2d(x_var, y_var, n_samples=n_samples)
    fx = np.array(space["feasible_x"])
    fy = np.array(space["feasible_y"])

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.set_facecolor("#F8F9FA")

    if len(fx) >= 3:
        # Konvexe Hülle → sauberes Polygon statt Punktewolke
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([fx, fy])
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    alpha=0.25, color=C_FEASIBLE)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=C_FEASIBLE, lw=2.5, label="Zulässiger Bereich")
        except Exception:
            ax.scatter(fx, fy, s=6, color=C_FEASIBLE, alpha=0.4,
                       label="Zulässige Punkte")
    else:
        ax.text(0.5, 0.5, "Kein zulässiger Bereich gefunden\n(Parameter prüfen)",
                ha="center", va="center", transform=ax.transAxes,
                color="red", fontsize=11)

    # ── Variablen-Grenzen einzeichnen ─────────────────────────────────────────
    x_v = next(v for v in model._variables if v.name == x_var)
    y_v = next(v for v in model._variables if v.name == y_var)

    ax.axvline(x_v.lower, color=C_BOUND_VAR, ls="--", lw=1.2, alpha=0.7,
               label=f"{x_var} Grenzen")
    if x_v.upper != float("inf"):
        ax.axvline(x_v.upper, color=C_BOUND_VAR, ls="--", lw=1.2, alpha=0.7)
    ax.axhline(y_v.lower, color=C_BOUND_DEP, ls=":", lw=1.2, alpha=0.7,
               label=f"{y_var} Grenzen")
    if y_v.upper != float("inf"):
        ax.axhline(y_v.upper, color=C_BOUND_DEP, ls=":", lw=1.2, alpha=0.7)

    # ── Optimaler Betriebspunkt ───────────────────────────────────────────────
    if opt_result and opt_result.status == "optimal":
        xo = opt_result.solution.get(x_var, 0)
        yo = opt_result.solution.get(y_var, 0)
        ax.scatter(xo, yo, color=C_OPTIMAL, s=200, zorder=6,
                   marker="*", edgecolors="white", linewidths=1.5,
                   label=f"Optimum ({xo:.1f} / {yo:.1f}) kW")
        x_range = float(np.ptp(fx)) if len(fx) > 0 else 100.0
        y_range = float(np.ptp(fy)) if len(fy) > 0 else 100.0
        ax.annotate(f"Optimum\n{opt_result.objective_value:.2f} EUR/h",
                    xy=(xo, yo),
                    xytext=(xo + max(x_range, 10) * 0.08,
                            yo + max(y_range, 10) * 0.08),
                    fontsize=9, color=C_OPTIMAL,
                    arrowprops=dict(arrowstyle="->", color=C_OPTIMAL, lw=1.5))

    ax.set_xlabel(space["x_label"], fontsize=11)
    ax.set_ylabel(space["y_label"], fontsize=11)
    ax.set_title(
        f"Entscheidungsraum — {model.name}\n"
        f"Rang = {analysis.rank}  |  Freiheitsgrad d = {analysis.dof}  |  "
        f"{space['n_feasible']} zulässige Betriebspunkte",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9)
    fig.tight_layout()
    return fig


def plot_solution_bars(
    model: CouplingModel,
    opt_result,
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Balkendiagramm der optimalen Lösung."""
    sol = opt_result.solution
    var_names = [v.name for v in model._variables]
    values    = [sol.get(n, 0) for n in var_names]
    analysis  = model.analyze()

    colors = []
    for n in var_names:
        if n in analysis.decision_variables:
            colors.append("#2980B9")   # Entscheidungsgröße — blau
        else:
            colors.append("#8E44AD")   # Folgegröße — lila

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.set_facecolor("#F8F9FA")
    bars = ax.bar(var_names, values, color=colors,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Werte über den Balken
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in values) * 0.02,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Leistung [kW]", fontsize=10)
    ax.set_title(f"Optimale Lösung — Betriebskosten: "
                 f"{opt_result.objective_value:.2f} EUR/h",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", color=C_GRID, lw=0.6, alpha=0.9, zorder=0)

    patches = [
        mpatches.Patch(color="#2980B9", label="Entscheidungsgröße (frei)"),
        mpatches.Patch(color="#8E44AD", label="Folgegröße (kausal)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_coupling_matrix_heatmap(
    model: CouplingModel,
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Heatmap der Kopplungsmatrix K mit GJ-Analyse."""
    K, B = model._build()
    analysis  = model.analyze()
    var_names = [v.name  for v in model._variables]
    bal_names = [b.name  for b in model._balances]

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=110)
    fig.patch.set_facecolor("#F8F9FA")

    def _draw(ax, matrix, title):
        vmax = max(abs(matrix).max(), 1e-9)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(var_names)))
        ax.set_xticklabels(var_names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(bal_names)))
        ax.set_yticklabels(bal_names, fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if abs(val) > 1e-10:
                    col = "white" if abs(val) / vmax > 0.55 else "black"
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center",
                            fontsize=8, color=col)
        # Pivot-Spalten (lila) und freie Spalten (blau)
        for j in analysis.gj.pivot_cols:
            for dx in (-0.48, 0.48):
                ax.axvline(j + dx, color="#8E44AD", lw=2, alpha=0.6)
        for j in analysis.gj.free_cols:
            for dx in (-0.48, 0.48):
                ax.axvline(j + dx, color="#2980B9", lw=2, alpha=0.6)
        ax.set_title(title, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    _draw(axes[0], K, "Kopplungsmatrix K")
    _draw(axes[1], analysis.gj.K_J, "GJ-Zeilenstufenform K_J")

    patches = [
        mpatches.Patch(color="#8E44AD", alpha=0.6, label="Folgegrößen (Pivot)"),
        mpatches.Patch(color="#2980B9", alpha=0.6, label="Entscheidungsgrößen (frei)"),
    ]
    fig.legend(handles=patches, fontsize=8, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Rang = {analysis.rank}  |  d = {analysis.dof}  |  "
                 f"V_e = {analysis.decision_variables}",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    return fig
