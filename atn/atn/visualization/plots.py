"""
Visualisierungsmodul für das ATN-Framework.

Funktionen:
  plot_network()           - Netzwerk-Topologie mit Flüssen und Potenzialen
  plot_decision_space()    - 2D-Entscheidungsraum (Hexagon nach Strelow 2024, Abb. 2)
  plot_coupling_matrix()   - Heatmap der Kopplungsmatrix K
  plot_pressure_profile()  - Druckprofil Fernwärmenetz (Strelow & Kouka 2025)
  plot_temperature_profile()  - Stationäres Temperaturprofil
  plot_dynamic_temperatures() - Dynamische Temperaturzeitreihen (Wärmewelle)
  plot_validation_residuals() - Messabweichungen nach Validierung (Strelow & Dawitz 2020)

Abhängigkeiten: matplotlib (Pflicht), networkx (optional, für bessere Layouts)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from typing import Sequence

# networkx optional
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# ── Farb-Schema (konsistent über alle Plots) ────────────────────────────────
COLORS = {
    'node':       '#2E86AB',   # blau
    'source':     '#A23B72',   # magenta
    'consumer':   '#F18F01',   # orange
    'edge':       '#444444',
    'flow_pos':   '#27AE60',   # grün (positiver Fluss)
    'flow_neg':   '#E74C3C',   # rot (negativer Fluss)
    'feasible':   '#3498DB',   # blau (zulässig)
    'infeasible': '#E0E0E0',   # grau (unzulässig)
    'optimal':    '#E74C3C',   # rot (optimaler Punkt)
    'heat':       '#E74C3C',
    'pressure':   '#2E86AB',
    'background': '#F8F9FA',
    'grid':       '#DEE2E6',
}

FIGSIZE_DEFAULT = (10, 6)
DPI = 120


# ═══════════════════════════════════════════════════════════════════════════
# 1. NETZWERK-TOPOLOGIE
# ═══════════════════════════════════════════════════════════════════════════

def plot_network(network,
                 result=None,
                 pos: dict | None = None,
                 title: str | None = None,
                 potential_label: str = "U",
                 potential_unit: str = "V",
                 flow_label: str = "I",
                 flow_unit: str = "A",
                 figsize: tuple = FIGSIZE_DEFAULT,
                 ax: plt.Axes | None = None) -> plt.Figure:
    """
    Visualisiert die Netzwerk-Topologie mit Flüssen und Potenzialen.

    Knoten: farbkodiert nach Potenzialwert
    Kanten: Breite und Farbe nach Flussrichtung/-stärke
    Labels: Potenziale an Knoten, Flüsse an Kanten

    Args:
        network : ATNNetwork-Instanz
        result  : NetworkResult (optional, für farbkodierte Darstellung)
        pos     : {node_name: (x, y)} Knotenposition, sonst automatisch
        potential_label/unit: Bezeichnung der Potenzialart
        flow_label/unit     : Bezeichnung der Flussgröße
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
        ax.set_facecolor(COLORS['background'])

    nodes = network._nodes
    edges = network._edges
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}

    # --- Knotenposition bestimmen ---
    if pos is None:
        if HAS_NETWORKX:
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from([(f, t) for f, t, _ in edges])
            pos = nx.spring_layout(G, seed=42, k=2.0)
            pos = {n: np.array(p) for n, p in pos.items()}
        else:
            # Kreislayout als Fallback
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            pos = {nodes[i]: np.array([np.cos(angles[i]), np.sin(angles[i])])
                   for i in range(n)}

    # --- Potenzialwerte für Farbkodierung ---
    potentials = {}
    if result is not None:
        potentials = result.potentials

    pot_vals = [potentials.get(node, 0.0) for node in nodes]
    vmin, vmax = (min(pot_vals), max(pot_vals)) if pot_vals else (0, 1)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- Kanten zeichnen ---
    flows = result.flows if result is not None else {}
    flow_vals = [abs(flows.get(label, 0)) for _, _, label in edges]
    max_flow = max(flow_vals) if flow_vals else 1.0

    for from_n, to_n, label in edges:
        p1 = np.array(pos[from_n])
        p2 = np.array(pos[to_n])
        flow = flows.get(label, 0)

        # Linienbreite: proportional zum Betrag des Flusses
        lw = 1.0 + 4.0 * abs(flow) / max(max_flow, 1e-9)
        color = COLORS['flow_pos'] if flow >= 0 else COLORS['flow_neg']

        # Linie mit Pfeil
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=15))

        # Kanten-Label (Fluss + Leitungsname)
        mid = (p1 + p2) / 2
        perp = np.array([-(p2-p1)[1], (p2-p1)[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9) * 0.06
        if result is not None:
            txt = f"{label}\n{flow:.2f} {flow_unit}"
        else:
            txt = label
        ax.text(*(mid + perp), txt, fontsize=7, ha='center', va='center',
                color=COLORS['edge'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    # --- Knoten zeichnen ---
    node_radius = 0.08
    for node in nodes:
        xy = pos[node]
        pot = potentials.get(node, 0.0)
        color = cmap(norm(pot))
        ext_flow = network._external_flows.get(node, 0.0)

        # Rand: dicker bei Quellen/Senken
        ec = COLORS['source'] if ext_flow > 0 else (COLORS['consumer'] if ext_flow < 0 else COLORS['edge'])
        lw = 2.5 if ext_flow != 0 else 1.0

        circle = plt.Circle(xy, node_radius, color=color, ec=ec, lw=lw, zorder=3)
        ax.add_patch(circle)

        # Knoten-Label
        if result is not None:
            label_txt = f"{node}\n{pot:.2f} {potential_unit}"
        else:
            label_txt = node
        ax.text(*xy, label_txt, ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=4,
                color='white' if norm(pot) > 0.5 else 'black')

    # --- Legende für externe Flüsse ---
    ext_flows = {n: v for n, v in network._external_flows.items() if v != 0}
    if ext_flows:
        legend_txt = "Externe Flüsse:\n" + "\n".join(
            f"  {n}: {v:+.2f} {flow_unit}" for n, v in ext_flows.items())
        ax.text(0.02, 0.98, legend_txt, transform=ax.transAxes,
                fontsize=8, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # --- Farbbalken ---
    if result is not None and vmax > vmin:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f"{potential_label} [{potential_unit}]", fontsize=9)

    # Legende: Quellen / Senken
    patches = [
        mpatches.Patch(color=COLORS['source'], label='Quelle (ext. Einspeisung)'),
        mpatches.Patch(color=COLORS['consumer'], label='Senke (ext. Entnahme)'),
        mpatches.Patch(color=COLORS['flow_pos'], label=f'Fluss positiv'),
        mpatches.Patch(color=COLORS['flow_neg'], label=f'Fluss negativ'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)

    ax.set_title(title or f"ATN-Netz: {network.name}", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    # GJ-Info
    gj = network.analyze()
    info = f"Rang: {gj.rank}  |  Freiheitsgrad: {gj.dof}  |  Maschen: {len(gj.mesh_rows)}"
    ax.text(0.5, 0.01, info, transform=ax.transAxes,
            ha='center', fontsize=8, color='gray')

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENTSCHEIDUNGSRAUM (Kopplungsmodell)
# ═══════════════════════════════════════════════════════════════════════════

def plot_decision_space(coupling_model,
                        x_var: str,
                        y_var: str,
                        n_samples: int = 80,
                        optimal_point: dict | None = None,
                        test_points: list[dict] | None = None,
                        title: str | None = None,
                        figsize: tuple = (9, 7),
                        ax: plt.Axes | None = None) -> plt.Figure:
    """
    Visualisiert den 2D-Entscheidungsraum eines Kopplungsmodells.

    Reproduziert das farbige Hexagon aus Strelow (2024), Abb. 2:
    - Blaue Fläche: zulässige Betriebspunkte
    - Rote Grenzlinien: aus Variablengrenzen der Folgegrößen
    - Roter Punkt: optimaler Betriebspunkt (falls übergeben)
    - Graue Punkte: unzulässige Testpunkte

    Args:
        coupling_model: CouplingModel-Instanz
        x_var, y_var  : Namen der beiden dargestellten Entscheidungsgrößen
        optimal_point : {'x_var': wert, 'y_var': wert} optimaler Punkt
        test_points   : Liste von Testpunkten zum Einzeichnen
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
        ax.set_facecolor(COLORS['background'])

    analysis = coupling_model.analyze()
    var_names = [v.name for v in coupling_model._variables]

    # Variablen-Grenzen für Achsen
    x_v = next(v for v in coupling_model._variables if v.name == x_var)
    y_v = next(v for v in coupling_model._variables if v.name == y_var)
    x_lo = x_v.lower
    x_hi = min(x_v.upper, x_v.lower + 1500) if x_v.upper != float('inf') else x_v.lower + 1000
    y_lo = y_v.lower
    y_hi = min(y_v.upper, y_v.lower + 1500) if y_v.upper != float('inf') else y_v.lower + 1000

    # Andere Entscheidungsgrößen: Mittelwert ihrer Grenzen
    other_decisions = {}
    for name in analysis.decision_variables:
        if name not in (x_var, y_var):
            v = next(vv for vv in coupling_model._variables if vv.name == name)
            mid = (v.lower + min(v.upper, v.lower + 1000)) / 2
            other_decisions[name] = mid

    # Gitter berechnen
    xs = np.linspace(x_lo, x_hi, n_samples)
    ys = np.linspace(y_lo, y_hi, n_samples)
    XX, YY = np.meshgrid(xs, ys)
    feasible = np.zeros(XX.shape, dtype=bool)

    for i in range(n_samples):
        for j in range(n_samples):
            sol = coupling_model.solve({x_var: XX[i, j], y_var: YY[i, j],
                                        **other_decisions})
            feasible[i, j] = coupling_model.is_feasible(sol)

    # Zulässige Fläche füllen
    ax.contourf(XX, YY, feasible.astype(float),
                levels=[0.5, 1.5], colors=[COLORS['feasible']], alpha=0.35)
    ax.contour(XX, YY, feasible.astype(float),
               levels=[0.5], colors=[COLORS['feasible']], linewidths=2.0)

    # Variablengrenzen einzeichnen (Strelow 2024: rot-gestrichelt)
    ax.axvline(x=x_lo, color='red', ls='--', lw=1.2, alpha=0.6, label=f'{x_var}_min')
    if x_v.upper != float('inf'):
        ax.axvline(x=x_v.upper, color='red', ls='--', lw=1.2, alpha=0.6,
                   label=f'{x_var}_max')
    ax.axhline(y=y_lo, color='darkred', ls='--', lw=1.2, alpha=0.6,
               label=f'{y_var}_min')
    if y_v.upper != float('inf'):
        ax.axhline(y=y_v.upper, color='darkred', ls='--', lw=1.2, alpha=0.6,
                   label=f'{y_var}_max')

    # Grenzlinien der Folgegrößen (implizite Beschränkungen)
    _draw_dependent_bounds(ax, coupling_model, x_var, y_var, xs, ys,
                           other_decisions, x_lo, x_hi, y_lo, y_hi)

    # Testpunkte
    if test_points:
        for pt in test_points:
            xp, yp = pt.get(x_var, 0), pt.get(y_var, 0)
            sol = coupling_model.solve({x_var: xp, y_var: yp, **other_decisions})
            ok = coupling_model.is_feasible(sol)
            color = COLORS['flow_pos'] if ok else '#888888'
            ax.scatter(xp, yp, color=color, s=60, zorder=5, edgecolors='white', lw=1)

    # Optimaler Punkt
    if optimal_point:
        xo = optimal_point.get(x_var, 0)
        yo = optimal_point.get(y_var, 0)
        ax.scatter(xo, yo, color=COLORS['optimal'], s=120, zorder=6,
                   marker='*', edgecolors='white', lw=1.5, label='Optimum')
        ax.annotate(f"Optimum\n({xo:.1f}, {yo:.1f})",
                    xy=(xo, yo), xytext=(xo + (x_hi-x_lo)*0.05, yo + (y_hi-y_lo)*0.05),
                    fontsize=8, color=COLORS['optimal'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['optimal']))

    # Beschriftung
    n_feasible = int(feasible.sum())
    n_total = n_samples * n_samples
    ax.set_xlabel(f"{x_var} [{x_v.unit}]", fontsize=10)
    ax.set_ylabel(f"{y_var} [{y_v.unit}]", fontsize=10)
    ax.set_title(
        title or f"Entscheidungsraum: {coupling_model.name}\n"
                 f"(dof={analysis.dof}, {n_feasible}/{n_total} Rasterpunkte zulässig)",
        fontsize=11, fontweight='bold'
    )
    ax.grid(True, color=COLORS['grid'], lw=0.5, alpha=0.8)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Legende
    patches = [
        mpatches.Patch(color=COLORS['feasible'], alpha=0.4, label='Zulässiger Bereich'),
        mpatches.Patch(color='white', label=f'Folgegröße: {", ".join(analysis.dependent_variables)}'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


def _draw_dependent_bounds(ax, model, x_var, y_var, xs, ys,
                            other_decisions, x_lo, x_hi, y_lo, y_hi):
    """Grenzlinien der Folgegrößen als gestrichelte Linien einzeichnen."""
    analysis = model.analyze()
    colors_dep = plt.cm.tab10(np.linspace(0, 0.8, len(analysis.dependent_variables)))

    for dep_name, col in zip(analysis.dependent_variables, colors_dep):
        dep_v = next(v for v in model._variables if v.name == dep_name)

        # Grenzlinie: dep(x, y) = lower bzw. upper → Linie im x-y-Raum
        for bound_val, ls in [(dep_v.lower, ':'), (dep_v.upper, '-.')]:
            if bound_val in (float('inf'), -float('inf')):
                continue

            # Punkte auf der Grenzlinie suchen (x von x_lo bis x_hi, y berechnen)
            boundary_x, boundary_y = [], []
            for x in xs:
                # Binärsuche: finde y sodass dep(x, y) = bound_val
                sol_lo = model.solve({x_var: x, y_var: y_lo, **other_decisions})
                sol_hi = model.solve({x_var: x, y_var: y_hi, **other_decisions})
                v_lo = sol_lo.get(dep_name, 0)
                v_hi = sol_hi.get(dep_name, 0)

                if (v_lo - bound_val) * (v_hi - bound_val) < 0:
                    # Nullstelle zwischen y_lo und y_hi: Bisektion
                    b_lo, b_hi = y_lo, y_hi
                    v_b_lo = v_lo
                    for _ in range(20):
                        y_mid = (b_lo + b_hi) / 2
                        sol_mid = model.solve({x_var: x, y_var: y_mid, **other_decisions})
                        v_mid = sol_mid.get(dep_name, 0)
                        if (v_b_lo - bound_val) * (v_mid - bound_val) < 0:
                            b_hi = y_mid
                        else:
                            b_lo = y_mid
                            v_b_lo = v_mid
                    boundary_x.append(x)
                    boundary_y.append(y_mid)

            if len(boundary_x) > 1:
                label = f"{dep_name} = {bound_val:.0f}"
                ax.plot(boundary_x, boundary_y, color=col, ls=ls, lw=1.5,
                        alpha=0.7, label=label)


# ═══════════════════════════════════════════════════════════════════════════
# 3. KOPPLUNGSMATRIX HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_coupling_matrix(coupling_model,
                         title: str | None = None,
                         figsize: tuple = (10, 5),
                         show_gj: bool = True) -> plt.Figure:
    """
    Heatmap der Kopplungsmatrix K (und optional der GJ-Zeilenstufenform K_J).

    Pivot-Spalten (Folgegrößen) sind blau markiert,
    freie Spalten (Entscheidungsgrößen) sind grün markiert.

    Args:
        show_gj: Wenn True, K und K_J nebeneinander zeigen
    """
    K, B = coupling_model._build()
    analysis = coupling_model.analyze()
    var_names = [v.name for v in coupling_model._variables]
    bal_names = [b.name for b in coupling_model._balances]

    ncols = 2 if show_gj else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=DPI)
    if ncols == 1:
        axes = [axes]

    def _heatmap(ax, matrix, col_labels, row_labels, title_txt, pivot_cols, free_cols):
        vmax = max(abs(matrix).max(), 1e-9)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        # Werte in Zellen
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if abs(val) > 1e-10:
                    color = 'white' if abs(val) / vmax > 0.5 else 'black'
                    ax.text(j, i, f"{val:.3g}", ha='center', va='center',
                            fontsize=7, color=color)

        # Pivot-Spalten (blau) und freie Spalten (grün) markieren
        for j in pivot_cols:
            ax.axvline(x=j - 0.5, color='#2E86AB', lw=2, alpha=0.5)
            ax.axvline(x=j + 0.5, color='#2E86AB', lw=2, alpha=0.5)
        for j in free_cols:
            ax.axvline(x=j - 0.5, color='#27AE60', lw=2, alpha=0.5)
            ax.axvline(x=j + 0.5, color='#27AE60', lw=2, alpha=0.5)

        ax.set_title(title_txt, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6)

        # Legende
        patches = [
            mpatches.Patch(color='#2E86AB', alpha=0.5, label='Folgegroessen (pivot)'),
            mpatches.Patch(color='#27AE60', alpha=0.5, label='Entscheidungsgroessen (frei)'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=7)

    _heatmap(axes[0], K, var_names, bal_names,
             "Kopplungsmatrix K",
             analysis.gj.pivot_cols, analysis.gj.free_cols)

    if show_gj and len(axes) > 1:
        K_J = analysis.gj.K_J
        _heatmap(axes[1], K_J, var_names, bal_names,
                 "GJ-Zeilenstufenform K_J\n(Nullzeilen = Maschen)",
                 analysis.gj.pivot_cols, analysis.gj.free_cols)

    fig.suptitle(
        title or f"Kopplungsmatrix: {coupling_model.name}\n"
                 f"Rang={analysis.rank}, dof={analysis.dof}",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. FERNWÄRME: DRUCK- UND TEMPERATURPROFIL
# ═══════════════════════════════════════════════════════════════════════════

def plot_pressure_profile(dh_network,
                           hyd_result,
                           node_order: list[str] | None = None,
                           title: str | None = None,
                           figsize: tuple = FIGSIZE_DEFAULT,
                           ax: plt.Axes | None = None) -> plt.Figure:
    """
    Druckprofil eines Fernwärmenetzes (Strelow & Kouka 2025, Kap. 3.4).

    Zeigt statisches Druckprofil entlang der Netzachse:
    - Balken: Knotendruck [bar]
    - Farb-Gradient: von hohem (rot) zu niedrigem (blau) Druck
    - Druckverlust zwischen Knoten annotiert
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
        ax.set_facecolor(COLORS['background'])

    pressures = hyd_result.pressures
    if node_order is None:
        node_order = list(pressures.keys())

    p_vals = [pressures.get(n, 0) / 1e5 for n in node_order]  # Pa → bar
    p_max = max(p_vals)
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=min(p_vals), vmax=p_max)

    bars = ax.bar(range(len(node_order)), p_vals,
                  color=[cmap(norm(p)) for p in p_vals],
                  edgecolor='white', linewidth=1.5, width=0.6, zorder=3)

    # Druckverlust annotieren
    for i in range(len(node_order) - 1):
        dp = p_vals[i] - p_vals[i+1]
        mid_x = i + 0.5
        mid_y = (p_vals[i] + p_vals[i+1]) / 2 + p_max * 0.03
        if dp > 0.001:
            ax.annotate(f"Δp={dp*1000:.1f} mbar",
                        xy=(mid_x, mid_y), ha='center', fontsize=8,
                        color='#555555',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Massenstrom-Labels auf Balken
    mass_flows = hyd_result.mass_flows
    for i, node in enumerate(node_order):
        ax.text(i, p_vals[i] + p_max * 0.01,
                f"{p_vals[i]:.3f} bar",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#333333')

    ax.set_xticks(range(len(node_order)))
    ax.set_xticklabels(node_order, fontsize=10)
    ax.set_ylabel("Druck [bar]", fontsize=10)
    ax.set_title(title or f"Statisches Druckprofil — {dh_network.name}",
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=COLORS['grid'], lw=0.5, alpha=0.8)
    ax.set_ylim(0, p_max * 1.15)
    ax.axhline(y=0, color='black', lw=0.8)

    # Massenströme als Sekundärachse
    ax2 = ax.twinx()
    edge_flows = [abs(mass_flows.get(e[2], 0))
                  for e in dh_network._edges
                  if e[0] in node_order and e[1] in node_order]
    if edge_flows:
        edge_positions = [node_order.index(e[0]) + 0.5
                         for e in dh_network._edges
                         if e[0] in node_order and e[1] in node_order]
        ax2.plot(edge_positions, edge_flows, 'D--', color='#7B2D8B',
                 markersize=8, lw=1.5, label='Massenstrom [kg/s]', zorder=5)
        ax2.set_ylabel("Massenstrom [kg/s]", fontsize=9, color='#7B2D8B')
        ax2.tick_params(axis='y', colors='#7B2D8B')
        ax2.legend(loc='upper right', fontsize=8)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


def plot_temperature_profile(dh_network,
                              thermal_result,
                              node_order: list[str] | None = None,
                              title: str | None = None,
                              figsize: tuple = FIGSIZE_DEFAULT,
                              ax: plt.Axes | None = None) -> plt.Figure:
    """
    Stationäres Temperaturprofil eines Fernwärmenetzes (Strelow & Kouka 2025, Kap. 4.3).

    Zeigt Temperaturabfall entlang des Vorlaufnetzes:
    - Linie + Punkte: Temperaturen an Knoten
    - Schraffur: Temperaturdifferenz zur Erdreichtemperatur (Wärmeverlust)
    - Annotierungen: Wärmeverluste je Leitung
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
        ax.set_facecolor(COLORS['background'])

    temps = thermal_result.temperatures
    heat_losses = thermal_result.heat_losses

    if node_order is None:
        node_order = list(temps.keys())

    T_vals = [temps.get(n, 10.0) for n in node_order]
    T_ground = 10.0

    # Temperaturlinie
    ax.plot(range(len(node_order)), T_vals, 'o-',
            color=COLORS['heat'], lw=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=COLORS['heat'], zorder=4, label='Vorlauftemperatur')

    # Erdreichtemperatur-Referenz
    ax.axhline(y=T_ground, color='brown', ls=':', lw=1.5,
               alpha=0.6, label=f'Erdreichtemperatur ({T_ground}°C)')

    # Schraffierte Fläche: Übertemperatur → Wärme-Potenzial
    ax.fill_between(range(len(node_order)), T_vals, T_ground,
                    alpha=0.15, color=COLORS['heat'])

    # Temperaturen und Wärmeverluste annotieren
    for i, (node, T) in enumerate(zip(node_order, T_vals)):
        ax.text(i, T + 0.3, f"{T:.2f}°C",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['heat'])

    # Wärmeverluste zwischen Knoten
    edge_loss_pairs = []
    for from_n, to_n, label in dh_network._edges:
        if from_n in node_order and to_n in node_order and label in heat_losses:
            i1 = node_order.index(from_n)
            i2 = node_order.index(to_n)
            loss = heat_losses[label]
            edge_loss_pairs.append((i1, i2, label, loss))

    for i1, i2, label, loss in edge_loss_pairs:
        mid_x = (i1 + i2) / 2
        mid_y = (T_vals[i1] + T_vals[i2]) / 2 - 1.5
        ax.annotate(f"{label}\n{loss/1000:.2f} kW",
                    xy=(mid_x, mid_y), ha='center', fontsize=8,
                    color='#7B2D8B',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    total_loss = sum(heat_losses.values())
    ax.set_xticks(range(len(node_order)))
    ax.set_xticklabels(node_order, fontsize=10)
    ax.set_ylabel("Temperatur [°C]", fontsize=10)
    ax.set_title(
        title or f"Stationäres Temperaturprofil — {dh_network.name}\n"
                 f"Gesamtwärmeverlust: {total_loss/1000:.2f} kW",
        fontsize=11, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, color=COLORS['grid'], lw=0.5, alpha=0.8)
    ax.set_ylim(T_ground - 5, max(T_vals) * 1.08)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ═══════════════════════════════════════════════════════════════════════════
# 5. DYNAMISCHES TEMPERATURPROFIL (Wärmewelle)
# ═══════════════════════════════════════════════════════════════════════════

def plot_dynamic_temperatures(dynamic_result,
                               nodes: list[str] | None = None,
                               time_unit: str = 'min',
                               title: str | None = None,
                               figsize: tuple = (12, 5),
                               ax: plt.Axes | None = None) -> plt.Figure:
    """
    Zeitverlauf der Knotentemperaturen — die Wärmewelle (Strelow & Kouka 2025, Kap. 4.4).

    Zeigt, wie sich eine Temperaturänderung an der Quelle zeitverzögert
    zu den Verbrauchern ausbreitet.

    Args:
        dynamic_result: DynamicThermalResult
        nodes         : Auszuwählende Knoten (default: alle)
        time_unit     : 's', 'min' oder 'h'
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
        ax.set_facecolor(COLORS['background'])

    history = dynamic_result.temperature_history
    time_axis = dynamic_result.time_axis

    # Zeitachse konvertieren
    scale = {'s': 1, 'min': 1/60, 'h': 1/3600}.get(time_unit, 1)
    t = time_axis * scale

    if nodes is None:
        nodes = list(history.keys())

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.1, 0.9, len(nodes)))

    for node, color in zip(nodes, colors):
        T_hist = history.get(node, np.zeros_like(t))
        ax.plot(t, T_hist, lw=2.0, label=node, color=color)

        # Einschwingzeit markieren (10%-90% Anstieg)
        T_min, T_max = T_hist.min(), T_hist.max()
        if T_max - T_min > 1.0:
            T_10 = T_min + 0.1 * (T_max - T_min)
            T_90 = T_min + 0.9 * (T_max - T_min)
            idx_10 = np.argmax(T_hist > T_10)
            idx_90 = np.argmax(T_hist > T_90)
            if idx_10 > 0 and idx_90 > idx_10:
                t_10 = t[idx_10]
                t_90 = t[idx_90]
                ax.axvline(x=t_90, color=color, ls=':', lw=1, alpha=0.5)
                ax.text(t_90, T_min + 0.5, f"t90={t_90:.1f}{time_unit}",
                        rotation=90, fontsize=7, color=color, alpha=0.8,
                        va='bottom')

    ax.set_xlabel(f"Zeit [{time_unit}]", fontsize=10)
    ax.set_ylabel("Temperatur [°C]", fontsize=10)
    ax.set_title(
        title or "Dynamisches Temperaturprofil — Wärmeausbreitung im Netz",
        fontsize=11, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, color=COLORS['grid'], lw=0.5, alpha=0.8)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ═══════════════════════════════════════════════════════════════════════════
# 6. MESSDATEN-VALIDIERUNG
# ═══════════════════════════════════════════════════════════════════════════

def plot_validation_residuals(validator,
                               validation_result,
                               var_names: list[str] | None = None,
                               title: str | None = None,
                               figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Visualisiert Messabweichungen nach der Validierung (Strelow & Dawitz 2020).

    Zeigt für jeden gemessenen Sensor:
    - Messwert (blauer Balken)
    - Validierter Wert (roter Punkt)
    - Abweichung / Korrektur (Pfeil)
    - 2σ-Grenze für Ausreißerdetektion

    Args:
        validator         : SystemValidator-Instanz
        validation_result : ValidationResult
        var_names         : Variablenbezeichnungen
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(COLORS['background'])

    measurements = validator._measurements
    corrections  = validation_result.measurement_corrections
    x_val = validation_result.validated_state
    rms = validation_result.rms_correction

    # --- Links: Messwerte vs. validierte Werte ---
    ax1.set_facecolor(COLORS['background'])
    n_meas = len(measurements)
    x_pos = np.arange(n_meas)
    labels = [m.sensor_id or f"S{m.variable_index}" for m in measurements]
    y_meas = np.array([m.value for m in measurements])
    y_val  = y_meas - corrections

    ax1.bar(x_pos - 0.2, y_meas, width=0.35, color=COLORS['feasible'],
            alpha=0.7, label='Messwert', zorder=3)
    ax1.bar(x_pos + 0.2, y_val,  width=0.35, color=COLORS['heat'],
            alpha=0.7, label='Validierter Wert', zorder=3)

    # Pfeile für Korrekturen
    for i, (ym, yv) in enumerate(zip(y_meas, y_val)):
        if abs(ym - yv) > 1e-6:
            ax1.annotate("", xy=(i+0.2, yv), xytext=(i-0.2, ym),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel("Wert", fontsize=10)
    ax1.set_title("Messwerte vs. validierte Werte", fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', color=COLORS['grid'], lw=0.5, alpha=0.8)

    # --- Rechts: Korrekturen / Residuen ---
    ax2.set_facecolor(COLORS['background'])
    sigma = np.std(corrections)
    colors_bar = [COLORS['flow_neg'] if abs(c) > 2*sigma else COLORS['feasible']
                  for c in corrections]

    ax2.bar(x_pos, corrections, color=colors_bar, alpha=0.8, zorder=3)
    ax2.axhline(y=0, color='black', lw=0.8)
    ax2.axhline(y=2*sigma, color='red', ls='--', lw=1.5, alpha=0.7, label='2σ-Grenze')
    ax2.axhline(y=-2*sigma, color='red', ls='--', lw=1.5, alpha=0.7)
    ax2.fill_between([-0.5, n_meas-0.5], -2*sigma, 2*sigma,
                     alpha=0.08, color='green', label='Normalbereich')

    # Ausreißer annotieren
    for i, (c, label) in enumerate(zip(corrections, labels)):
        if abs(c) > 2*sigma:
            ax2.text(i, c + np.sign(c) * rms * 0.1, f"Ausreißer!\n{c:.3f}",
                    ha='center', va='bottom' if c > 0 else 'top',
                    fontsize=8, color='red', fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("Korrektur (Messwert − validiert)", fontsize=10)
    ax2.set_title(f"Messabweichungen\nRMS = {rms:.4f}", fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', color=COLORS['grid'], lw=0.5, alpha=0.8)

    fig.suptitle(title or "Messdaten-Validierung (Strelow & Dawitz 2020)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. KOMBINIERTER FERNWÄRME-DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_district_heating_dashboard(dh_network,
                                     hyd_result,
                                     thermal_result=None,
                                     dynamic_result=None,
                                     node_order: list[str] | None = None,
                                     title: str | None = None,
                                     figsize: tuple = (14, 9)) -> plt.Figure:
    """
    Kombinierter Dashboard für Fernwärmeberechnung.

    Zeigt alle Ergebnisse auf einem Blick:
    - Oben links: Druckprofil
    - Oben rechts: Stationäres Temperaturprofil
    - Unten: Dynamischer Temperaturverlauf (falls vorhanden)
    """
    n_rows = 2 if dynamic_result else 1
    fig = plt.figure(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(COLORS['background'])

    if dynamic_result:
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        ax_p = fig.add_subplot(gs[0, 0])
        ax_t = fig.add_subplot(gs[0, 1])
        ax_d = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_p = fig.add_subplot(gs[0, 0])
        ax_t = fig.add_subplot(gs[0, 1])

    # Druckprofil
    plot_pressure_profile(dh_network, hyd_result,
                          node_order=node_order, ax=ax_p,
                          title="Druckprofil")

    # Stationäres Temperaturprofil
    if thermal_result:
        plot_temperature_profile(dh_network, thermal_result,
                                 node_order=node_order, ax=ax_t,
                                 title="Temperaturprofil (stationär)")
    else:
        ax_t.text(0.5, 0.5, "kein thermisches\nErgebnis übergeben",
                  ha='center', va='center', transform=ax_t.transAxes,
                  fontsize=12, color='gray')

    # Dynamischer Verlauf
    if dynamic_result:
        plot_dynamic_temperatures(dynamic_result, ax=ax_d,
                                  title="Wärmeausbreitung (dynamisch)")

    # Hydraulik-Info
    info_lines = [
        f"Netz: {dh_network.name}",
        f"Konvergenz: {'Ja' if hyd_result.converged else 'Nein'} ({hyd_result.iterations} Iter.)",
        f"Massenströme [kg/s]: " +
        ", ".join(f"{k}={v:.2f}" for k, v in hyd_result.mass_flows.items()),
    ]
    fig.text(0.5, 0.01, " | ".join(info_lines), ha='center', fontsize=8,
             color='gray', transform=fig.transFigure)

    fig.suptitle(
        title or f"Fernwärme-Dashboard — {dh_network.name}",
        fontsize=13, fontweight='bold', y=0.98
    )
    return fig
