"""
Folium-Kartenvisualisierung für das ATN Dashboard.

Rendert ATN-Netze auf einer OpenStreetMap-Karte von Gießen.
Knoten und Leitungen werden farbkodiert nach Domäne und Ergebniswerten
dargestellt.
"""
from __future__ import annotations
import math
import folium
from folium.plugins import MiniMap

# Domain-Farben
DOMAIN_COLOR = {
    "Wasser":     "#2E86AB",
    "Gas":        "#E8920A",
    "Fernwärme":  "#E74C3C",
    "Strom (AC)": "#8E44AD",
    "Strom (DC)": "#27AE60",
}

NODE_SOURCE_COLOR   = "#C0392B"   # rot  — Quelle / Einspeisung
NODE_CONSUMER_COLOR = "#2980B9"   # blau — Verbraucher
NODE_JUNCTION_COLOR = "#7F8C8D"   # grau — Knotenpunkt


def _val_to_hex(val: float, vmin: float, vmax: float,
                cmap_name: str = "RdYlGn") -> str:
    """Normierter Wert → Hex-Farbe (matplotlib colormap)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    cmap = plt.get_cmap(cmap_name)
    t = (val - vmin) / (vmax - vmin + 1e-9)
    t = max(0.0, min(1.0, t))
    r, g, b, _ = cmap(t)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def _node_color_from_result(node: str, domain: str, result) -> str | None:
    """Farbwert für einen Knoten aus dem Berechnungsergebnis."""
    if result is None:
        return None
    try:
        if domain == "Wasser":
            res = result[1]
            p = res.pressures.get(node)
            if p is not None:
                return _val_to_hex(p, 1.5, 8.0, "RdYlGn")
        elif domain == "Gas":
            res = result[1]
            p = res.pressures_bar.get(node)
            if p is not None:
                return _val_to_hex(p, 0, res.pressures_bar.get(
                    max(res.pressures_bar, key=res.pressures_bar.get), 1.0),
                    "RdYlGn")
        elif domain == "Fernwärme":
            therm = result[2]
            t = therm.temperatures.get(node)
            if t is not None:
                return _val_to_hex(t, 55, 85, "RdYlBu_r")
        elif domain in ("Strom (AC)", "Strom (DC)"):
            res = result[1]
            if domain == "Strom (DC)":
                u = res.potentials.get(node)
            else:
                import cmath
                u_c = res["voltages"].get(node)
                u = abs(u_c) if u_c is not None else None
            if u is not None:
                return _val_to_hex(u, 0, max(abs(v) for v in
                    (res.potentials.values() if domain == "Strom (DC)"
                     else [abs(x) for x in res["voltages"].values()])),
                    "RdYlGn")
    except Exception:
        pass
    return None


def _edge_popup(label: str, domain: str, result) -> str:
    """Popup-Text für eine Leitung."""
    if result is None:
        return f"<b>{label}</b>"
    try:
        if domain == "Wasser":
            res = result[1]
            q = res.flows.get(label, 0)
            v = res.velocities.get(label, 0)
            dh = res.head_losses.get(label, 0)
            return (f"<b>{label}</b><br>"
                    f"Q = {q*1000:.2f} l/s<br>"
                    f"v = {v:.3f} m/s<br>"
                    f"Δh = {dh:.3f} m")
        elif domain == "Gas":
            res = result[1]
            q = res.flows_norm.get(label, 0)
            dp = res.pressure_drops.get(label, 0)
            return (f"<b>{label}</b><br>"
                    f"Q = {q:.1f} Nm³/h<br>"
                    f"Δp = {dp:.3f} mbar")
        elif domain == "Fernwärme":
            hyd = result[1]
            m = hyd.mass_flows.get(label, 0)
            return f"<b>{label}</b><br>ṁ = {m:.4f} kg/s"
        elif domain in ("Strom (DC)",):
            res = result[1]
            i = res.flows.get(label, 0)
            return f"<b>{label}</b><br>I = {i:.3f} A"
        elif domain == "Strom (AC)":
            res = result[1]
            i_c = res["currents"].get(label, 0)
            return f"<b>{label}</b><br>|I| = {abs(i_c):.3f} A"
    except Exception:
        pass
    return f"<b>{label}</b>"


def _node_popup(node: str, domain: str, network, result) -> str:
    """Popup-Text für einen Knoten."""
    ext = network._external_flows.get(node, 0)
    base = f"<b>{node}</b>"
    if result is None:
        return base + f"<br>Ext. Fluss: {ext:+.4f}"
    try:
        if domain == "Wasser":
            res = result[1]
            p = res.pressures.get(node, 0)
            h = res.heads.get(node, 0)
            ok = "✓ OK" if node not in res.pressure_violations else "⚠ Verletzung"
            return (f"{base}<br>p = {p:.3f} bar<br>"
                    f"h = {h:.2f} m WS<br>{ok}")
        elif domain == "Gas":
            res = result[1]
            p = res.pressures_bar.get(node, 0)
            ok = "✓ OK" if node not in res.pressure_violations else "⚠ Verletzung"
            return f"{base}<br>p = {p:.4f} bar<br>{ok}"
        elif domain == "Fernwärme":
            therm = result[2]
            t = therm.temperatures.get(node, 0)
            return f"{base}<br>T = {t:.2f} °C"
        elif domain == "Strom (DC)":
            res = result[1]
            u = res.potentials.get(node, 0)
            return f"{base}<br>U = {u:.3f} V"
        elif domain == "Strom (AC)":
            res = result[1]
            import cmath
            u_c = res["voltages"].get(node, 0)
            return (f"{base}<br>|U| = {abs(u_c):.3f} V<br>"
                    f"φ = {cmath.phase(u_c)*180/math.pi:.1f}°")
    except Exception:
        pass
    return base


def build_map(network, coords: dict[str, tuple],
              domain: str, result=None,
              zoom: int = 14) -> folium.Map:
    """
    Erstellt eine Folium-Karte mit dem ATN-Netz auf OSM-Hintergrund.

    Args:
        network : ATNNetwork-Instanz
        coords  : {Knotenname: (lat, lon)}
        domain  : Netzdomäne
        result  : Berechnungsergebnis (optional, für Farbkodierung)
        zoom    : Anfangs-Zoomstufe

    Returns:
        folium.Map
    """
    # Kartenmittelpunkt
    lats = [v[0] for v in coords.values()]
    lons = [v[1] for v in coords.values()]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    color = DOMAIN_COLOR.get(domain, "#444")

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Zweite Tile-Option als Layer
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB (hell)").add_to(m)

    # ── Leitungen ────────────────────────────────────────────────────────────
    edge_group = folium.FeatureGroup(name="Leitungen", show=True)
    for from_n, to_n, label in network._edges:
        p1 = coords.get(from_n)
        p2 = coords.get(to_n)
        if p1 is None or p2 is None:
            continue

        # Linienstärke: bei Ergebnis proportional zum Fluss
        weight = 5
        if result is not None:
            try:
                if domain == "Wasser":
                    q = abs(result[1].flows.get(label, 0))
                    qmax = max(abs(v) for v in result[1].flows.values()) or 1
                    weight = 3 + 8 * q / qmax
                elif domain == "Gas":
                    q = abs(result[1].flows_norm.get(label, 0))
                    qmax = max(abs(v) for v in result[1].flows_norm.values()) or 1
                    weight = 3 + 8 * q / qmax
                elif domain == "Fernwärme":
                    m_val = abs(result[1].mass_flows.get(label, 0))
                    mmax = max(abs(v) for v in result[1].mass_flows.values()) or 1
                    weight = 3 + 8 * m_val / mmax
            except Exception:
                pass

        popup_html = _edge_popup(label, domain, result)

        # Leitung zeichnen
        folium.PolyLine(
            locations=[p1, p2],
            color=color,
            weight=weight,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=label,
        ).add_to(edge_group)

        # Pfeilmarkierung in der Mitte
        mid_lat = (p1[0] + p2[0]) / 2
        mid_lon = (p1[1] + p2[1]) / 2
        folium.CircleMarker(
            location=(mid_lat, mid_lon),
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.6,
        ).add_to(edge_group)

    edge_group.add_to(m)

    # ── Knoten ────────────────────────────────────────────────────────────────
    node_group = folium.FeatureGroup(name="Knoten", show=True)
    for node in network._nodes:
        loc = coords.get(node)
        if loc is None:
            continue

        ext = network._external_flows.get(node, 0)

        # Farbe aus Ergebnis (falls vorhanden), sonst nach Typ
        res_color = _node_color_from_result(node, domain, result)
        if res_color:
            fill_color = res_color
            border_color = "white"
        elif ext > 0:
            fill_color = NODE_SOURCE_COLOR
            border_color = "white"
        elif ext < 0:
            fill_color = NODE_CONSUMER_COLOR
            border_color = "white"
        else:
            fill_color = NODE_JUNCTION_COLOR
            border_color = "white"

        radius = 10 if ext > 0 else (8 if ext < 0 else 6)

        popup_html = _node_popup(node, domain, network, result)

        folium.CircleMarker(
            location=loc,
            radius=radius,
            color=border_color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"<b>{node}</b>",
        ).add_to(node_group)

        # Knotenname als Label
        folium.Marker(
            location=(loc[0] + 0.00035, loc[1] + 0.00005),
            icon=folium.DivIcon(
                html=(f'<div style="font-size:11px;font-weight:600;'
                      f'color:{color};text-shadow:1px 1px 2px white,'
                      f'-1px -1px 2px white">{node}</div>'),
                icon_size=(140, 20),
                icon_anchor=(0, 10),
            ),
        ).add_to(node_group)

    node_group.add_to(m)

    # ── Legende ───────────────────────────────────────────────────────────────
    if result is not None:
        legend_items = _build_legend(domain, result)
    else:
        legend_items = (
            f'<i style="background:{NODE_SOURCE_COLOR};'
            f'border-radius:50%;display:inline-block;width:12px;height:12px"></i>'
            f' Einspeisung &nbsp;'
            f'<i style="background:{NODE_CONSUMER_COLOR};'
            f'border-radius:50%;display:inline-block;width:12px;height:12px"></i>'
            f' Verbraucher'
        )

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:12px;
                border-left:4px solid {color}">
        <b style="color:{color}">{domain}</b><br>
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer-Control
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    return m


def _build_legend(domain: str, result) -> str:
    """HTML für die Ergebnis-Legende."""
    try:
        if domain == "Wasser":
            res = result[1]
            p_min = min(res.pressures.values())
            p_max = max(res.pressures.values())
            return (f"Druck: {p_min:.2f} – {p_max:.2f} bar<br>"
                    f'<span style="color:#c0392b">■</span> niedrig &nbsp;'
                    f'<span style="color:#27ae60">■</span> hoch')
        elif domain == "Gas":
            res = result[1]
            p_min = min(res.pressures_bar.values())
            p_max = max(res.pressures_bar.values())
            return (f"Druck: {p_min:.4f} – {p_max:.4f} bar<br>"
                    f'<span style="color:#c0392b">■</span> niedrig &nbsp;'
                    f'<span style="color:#27ae60">■</span> hoch')
        elif domain == "Fernwärme":
            therm = result[2]
            t_min = min(therm.temperatures.values())
            t_max = max(therm.temperatures.values())
            return (f"Temperatur: {t_min:.1f} – {t_max:.1f} °C<br>"
                    f'<span style="color:#2980b9">■</span> kalt &nbsp;'
                    f'<span style="color:#c0392b">■</span> warm')
        elif domain in ("Strom (DC)", "Strom (AC)"):
            return "Spannung (Betrag)<br>hoch = grün, niedrig = rot"
    except Exception:
        pass
    return ""
