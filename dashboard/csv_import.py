"""
CSV-Import für das ATN Dashboard.

Unterstützt eine Leitungs-CSV mit automatischer Spalten-Erkennung.
Die CSV beschreibt die Kanten (Leitungen/Widerstände) des Netzes.
Knoten werden aus den Von/Nach-Spalten abgeleitet.

Mindest-Spalten (alle Schreibweisen werden erkannt):
    von, nach, label, laenge, durchmesser

Domänen-spezifische Zusatzspalten (optional):
    Wasser  : hazen_c, verbrauch, elevation
    Gas     : abnahme_m3h, min_druck
    Fernwärme: heizlast, rl_temp
    DC/AC   : widerstand, reaktanz

Quell-Knoten werden automatisch erkannt: Knoten ohne eingehende Leitung.
"""

from __future__ import annotations
import pandas as pd
import sys, os

# ATN-Imports (Pfad wird vom Dashboard gesetzt)
_HERE = os.path.dirname(__file__)
_ATN  = os.path.join(_HERE, "..", "atn")
if _ATN not in sys.path:
    sys.path.insert(0, _ATN)

from atn.networks.water          import WaterNetwork
from atn.networks.gas            import GasNetwork, PressureLevel
from atn.networks.district_heating import DistrictHeatingNetwork
from atn.networks.electrical     import DCNetwork, ACNetwork


# ── Spalten-Keywords je semantischer Rolle ────────────────────────────────────
_KEYWORDS: dict[str, list[str]] = {
    # Universell
    "from_node":  ["von", "from", "start", "quelle_knoten", "anfang",
                   "von_knoten", "startknoten", "from_node", "node_from"],
    "to_node":    ["nach", "to", "end", "ziel", "ende", "nach_knoten",
                   "zielknoten", "to_node", "node_to"],
    "label":      ["label", "name", "id", "leitung", "rohr", "pipe",
                   "leitungsname", "rohrname", "bezeichnung"],
    "length":     ["laenge", "laenge_m", "l", "length", "len",
                   "strecke", "l_m", "streckenlänge", "leitungslaenge"],
    "diameter":   ["durchmesser", "durchmesser_m", "d", "dn", "diameter",
                   "diam", "nennweite", "d_m", "innen_d"],
    # Wasser
    "hazen_c":    ["hazen_c", "c", "hazen", "hwc", "c_hazen", "hazen_williams",
                   "hw_c", "rauheitsbeiwert"],
    "demand":     ["verbrauch", "entnahme", "abnahme", "demand", "flow",
                   "q", "q_ls", "abfluss", "q_m3s"],
    "elevation":  ["hoehe", "hoehe_m", "elevation", "z", "geodaetisch",
                   "gelaende", "gelaendehoehe"],
    # Gas
    "offtake":    ["abnahme_m3h", "entnahme_m3h", "flow_m3h", "q_m3h",
                   "abnahme", "verbrauch_gas"],
    "min_pressure": ["min_druck", "min_p", "mindestdruck", "p_min",
                     "min_pressure", "mindest_p"],
    # Fernwärme
    "heat_load":    ["heizlast", "waerme", "last", "heat_load", "p_w",
                     "q_w", "waermebedarf"],
    "return_temp":  ["ruecklauf", "rl_temp", "return_temp", "t_rl",
                     "ruecklauftemperatur", "t_ruecklauf"],
    # Strom DC / AC
    "resistance":   ["widerstand", "r", "resistance", "ohm", "r_ohm"],
    "reactance":    ["reaktanz", "x", "reactance", "x_ohm", "blindwiderstand"],
    "current":      ["strom", "current", "i", "i_a", "quellstrom"],
}

# Pflichtfelder je Domäne  (from_node + to_node immer Pflicht)
_REQUIRED: dict[str, list[str]] = {
    "Wasser":     ["from_node", "to_node", "length", "diameter"],
    "Gas":        ["from_node", "to_node", "length", "diameter"],
    "Fernwärme":  ["from_node", "to_node", "length", "diameter"],
    "Strom (DC)": ["from_node", "to_node", "resistance"],
    "Strom (AC)": ["from_node", "to_node", "resistance"],
}

# Alle relevanten Rollen je Domäne (für das Mapping-UI)
_ROLES: dict[str, list[str]] = {
    "Wasser":     ["from_node", "to_node", "label", "length", "diameter",
                   "hazen_c", "demand", "elevation"],
    "Gas":        ["from_node", "to_node", "label", "length", "diameter",
                   "offtake", "min_pressure"],
    "Fernwärme":  ["from_node", "to_node", "label", "length", "diameter",
                   "heat_load", "return_temp"],
    "Strom (DC)": ["from_node", "to_node", "label", "resistance", "current"],
    "Strom (AC)": ["from_node", "to_node", "label", "resistance", "reactance",
                   "current"],
}

# Lesbare Deutsche Bezeichnungen für das UI
ROLE_LABELS: dict[str, str] = {
    "from_node":    "Von-Knoten *",
    "to_node":      "Nach-Knoten *",
    "label":        "Leitungsname",
    "length":       "Länge [m] *",
    "diameter":     "Durchmesser [m] *",
    "hazen_c":      "Hazen-Williams C",
    "demand":       "Entnahme [m³/s]",
    "elevation":    "Geländehöhe [m]",
    "offtake":      "Abnahme [Nm³/h]",
    "min_pressure": "Min.-Druck [bar]",
    "heat_load":    "Heizlast [W]",
    "return_temp":  "Rücklauf [°C]",
    "resistance":   "Widerstand [Ω] *",
    "reactance":    "Reaktanz [Ω]",
    "current":      "Quellstrom [A]",
}


def _normalize(s: str) -> str:
    """Spaltenname normalisieren für Vergleich."""
    return (s.lower()
            .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            .replace("ß", "ss").replace(" ", "_").replace("-", "_")
            .strip())


def detect_columns(df: pd.DataFrame, domain: str) -> dict[str, str | None]:
    """
    Erkennt automatisch welche CSV-Spalte welcher semantischen Rolle entspricht.

    Returns:
        dict { rolle: spaltenname_original | None }
    """
    norm_to_orig = {_normalize(c): c for c in df.columns}
    result: dict[str, str | None] = {}

    for role in _ROLES.get(domain, []):
        found = None
        for kw in _KEYWORDS.get(role, []):
            if kw in norm_to_orig:
                found = norm_to_orig[kw]
                break
        result[role] = found

    return result


def missing_required(col_map: dict[str, str | None], domain: str) -> list[str]:
    """Gibt Liste der fehlenden Pflichtfelder zurück."""
    return [r for r in _REQUIRED.get(domain, [])
            if not col_map.get(r)]


def roles_for_domain(domain: str) -> list[str]:
    return _ROLES.get(domain, [])


def _get(row, col_map: dict, role: str, default=None):
    """Wert aus einer DataFrame-Zeile per Rolle holen."""
    col = col_map.get(role)
    if col and col in row.index:
        v = row[col]
        if pd.notna(v):
            return v
    return default


def _source_nodes(df: pd.DataFrame, from_col: str, to_col: str) -> set[str]:
    """Knoten ohne eingehende Leitung = Quell-/Einspeiseknoten."""
    all_from = set(df[from_col].dropna().astype(str))
    all_to   = set(df[to_col].dropna().astype(str))
    return all_from - all_to


def build_network(df: pd.DataFrame,
                  col_map: dict[str, str | None],
                  domain: str,
                  pressure_level: str = "MD",
                  net_name: str = "CSV-Import") -> object:
    """
    Baut ein ATN-Netzwerk aus einem DataFrame mit gemappten Spalten.

    Args:
        df            : Leitungs-DataFrame
        col_map       : { rolle: spaltenname } aus detect_columns / UI
        domain        : Netz-Domäne ("Wasser", "Gas", …)
        pressure_level: Nur für Gas: "ND" | "MD" | "HD"
        net_name      : Name des Netzwerks

    Returns:
        ATNNetwork-Instanz
    """
    from_col = col_map["from_node"]
    to_col   = col_map["to_node"]
    sources  = _source_nodes(df, from_col, to_col)

    # ── Wasser ────────────────────────────────────────────────────────────────
    if domain == "Wasser":
        net = WaterNetwork(net_name)
        added_nodes: set[str] = set()

        for _, row in df.iterrows():
            frm = str(row[from_col])
            to  = str(row[to_col])
            lbl = str(_get(row, col_map, "label", f"P{_+1}"))
            L   = float(_get(row, col_map, "length",   500))
            D   = float(_get(row, col_map, "diameter", 0.1))
            C   = float(_get(row, col_map, "hazen_c",  130))

            # Quell-Knoten registrieren
            if frm in sources and frm not in added_nodes:
                elev = float(_get(row, col_map, "elevation", 0.0))
                net.add_reservoir(frm, head=45.0, elevation=elev)
                added_nodes.add(frm)

            # Ziel-Knoten: Entnahme
            if to not in added_nodes and to not in sources:
                demand = float(_get(row, col_map, "demand", 0.003))
                elev   = float(_get(row, col_map, "elevation", 0.0))
                net.add_demand(to, flow=demand, elevation=elev)
                added_nodes.add(to)

            net.add_pipe(frm, to, length=L, diameter=D,
                         hazen_williams_c=C, label=lbl)
        return net

    # ── Gas ───────────────────────────────────────────────────────────────────
    elif domain == "Gas":
        pl_map = {"ND": PressureLevel.ND, "MD": PressureLevel.MD,
                  "HD": PressureLevel.HD}
        net = GasNetwork(net_name, pressure_level=pl_map.get(pressure_level,
                                                              PressureLevel.MD))
        p_feed = {"ND": 0.023, "MD": 0.5, "HD": 40.0}[pressure_level]
        added_nodes: set[str] = set()

        for _, row in df.iterrows():
            frm = str(row[from_col])
            to  = str(row[to_col])
            lbl = str(_get(row, col_map, "label", f"G{_+1}"))
            L   = float(_get(row, col_map, "length",   500))
            D   = float(_get(row, col_map, "diameter", 0.1))

            if frm in sources and frm not in added_nodes:
                net.add_feed(frm, pressure_bar=p_feed)
                added_nodes.add(frm)

            if to not in added_nodes and to not in sources:
                q = float(_get(row, col_map, "offtake", 50.0))
                p_min = float(_get(row, col_map, "min_pressure", 0.017))
                net.add_offtake(to, flow_m3h=q, min_pressure_bar=p_min)
                added_nodes.add(to)

            net.add_pipe(frm, to, length=L, diameter=D, label=lbl)
        return net

    # ── Fernwärme ─────────────────────────────────────────────────────────────
    elif domain == "Fernwärme":
        net = DistrictHeatingNetwork(net_name)
        added_nodes: set[str] = set()

        for _, row in df.iterrows():
            frm = str(row[from_col])
            to  = str(row[to_col])
            lbl = str(_get(row, col_map, "label", f"R{_+1}"))
            L   = float(_get(row, col_map, "length",   300))
            D   = float(_get(row, col_map, "diameter", 0.1))

            if frm in sources and frm not in added_nodes:
                net.add_heat_source(frm, supply_temp=80.0, mass_flow=5.0)
                added_nodes.add(frm)

            if to not in added_nodes and to not in sources:
                q_w   = float(_get(row, col_map, "heat_load",   80_000))
                t_rl  = float(_get(row, col_map, "return_temp", 55.0))
                net.add_consumer(to, heat_load=q_w, return_temp=t_rl)
                added_nodes.add(to)

            net.add_pipe(frm, to, length=L, diameter=D, label=lbl)
        return net

    # ── Strom DC ──────────────────────────────────────────────────────────────
    elif domain == "Strom (DC)":
        net = DCNetwork(net_name)
        added_nodes: set[str] = set()

        for _, row in df.iterrows():
            frm = str(row[from_col])
            to  = str(row[to_col])
            lbl = str(_get(row, col_map, "label", f"R{_+1}"))
            R   = float(_get(row, col_map, "resistance", 1.0))

            if frm in sources and frm not in added_nodes:
                I_src = float(_get(row, col_map, "current", 10.0))
                net.add_current_source(frm, current=I_src)
                added_nodes.add(frm)

            net.add_resistor(frm, to, resistance=R, label=lbl)
        return net

    # ── Strom AC ──────────────────────────────────────────────────────────────
    elif domain == "Strom (AC)":
        net = ACNetwork(net_name)
        added_nodes: set[str] = set()

        for _, row in df.iterrows():
            frm = str(row[from_col])
            to  = str(row[to_col])
            lbl = str(_get(row, col_map, "label", f"Z{_+1}"))
            R   = float(_get(row, col_map, "resistance", 1.0))
            X   = float(_get(row, col_map, "reactance",  0.0))

            if frm in sources and frm not in added_nodes:
                I_src = float(_get(row, col_map, "current", 10.0))
                net.add_node(frm, external_flow=I_src)
                added_nodes.add(frm)

            net.add_impedance(frm, to, R=R, X=X, label=lbl)
        return net

    else:
        raise ValueError(f"Unbekannte Domäne: {domain}")
