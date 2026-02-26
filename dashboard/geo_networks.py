"""
Geo-referenzierte Beispielnetze für Gießen.

Koordinaten (lat, lon) basieren auf realen Standorten in Gießen (Hessen).
Rohrlängen werden automatisch aus den Koordinaten berechnet (Haversine).
"""
from __future__ import annotations
import math, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atn"))

from atn.networks.water            import WaterNetwork
from atn.networks.gas              import GasNetwork, PressureLevel
from atn.networks.district_heating import DistrictHeatingNetwork
from atn.networks.electrical       import ACNetwork, DCNetwork

GIESSEN_CENTER = (50.5841, 8.6784)


def _dist(p1: tuple, p2: tuple) -> int:
    """Haversine-Distanz in Metern (gerundet auf 10 m)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return round(2 * R * math.asin(math.sqrt(a)) / 10) * 10


# ── Knotenkoordinaten ─────────────────────────────────────────────────────────

COORDS: dict[str, dict[str, tuple]] = {

    "Wasser": {
        "Wasserwerk":    (50.5873, 8.6575),  # Wasserwerk Gießen, Lahn-Ufer
        "Seltersweg":    (50.5841, 8.6784),  # Fußgängerzone / Stadtmitte
        "Weststadt":     (50.5805, 8.6642),  # Weststadt / Rodtberggebiet
        "Nordstadt":     (50.5935, 8.6761),  # Nordstadt / Oswaldsgarten
        "Universität":   (50.5898, 8.6833),  # Uni-Campus Naturwiss.
        "Hauptbahnhof":  (50.5722, 8.6672),  # Hbf Gießen
    },

    "Gas": {
        "UeST-Wieseck":  (50.5960, 8.7050),  # Übergabestation Wieseck (Ost)
        "Gewerbe-Ost":   (50.5841, 8.6960),  # Gewerbegebiet Ost / Lützellinden
        "Südstadt":      (50.5697, 8.6844),  # Südstadt / Kleinlinden
        "Wieseck-Nord":  (50.5978, 8.6921),  # Wieseck Wohngebiet
        "Innenstadt":    (50.5841, 8.6784),  # Innenstadt
    },

    "Fernwärme": {
        "HKW":           (50.5741, 8.6893),  # Heizkraftwerk (Klinikum-Nähe)
        "Klinikum":      (50.5755, 8.6839),  # Uniklinikum Gießen
        "Marktplatz":    (50.5841, 8.6784),  # Marktplatz / Stadtmitte
        "Hochschulvtl":  (50.5871, 8.6861),  # Hochschulviertel / Philosophikum
    },

    "Strom (AC)": {
        "UW-Süd":        (50.5651, 8.6615),  # Umspannwerk 110/20 kV Süd
        "Bahnhof":       (50.5722, 8.6672),  # Verteiler Bahnhof
        "Stadtmitte":    (50.5841, 8.6784),  # Verteiler Stadtmitte
        "Nordstadt":     (50.5935, 8.6761),  # Verteiler Nordstadt
        "Uni-Campus":    (50.5898, 8.6833),  # Verteiler Universität
    },

    "Strom (DC)": {
        "UW-Süd":        (50.5651, 8.6615),
        "Bahnhof":       (50.5722, 8.6672),
        "Stadtmitte":    (50.5841, 8.6784),
        "Nordstadt":     (50.5935, 8.6761),
    },
}


# ── Netz-Builder ──────────────────────────────────────────────────────────────

def build_water(pressure_level: str = "MD") -> tuple[WaterNetwork, dict]:
    c = COORDS["Wasser"]
    net = WaterNetwork("Wasserversorgung Gießen (Innenstadt)")

    net.add_reservoir("Wasserwerk",   head=58.0, elevation=0.0)

    net.add_pipe("Wasserwerk",  "Seltersweg",
                 length=_dist(c["Wasserwerk"],  c["Seltersweg"]),
                 diameter=0.25, hazen_williams_c=140, label="DN250-W1")
    net.add_pipe("Wasserwerk",  "Weststadt",
                 length=_dist(c["Wasserwerk"],  c["Weststadt"]),
                 diameter=0.20, hazen_williams_c=140, label="DN200-W2")
    net.add_pipe("Seltersweg",  "Nordstadt",
                 length=_dist(c["Seltersweg"],  c["Nordstadt"]),
                 diameter=0.15, hazen_williams_c=140, label="DN150-W3")
    net.add_pipe("Weststadt",   "Universität",
                 length=_dist(c["Weststadt"],   c["Universität"]),
                 diameter=0.15, hazen_williams_c=130, label="DN150-W4")
    net.add_pipe("Nordstadt",   "Hauptbahnhof",
                 length=_dist(c["Nordstadt"],   c["Hauptbahnhof"]),
                 diameter=0.15, hazen_williams_c=130, label="DN150-W5")

    net.add_demand("Seltersweg",   flow=0.015, elevation=0.0)   # 54 m³/h Stadtmitte
    net.add_demand("Weststadt",    flow=0.010, elevation=0.0)   # 36 m³/h Wohngebiet
    net.add_demand("Nordstadt",    flow=0.008, elevation=0.0)   # 29 m³/h Wohngebiet
    net.add_demand("Universität",  flow=0.012, elevation=0.0)   # 43 m³/h Uni-Campus
    net.add_demand("Hauptbahnhof", flow=0.005, elevation=0.0)   # 18 m³/h Bahnhof

    return net, c


def build_gas(pressure_level: str = "MD") -> tuple[GasNetwork, dict]:
    c = COORDS["Gas"]
    pl = {"ND": PressureLevel.ND, "MD": PressureLevel.MD,
          "HD": PressureLevel.HD}[pressure_level]
    p_feed = {"ND": 0.023, "MD": 0.5, "HD": 40.0}[pressure_level]
    q_scale = {"ND": 1, "MD": 100, "HD": 5000}[pressure_level]

    net = GasNetwork(
        f"Gasversorgung Gießen ({pressure_level})", pressure_level=pl)

    net.add_feed("UeST-Wieseck", pressure_bar=p_feed)

    net.add_pipe("UeST-Wieseck", "Gewerbe-Ost",
                 length=_dist(c["UeST-Wieseck"], c["Gewerbe-Ost"]),
                 diameter=0.15, label="G1")
    net.add_pipe("UeST-Wieseck", "Wieseck-Nord",
                 length=_dist(c["UeST-Wieseck"], c["Wieseck-Nord"]),
                 diameter=0.10, label="G2")
    net.add_pipe("Gewerbe-Ost",  "Innenstadt",
                 length=_dist(c["Gewerbe-Ost"],  c["Innenstadt"]),
                 diameter=0.12, label="G3")
    net.add_pipe("Gewerbe-Ost",  "Südstadt",
                 length=_dist(c["Gewerbe-Ost"],  c["Südstadt"]),
                 diameter=0.10, label="G4")
    net.add_pipe("Innenstadt",   "Südstadt",
                 length=_dist(c["Innenstadt"],   c["Südstadt"]),
                 diameter=0.08, label="G5")   # Ringschluss

    net.add_offtake("Gewerbe-Ost",  flow_m3h=1.5  * q_scale, min_pressure_bar=p_feed * 0.35)
    net.add_offtake("Wieseck-Nord", flow_m3h=0.8  * q_scale, min_pressure_bar=p_feed * 0.35)
    net.add_offtake("Innenstadt",   flow_m3h=1.2  * q_scale, min_pressure_bar=p_feed * 0.35)
    net.add_offtake("Südstadt",     flow_m3h=0.5  * q_scale, min_pressure_bar=p_feed * 0.35)

    return net, c


def build_heating() -> tuple[DistrictHeatingNetwork, dict]:
    c = COORDS["Fernwärme"]
    net = DistrictHeatingNetwork("Fernwärmeversorgung Gießen (Innenstadt)")

    net.add_heat_source("HKW", supply_temp=85.0, mass_flow=8.0)

    net.add_pipe("HKW",          "Klinikum",
                 length=_dist(c["HKW"],       c["Klinikum"]),
                 diameter=0.15, label="FW1")
    net.add_pipe("HKW",          "Marktplatz",
                 length=_dist(c["HKW"],       c["Marktplatz"]),
                 diameter=0.20, label="FW2")
    net.add_pipe("Marktplatz",   "Hochschulvtl",
                 length=_dist(c["Marktplatz"], c["Hochschulvtl"]),
                 diameter=0.10, label="FW3")

    net.add_consumer("Klinikum",     heat_load=480_000, return_temp=55.0)  # 480 kW
    net.add_consumer("Marktplatz",   heat_load=320_000, return_temp=55.0)  # 320 kW
    net.add_consumer("Hochschulvtl", heat_load=200_000, return_temp=55.0)  # 200 kW

    return net, c


def build_ac() -> tuple[ACNetwork, dict]:
    c = COORDS["Strom (AC)"]
    net = ACNetwork("Mittelspannungsnetz Gießen 20 kV")

    # Umspannwerk als Einspeisung (Quellstrom proportional zu Last)
    net.add_node("UW-Süd", external_flow=250.0)   # 250 A Einspeisung

    d_uw_bhf  = _dist(c["UW-Süd"],    c["Bahnhof"])
    d_bhf_st  = _dist(c["Bahnhof"],   c["Stadtmitte"])
    d_st_nord = _dist(c["Stadtmitte"], c["Nordstadt"])
    d_st_uni  = _dist(c["Stadtmitte"], c["Uni-Campus"])

    # Kabelimpedanz: r=0.08 Ω/km, x=0.10 Ω/km (typisch 20kV XLPE)
    r_km, x_km = 0.08, 0.10
    net.add_impedance("UW-Süd",    "Bahnhof",
                      R=round(r_km * d_uw_bhf  / 1000, 4),
                      X=round(x_km * d_uw_bhf  / 1000, 4), label="K1-UW→Bhf")
    net.add_impedance("Bahnhof",   "Stadtmitte",
                      R=round(r_km * d_bhf_st  / 1000, 4),
                      X=round(x_km * d_bhf_st  / 1000, 4), label="K2-Bhf→St")
    net.add_impedance("Stadtmitte","Nordstadt",
                      R=round(r_km * d_st_nord / 1000, 4),
                      X=round(x_km * d_st_nord / 1000, 4), label="K3-St→Nord")
    net.add_impedance("Stadtmitte","Uni-Campus",
                      R=round(r_km * d_st_uni  / 1000, 4),
                      X=round(x_km * d_st_uni  / 1000, 4), label="K4-St→Uni")

    net.add_node("Bahnhof",    external_flow=-60.0)
    net.add_node("Stadtmitte", external_flow=-80.0)
    net.add_node("Nordstadt",  external_flow=-55.0)
    net.add_node("Uni-Campus", external_flow=-55.0)

    return net, c


def build_dc() -> tuple[DCNetwork, dict]:
    c = COORDS["Strom (DC)"]
    net = DCNetwork("Gleichstromnetz Gießen (Beispiel)")

    net.add_current_source("UW-Süd", current=150.0)

    net.add_resistor("UW-Süd",    "Bahnhof",    resistance=2.0, label="R1")
    net.add_resistor("UW-Süd",    "Stadtmitte", resistance=3.5, label="R2")
    net.add_resistor("Bahnhof",   "Stadtmitte", resistance=1.5, label="R3")
    net.add_resistor("Stadtmitte","Nordstadt",  resistance=2.0, label="R4")

    net.add_node("Bahnhof",    external_flow=-45.0)
    net.add_node("Stadtmitte", external_flow=-60.0)
    net.add_node("Nordstadt",  external_flow=-45.0)

    return net, c


BUILDERS = {
    "Wasser":     build_water,
    "Gas":        build_gas,
    "Fernwärme":  build_heating,
    "Strom (AC)": build_ac,
    "Strom (DC)": build_dc,
}
