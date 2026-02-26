"""
ATN — Allgemeine Theorie der Technischen Netze
===============================================
Python-Implementierung des ATN-Frameworks nach Prof. Dr.-Ing. Olaf Strelow,
Technische Hochschule Mittelhessen, Fachbereich Maschinenbau & Energietechnik.

Veröffentlichungsgrundlage:
  Band  3 (2017): Elektrische Netze
  Band  4 (2017): Wärmeübertragerschaltungen
  Band  5 (2017): Plattenwärmeübertrager
  Band 11 (2019): Wirtschaftssysteme
  Band 14 (2020): Validierung von Messdaten     [mit F. Dawitz]
  Band 35 (2025): Fernwärmenetze                [mit B. Kouka]
        (2024):   Kopplungsmodelle / Sektorenkopplung

Schnellstart:
    from atn.networks.electrical import DCNetwork
    from atn.networks.district_heating import DistrictHeatingNetwork
    from atn.coupling.model import CouplingModel
    from atn.validation.measurements import SystemValidator
"""

from .core.network import ATNNetwork, NetworkResult
from .core.gauss_jordan import partial_gauss_jordan, GaussJordanResult

__version__ = "0.1.0"
__author__  = "ATN Python Implementation"
__all__ = [
    "ATNNetwork", "NetworkResult",
    "partial_gauss_jordan", "GaussJordanResult",
]
