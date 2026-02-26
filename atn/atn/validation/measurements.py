"""
Validierung von Messdaten in energetisch/stofflich verflochtenen Systemen.

Implementierung nach Strelow & Dawitz, THM-Hochschulschriften Band 14 (2020).

Aufgabe: Gegeben ein System K · x = b und unvollständige/fehlerhafte Messdaten,
finde den physikalisch konsistenten Systemzustand x* der:
  1. Die Bilanzen K · x* ≈ b einhält
  2. Minimal von den Messungen abweicht (Methode der Fehlerquadrate)
  3. Fehlerbehaftete Sensoren identifiziert

Anwendungsbeispiele (Strelow & Dawitz 2020, Kap. 5):
  - Verbrennung lösungsmittelbeladener Abluftströme
  - Wärmerückgewinnungsschaltungen mit Temperaturmessung
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Measurement:
    variable_index: int
    value: float
    weight: float = 1.0
    unit: str = ""
    sensor_id: str = ""


@dataclass
class ValidationResult:
    validated_state: np.ndarray           # Korrigierter Zustandsvektor x*
    measurement_corrections: np.ndarray   # Δy = H·x* - y (Messabweichungen)
    balance_errors: np.ndarray            # K·x* - b (Bilanzverletzungen)
    rms_correction: float                 # RMS der Korrekturen
    max_correction: float                 # Größte Einzelkorrektur
    status: str                           # 'ok' | 'underdetermined' | 'failed'


@dataclass
class FaultySensor:
    measurement_index: int
    sensor_id: str
    variable_index: int
    measured_value: float
    validated_value: float
    deviation: float
    relative_deviation: float


class SystemValidator:
    """
    Validiert Messdaten eines energetisch/stofflich verflochtenen Systems.

    Löst das Ausgleichsproblem (Strelow & Dawitz 2020, Kap. 2):

        min  (H·x - y)^T · W · (H·x - y)
        s.t. K · x = b

    Lagrange-Lösung:
        [2·H^T·W·H   K^T] · [x]   [2·H^T·W·y]
        [K            0 ]   [λ] = [b         ]

    Args:
        K: Systemmatrix (Bilanzen × Zustandsvariablen)
        b: rechte Seite der Systemgleichungen (default: Nullvektor)
    """

    def __init__(self, K: np.ndarray, b: np.ndarray | None = None):
        self.K = np.atleast_2d(K).astype(float)
        n_vars = K.shape[1]
        self.b = b if b is not None else np.zeros(K.shape[0])
        self._measurements: list[Measurement] = []

    def add_measurement(self, variable_index: int, value: float,
                        weight: float = 1.0, unit: str = "",
                        sensor_id: str = "") -> SystemValidator:
        """
        Messwert hinzufügen.

        Args:
            variable_index: Index der gemessenen Variablen in x
            value         : Messwert
            weight        : Gewichtungsfaktor (höher = mehr Vertrauen in Sensor)
                           Typisch: 1/σ² mit Messrauschen σ
            sensor_id     : Bezeichnung des Sensors (für Auswertung)
        """
        self._measurements.append(
            Measurement(variable_index, value, weight, unit, sensor_id)
        )
        return self

    def validate(self) -> ValidationResult:
        """
        Berechnet den validierten Systemzustand.

        Methode der gewichteten Fehlerquadrate mit Gleichungsrestriktionen
        (Strelow & Dawitz 2020, Kap. 2-3).
        """
        n_vars = self.K.shape[1]
        n_meas = len(self._measurements)

        if n_meas == 0:
            return ValidationResult(
                validated_state=np.zeros(n_vars),
                measurement_corrections=np.array([]),
                balance_errors=self.K @ np.zeros(n_vars) - self.b,
                rms_correction=0.0,
                max_correction=0.0,
                status='underdetermined'
            )

        # Messmatrix H und Messvektor y
        H = np.zeros((n_meas, n_vars))
        y = np.zeros(n_meas)
        W = np.zeros(n_meas)

        for i, m in enumerate(self._measurements):
            H[i, m.variable_index] = 1.0
            y[i] = m.value
            W[i] = m.weight

        W_mat = np.diag(W)
        n_eq = self.K.shape[0]

        # Lagrange-System aufstellen
        A_top = np.hstack([2 * H.T @ W_mat @ H, self.K.T])
        A_bot = np.hstack([self.K, np.zeros((n_eq, n_eq))])
        A = np.vstack([A_top, A_bot])
        rhs = np.concatenate([2 * H.T @ W_mat @ y, self.b])

        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
            x_val = sol[:n_vars]

            corrections = H @ x_val - y
            balance_err = self.K @ x_val - self.b

            return ValidationResult(
                validated_state=x_val,
                measurement_corrections=corrections,
                balance_errors=balance_err,
                rms_correction=float(np.sqrt(np.mean(corrections**2))),
                max_correction=float(np.max(np.abs(corrections))),
                status='ok'
            )
        except np.linalg.LinAlgError:
            return ValidationResult(
                validated_state=np.zeros(n_vars),
                measurement_corrections=np.zeros(n_meas),
                balance_errors=np.zeros(n_eq),
                rms_correction=float('inf'),
                max_correction=float('inf'),
                status='failed'
            )

    def validate_with_relative_weights(self, sigma: dict[int, float]
                                        ) -> ValidationResult:
        """
        Validierung mit relativer Wichtung nach Messgenauigkeit.
        (Strelow & Dawitz 2020, Kap. 3)

        sigma: {variable_index → Standardabweichung des Sensors}
        Gewicht w_i = 1 / σ_i²
        """
        for m in self._measurements:
            if m.variable_index in sigma:
                m.weight = 1.0 / max(sigma[m.variable_index]**2, 1e-12)
        return self.validate()

    def detect_faulty_sensors(self,
                               threshold_sigma: float = 2.0
                               ) -> list[FaultySensor]:
        """
        Identifiziert Sensoren mit unplausiblen Abweichungen.
        (Strelow & Dawitz 2020, Kap. 4 — Validierung unvollständiger Messdaten)

        Ein Sensor gilt als fehlerhaft, wenn seine Korrektur mehr als
        threshold_sigma Standardabweichungen vom Mittelwert abweicht.

        Returns:
            Liste fehlerverdächtiger Sensoren mit Abweichungsdetails.
        """
        result = self.validate()
        if result.status != 'ok':
            return []

        corrections = result.measurement_corrections
        std = np.std(corrections)
        mean = np.mean(corrections)

        faulty = []
        for i, (corr, m) in enumerate(zip(corrections, self._measurements)):
            z_score = abs(corr - mean) / max(std, 1e-10)
            if z_score > threshold_sigma:
                val_val = m.value - corr
                faulty.append(FaultySensor(
                    measurement_index=i,
                    sensor_id=m.sensor_id or f"Sensor_{i}",
                    variable_index=m.variable_index,
                    measured_value=m.value,
                    validated_value=float(val_val),
                    deviation=float(corr),
                    relative_deviation=float(corr / max(abs(m.value), 1e-10)),
                ))

        return sorted(faulty, key=lambda s: abs(s.deviation), reverse=True)

    def completeness(self) -> dict:
        """
        Analysiert die Vollständigkeit der Messdaten.
        Gibt an, welche Variablen gemessen sind und welche rekonstruiert werden.
        """
        n_vars = self.K.shape[1]
        measured_vars = {m.variable_index for m in self._measurements}
        unmeasured_vars = set(range(n_vars)) - measured_vars

        # Rang des messbaren Teilsystems
        H = np.zeros((len(self._measurements), n_vars))
        for i, m in enumerate(self._measurements):
            H[i, m.variable_index] = 1.0

        combined = np.vstack([self.K, H])
        rank = np.linalg.matrix_rank(combined)

        return {
            'n_variables': n_vars,
            'n_measured': len(measured_vars),
            'n_unmeasured': len(unmeasured_vars),
            'measured_indices': sorted(measured_vars),
            'unmeasured_indices': sorted(unmeasured_vars),
            'system_rank': rank,
            'fully_determined': rank == n_vars,
        }
