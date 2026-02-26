"""
Partieller Gauß-Jordan-Algorithmus — mathematisches Herzstück des ATN-Frameworks.

Grundlage: Strelow, THM-Hochschulschriften Band 3 (2017), Gleichungen (6)-(10).

Für eine Kopplungsmatrix K (m×n) liefert der Algorithmus:
  - K_J : Zeilenstufenform (Nullzeilen kennzeichnen Maschen)
  - J   : Jordan-Matrix (partielle Inverse von K)
  - Rang r und Freiheitsgrad d = n - r
  - Pivot-Spalten (abhängige Variablen) und freie Spalten (Entscheidungsgrößen)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GaussJordanResult:
    K_J: np.ndarray        # Zeilenstufenform
    J: np.ndarray          # Jordan-Matrix (partielle Inverse)
    rank: int              # Rang r der Matrix K
    dof: int               # Freiheitsgrad d = n - r
    pivot_cols: list[int]  # Spalten der abhängigen Variablen (V_f)
    free_cols: list[int]   # Spalten der Entscheidungsgrößen (V_e)
    mesh_rows: list[int]   # Zeilenindizes der Nullzeilen (= Maschen)


def partial_gauss_jordan(K: np.ndarray, tol: float = 1e-10) -> GaussJordanResult:
    """
    Partielle Gauß-Jordan-Inversion der Kopplungsmatrix K.

    Algorithmus (vgl. Strelow Band 3, S. 8-9):
      Erweiterte Matrix [K | I_m] → Zeilentransformationen → [K_J | J]

    Die Nullzeilen von K_J zeigen die Maschen des Netzes.
    Die Jordan-Matrix J (partielle Inverse) verknüpft Bilanzen mit Strömen/Flüssen.

    Args:
        K   : Kopplungsmatrix (m Knoten × n Leitungen)
        tol : Schwellwert für Pivot-Erkennung

    Returns:
        GaussJordanResult mit K_J, J, Rang, Freiheitsgrad, Pivot-/freien Spalten
    """
    m, n = K.shape
    # Erweiterte Matrix [K | I_m]
    aug = np.hstack([K.astype(float), np.eye(m)])

    pivot_cols = []
    row = 0

    for col in range(n):
        # Pivot-Zeile suchen (größtes Element für numerische Stabilität)
        pivot_row = None
        max_val = tol
        for r in range(row, m):
            if abs(aug[r, col]) > max_val:
                max_val = abs(aug[r, col])
                pivot_row = r

        if pivot_row is None:
            continue  # Keine Pivot-Zeile → freie Spalte (Entscheidungsgröße)

        # Zeilen tauschen
        aug[[row, pivot_row]] = aug[[pivot_row, row]]

        # Pivot-Zeile normieren
        aug[row] = aug[row] / aug[row, col]

        # Spalte eliminieren (alle anderen Zeilen)
        for r in range(m):
            if r != row:
                aug[r] -= aug[r, col] * aug[row]

        pivot_cols.append(col)
        row += 1

    K_J = aug[:, :n]
    J = aug[:, n:]
    rank = len(pivot_cols)
    dof = n - rank
    free_cols = [c for c in range(n) if c not in pivot_cols]

    # Maschen-Zeilen: Nullzeilen in K_J
    mesh_rows = [i for i in range(m) if np.allclose(K_J[i], 0, atol=tol)]

    return GaussJordanResult(
        K_J=K_J,
        J=J,
        rank=rank,
        dof=dof,
        pivot_cols=pivot_cols,
        free_cols=free_cols,
        mesh_rows=mesh_rows,
    )
