# ATN Dashboard

**KI-gestütztes Netzberechnungs-Dashboard**
nach der Allgemeinen Theorie der Technischen Netze (ATN)
von Prof. Dr.-Ing. Olaf Strelow, THM Gießen

## Unterstützte Netzdomänen

| Domäne | Berechnungsmodell | DVGW-Norm |
|--------|------------------|-----------|
| Wasserversorgung | Hazen-Williams | W 303 |
| Gasversorgung ND/MD/HD | Darcy-Weisbach / Weymouth | G 462, G 600 |
| Fernwärme | Darcy-Weisbach + Wärmetransport | — |
| Strom DC | Ohmsches Gesetz | — |
| Strom AC | Komplexe Admittanzmatrix | — |

## Lokal starten

```bash
pip install -r requirements.txt
pip install -e atn/          # ATN-Framework installieren
streamlit run dashboard/app.py
```

## CSV-Import

Die App unterstützt den Import eigener Netze als CSV-Datei (Leitungen).
Beispieldateien liegen in `dashboard/beispiel_csv/`.

Mindest-Spalten: `von`, `nach`, `laenge`, `durchmesser`
Spaltenbezeichnungen werden automatisch erkannt (Deutsch & Englisch).

## Projektstruktur

```
atn/                  ATN-Framework (Python-Paket)
  atn/networks/       Netzmodelle (Wasser, Gas, Fernwärme, Strom)
  atn/core/           ATN-Kern (Matrizen, Solver)
  atn/coupling/       Sektorenkopplungsmodell
  atn/visualization/  Plot-Funktionen
  examples/           Beispielskripte
dashboard/
  app.py              Streamlit-Dashboard
  csv_import.py       CSV-Import-Logik
  beispiel_csv/       Beispiel-Netzdaten
```

## Mathematischer Kern (ATN)

Alle Domänen basieren auf denselben drei Gleichungen:

```
Knotensatz:        K · I + I_ext = 0
Maschensatz:       ΔU = Kᵀ · U
Widerstandsgesetz: I = −R⁻¹ · ΔU
```

Nichtlineare Hydraulikformeln werden durch iterative Linearisierung
um den aktuellen Arbeitspunkt gelöst.
