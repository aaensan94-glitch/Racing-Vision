# Race Vision (CV1) – Mini Racing Telemetrie mit Webcam

Python-Projekt mit OpenCV für die Telemetrie von Miniatur-Rennen. Verfolgt
ein Fahrzeug mit einem farbigen Marker, erkennt handgezeichnete Gates
(zwei Pfosten + Linie + Ziffer als Gate-ID) und piept bei jeder
Gate-Durchfahrt.

## Was es kann
- **Live-Verfolgung** (HSV + Morphologie) für mehrere Fahrzeuge.
- **Handgezeichnete Gates**: Kreis-/Linien-Detektion + MNIST-CNN für die
  Gate-Ziffer, Reihenfolge wird automatisch bestimmt.
- **Gate-Kreuzungen**: Piepton und Overlay-Blink bei jeder Durchfahrt.
- **CSV-Logging** (`logs/log_<ts>.csv`, Spalten `t, car, x, y, speed_px_s`).

## Installation
```
pip install -r requirements.txt
# optional (für das MNIST-CNN):
pip install -r requirements-ml.txt
python train_digits.py   # einmalig; schreibt models/digits.pt
```

## Start
```
python main.py
```

## Bedienung
- **q / ESC**: Beenden.
- **p**: Pause.
- **t**: Gefahrene Spuren löschen.
- **g**: Gates kalibrieren (Kreise + Linien + Ziffern erkennen).
- **G**: Gate-Kandidaten (alle Kreise/Linien) ein-/ausblenden.
- **h**: HSV-Werte um jedes Auto loggen.
- **f**: Vollbild.

## Dateien
- `main.py` — Hauptloop, Overlay, Tasten.
- `vision.py` — Marker-Tracking (HSV + Morphologie).
- `gates.py` — Gate-Erkennung (Kreise, Linien, Pairing, OCR-Crop,
  Orientierung, Klassifikation).
- `digits.py` + `train_digits.py` — MNIST-CNN für Gate-Ziffern.
- `geometry.py` — Segment-Schnitt, Punkt-Abstand.
- `sound.py` — Low-Latency-Piepton bei Gate-Durchfahrt.
- `configs/cars.json` — HSV-Bereiche pro Fahrzeug.
