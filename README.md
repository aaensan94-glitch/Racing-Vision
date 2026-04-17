# Race Vision (CV1) – Mini Racing Telemetrie mit Webcam

Python-Projekt mit OpenCV für die Telemetrie von Miniatur-Rennen. Verfolgt
mehrere Fahrzeuge über farbige Marker, erkennt handgezeichnete Gates
(zwei Pfosten + Verbindungslinie + Ziffer als Gate-ID) und fährt komplette
Rennen mit Countdown, Rundenzählung, Bestzeiten und Sprachausgabe.

## Was es kann
- **Live-Verfolgung** (HSV + Morphologie) für mehrere Fahrzeuge parallel.
- **Handgezeichnete Gates**: Kreis-/Linien-Detektion + MNIST-CNN für die
  Gate-Ziffer, Reihenfolge wird automatisch bestimmt, Ziffer im leeren
  Pfosten-Kreis eingeblendet.
- **Rennen**: Mario-Kart-Countdown (Ampel-Overlay + Beeps + GO), Rundenlimit
  per Links/Rechts einstellbar, Platzierung bei Zieleinlauf.
- **Sprachausgabe** (espeak-ng): Rundenansagen `"<Farbe> <Runde>"`, bei
  Zieleinlauf Platz + beste Runde, nicht-blockierend in eigenem Thread.
- **Trails**: Verlauf transparent, aktuelle Runde opak, Bestzeit-Runde
  dicker; Anfang/Ende werden exakt auf den Gate-Schnittpunkt gesnappt.
- **Overlay/HUD**: Info-Panel links (Farbe, HSV, Position, Speed, Zeiten),
  FPS + Runden-Status rechts, Hilfe unten, HSV-Picker unter der Maus
  rechts unten; alles per `i` ein-/ausblendbar.
- **CSV-Logging** (`logs/log_<ts>.csv`, Spalten `t, car, x, y, speed_px_s`)
  plus Gate-Events und Runden-Zusammenfassung pro Rennen.
- **Simulator** als Kamera-Ersatz (Papier + zwei Punkte auf Kreisbahn
  durch drei Gates) — kein Hardware-Setup nötig zum Testen.

## Installation
```
pip install -r requirements.txt
# optional (für das MNIST-CNN):
pip install -r requirements-ml.txt
python train_digits.py   # einmalig; schreibt models/digits.pt
# optional (für Sprachausgabe):
sudo apt install espeak-ng
```

## Start
```
python main.py
```
Beim Start wird eine Kamera-Liste angezeigt; `s` wählt den Simulator.

## Licht
Helligkeit beeinflusst die effektive Kamera-FPS: bei wenig Licht verlängert
die Auto-Belichtung die Integrationszeit, die reale Frame-Rate sinkt auf
10–15 fps. 1280×720 @ 30 fps über USB (MJPG) ist nur mit guter Beleuchtung
erreichbar. Für konsistente Telemetrie: Tischlampe oder diffuses Oberlicht,
keine harten Schatten über der Strecke.

## Bedienung
Renn-Ablauf:
- **g**: Gates kalibrieren (Kreise + Linien + Ziffern erkennen; speichert
  `configs/gates.json`).
- **s**: Countdown starten (3 Beeps + GO, Ampel im Overlay).
- **n**: Neues Rennen (Log speichern, Timing + Trails zurücksetzen, Gates
  bleiben).
- **Left/Right**: Rundenlimit ± 5 (Min 5, Max 100) — nur vor dem Rennen.

Bildverarbeitung:
- **k**: CLAHE-Kontrastverstärkung.
- **a**: Adjust-Fenster mit Slidern (Brightness/Contrast/Gamma/Saturation)
  und Live-Histogramm. Werte werden persistiert und auch nach Schließen
  angewandt.
- **c**: ROI als 4-Punkt-Polygon per Mausklick setzen. Erneuter Druck
  bricht ab / löscht.
- **h**: Kamera-Modus toggeln (Race ↔ Kalibrierung). Beim Start werden
  verfügbare Modi per `v4l2-ctl` ermittelt: Race = niedrigste Auflösung
  mit max FPS (≤1280×720), Kalibrierung = höchste Auflösung. Gates und
  ROI werden proportional skaliert, Trails gelöscht.

Anzeige:
- **p**: Pause.
- **t**: Gefahrene Spuren löschen.
- **f**: Vollbild.
- **i**: HUD (alle Panels) ein-/ausblenden.
- **q / ESC**: Beenden (speichert Session).

Simulator (nur wenn Quelle = Simulator):
- **r**: Fahrtrichtung umkehren.
- **Up/Down**: Simulator-Geschwindigkeit ×/÷ 1.25.

## Dateien
- `main.py` — Hauptloop, Overlay/HUD, Tasten, Rennlogik.
- `vision.py` — Marker-Tracking (HSV + Morphologie).
- `gates.py` — Gate-Erkennung (Kreise, Linien, Pairing, OCR-Crop,
  Orientierung, Klassifikation, Crossing-Check).
- `digits.py` + `train_digits.py` — MNIST-CNN für Gate-Ziffern.
- `timing.py` — Rundenzählung, Bestzeiten, Gate-Event-Log.
- `geometry.py` — Segment-Schnitt, Punkt-Abstand, Catmull-Rom-Spline.
- `sound.py` — Low-Latency PCM-Sounds + espeak-ng-Sprachausgabe, beides
  in Worker-Threads mit Queues.
- `sim.py` — Simulator-Kamera (duck-typed `cv2.VideoCapture`).
- `configs/cars.json` — HSV-Bereiche pro Fahrzeug.
- `configs/hud.json` — HUD-Schrift, Linienhöhe, Padding, Panel-Alpha.
- `configs/session.json` (auto, gitignored) — Rundenlimit, ROI-Polygon,
  Filter-Werte. Beim Beenden geschrieben, beim Start geladen.
- `configs/gates.json` (auto, gitignored) — persistierte Gate-Kalibrierung.
  Wird nach `g` geschrieben, beim Start geladen.
