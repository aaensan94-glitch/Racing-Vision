# Race Vision (CV1) – Mini Racing Telemetrie mit Webcam

**Kompakte Anleitung zum Verstehen und Starten des Programms**

Dieses Python-Projekt verwendet klassische Computer-Vision-Techniken (OpenCV) für die Telemetrie von Miniatur-Rennen. Es verfolgt ein Fahrzeug mit einem farbigen Marker, misst Position, Geschwindigkeit, Lap-Zeiten und Abweichungen von einer idealen Linie. Ideal für Bildverarbeitung-Kurse oder Hobby-Projekte.

## Was es kann
- **Live-Verfolgung**: Webcam-Feed mit optionaler ROI (Region of Interest).
- **Marker-Erkennung**: HSV-Filterung, Morphologie und Schwerpunktberechnung für präzise Positionsverfolgung.
- **Interaktive Linien**: Ideallinie per Maus zeichnen/speichern; Startlinie mit 2 Klicks definieren.
- **Timing**: Rundenzeiten messen (Linienkreuzung mit Debounce, min. 2s Abstand).
- **Overlay & Logging**: Live-Anzeige (Position, Speed in px/s, Lap-Zeit, Abweichung); CSV-Logs für Analyse.
- **Anpassungen**: ROI setzen, Pause, HSV-Debug, Timer-Reset.

## Voraussetzungen & Setup
- **Hardware**: Webcam (fest montiert, top-down), konstantes Indoor-Licht, Strecke mit hohem Kontrast (z.B. Klebeband auf dunklem Boden).
- **Marker**: Klebe einen knalligen Farbmarker (z.B. neon-grün) oben auf das Fahrzeug.
- **Software**: Python 3.10+, Webcam-Zugang (Index 0).

## Installation
1. Python installieren (falls nicht vorhanden).
2. Abhängigkeiten:
   ```
   pip install opencv-python numpy
   ```

## Start
1. Terminal im Projektordner öffnen.
2. Ausführen:
   ```
   python main.py
   ```
3. Fenster öffnet sich – Marker sollte sichtbar sein.

## Bedienung (Tasten & Maus)
- **q / ESC**: Beenden.
- **p**: Pause (Frame einfrieren).
- **r**: ROI wählen (Maus-Rechteck ziehen, ENTER bestätigen).
- **l**: Ideallinie-Modus (Linksklick setzt Punkte).
- **s**: Ideallinie speichern (`configs/ideal_line.json`).
- **c**: Ideallinie laden.
- **x**: Ideallinie löschen (RAM).
- **b**: Startlinie setzen (2 Klicks: A, B).
- **n**: Lap-Timer reset.
- **h**: HSV-Werte debuggen (Terminal-Ausgabe).

## Konfiguration
- **Marker (HSV)**: `configs/marker_hsv.json` anpassen (z.B. für andere Farben). Verwende `h` für Werte.
- **Ideallinie**: JSON mit Punkten (x,y).
- **Startlinie**: JSON mit 2 Punkten.
- **Logs**: Automatisch in `logs/` als CSV (Spalten: Zeit, x, y, Speed, Distanz, Lap-Event).

## Dateien-Übersicht
- `main.py`: Hauptloop, UI, Overlay, Tasten.
- `vision.py`: Marker-Tracking (HSV, Morphologie, Glättung).
- `timing.py`: Lap-Timing (Kreuzungserkennung, Debounce).
- `geometry.py`: Geometrie (Distanz zu Polylinie, Linienkreuzung).
- `configs/`: JSON-Konfigs (Marker, Linien).
- `logs/`: CSV-Logs (zeitgestempelt).

## Möglichkeiten & Erweiterungen
- **Analyse**: CSV-Daten in Excel/Python plotten (z.B. Speed-Kurven, Lap-Vergleiche).
- **Kalibrierung**: Pixel zu realen Einheiten (cm) umrechnen (z.B. bekannte Distanz messen).
- **Mehr Marker**: Code erweitern für mehrere Fahrzeuge.
- **Outdoor**: Lichtanpassung, Schattenfilter.
- **Integration**: Mit ROS oder anderen Frameworks kombinieren.
- **Debug**: HSV-Werte anpassen; Morphologie-Parameter (Kernel, Iterationen) testen.

## Troubleshooting
- **Marker nicht erkannt**: HSV-Werte mit `h` prüfen und `marker_hsv.json` anpassen. Licht/Kontrast überprüfen.
- **Falsche Laps**: Debounce-Zeit (min_lap_time_s) erhöhen; Startlinie korrekt setzen.
- **Performance**: ROI verwenden; Webcam-Auflösung reduzieren (1280x720 Standard).
- **Fehler**: Logs prüfen; Python-Version bestätigen; OpenCV installiert?
- **Allgemein**: Frame spiegeln (cv2.flip) falls nötig; Webcam-Index anpassen.

## Projektplan (Beispiel für 4 Wochen)
- **Woche 1**: Setup, Marker-Detektion, Tracking, Logging.
- **Woche 2**: Startlinie, Lap-Timing.
- **Woche 3**: Ideallinie, Abweichung.
- **Woche 4**: Polish, Demo, Analyse.

Für Fragen oder Beiträge: Code modular halten – erweitere in den Modulen!
