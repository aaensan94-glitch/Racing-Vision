# Feature: Handwritten Gates (feature/gates)

## Idee
Strecke aus handgezeichneten Gates. Jedes Gate:
- **Zwei Kreise** = Pfosten
- **Strich zwischen den Kreisen** = Kreuzungslinie (muss überfahren werden)
- **Ziffer (0–9) auf einer Seite** des Strichs = Gate-ID **und** Fahrtrichtung
  - Konvention (v1): Ziffer liegt **links in Fahrtrichtung**
  - `0` = Start/Ziel (Lap-Counter-Linie)
  - `1..9` = Gates in Reihenfolge

## Ablauf
1. **Kalibrieren** (einmalig beim Start, Taste `g`): aktueller Frame → alle Gates erkennen → `configs/track.json` speichern
2. **Rennen**: Gates sind als feste Linien im Bild fixiert. Reihenfolge `0 → 1 → 2 … → N → 0` = eine Runde

## Pipeline (Kalibrierung)

```
Frame (BGR)
  │
  ├─▶ HoughCircles ──▶ Kandidaten-Kreise
  │
  ├─▶ HoughLinesP ──▶ Kandidaten-Strecken
  │
  ├─▶ Pair: Strich verbindet genau zwei Kreise (Endpunkte nahe Kreiszentren)
  │       └─▶ Gate-Kandidat (Kreis A, Kreis B, Linie)
  │
  ├─▶ Für jedes Gate:
  │     a) Bounding-Box seitlich der Linie extrahieren (beide Seiten)
  │     b) Seite mit Ziffern-Blob wählen (mehr dunkle Pixel / Kontrast)
  │     c) Crop um Liniennormale rotieren → Ziffer aufrecht
  │     d) MNIST-CNN → Klassifikation
  │     e) Fahrtrichtung = Liniennormale Richtung Ziffer-Seite (oder invers, je nach Konvention)
  │
  ├─▶ Validierung: IDs 0..N lückenlos, keine Duplikate
  │
  └─▶ Speichern in configs/track.json
```

## `configs/track.json` (Entwurf)
```json
{
  "gates": [
    {
      "id": 0,
      "post_a": [x, y],
      "post_b": [x, y],
      "direction": [dx, dy],   // Einheits-Vektor Durchfahrtsrichtung
      "confidence": 0.98
    }
  ]
}
```

## Runtime (Lap-Counting)
- Jedes Frame: für jedes Auto prüfen, ob Linie Gate `expected_next_id` gekreuzt wurde (mit richtiger Richtung)
- Kreuzung + Richtung korrekt → `expected_next_id = (expected_next_id + 1) % N`
- Gate `0` erreicht → Runde +1

## Ziffernerkennung (CNN)
- **Architektur**: 2× Conv(3×3) + MaxPool + FC → 10 Klassen. ~20k Parameter.
- **Trainingsdaten**: MNIST (60k). Keine Rotations-Augmentation nötig (Rotation ist geometrisch gelöst).
- **Augmentation trotzdem**: kleine Rotation ±10°, leichte Skalierung, Rauschen → robust gegen Crop-Ungenauigkeit.
- **Framework**: PyTorch. Training einmalig offline → `models/digits.pt`.
- **Inference**: CPU reicht (Einzelbilder bei Kalibrierung, ~ms).

## Neue Dateien
- `gates.py` — Erkennung (Kreise, Linien, Pairing, Rotation, Ziffern-Crop)
- `digits.py` — CNN-Wrapper (load + predict)
- `train_digits.py` — MNIST-Training-Script
- `configs/track.json` — Strecken-Layout
- `models/digits.pt` — trainiertes Modell (git-ignored, separat gehostet oder beim ersten Start trainieren)
- `docs/gates_plan.md` — dieses Dokument

## Integration in `main.py`
- Taste `g` → Kalibrierung auslösen, `track.json` speichern/laden
- Overlay: Gate-Linien mit ID beschriften, Pfeil für Richtung
- `LapTimer` erweitern oder ersetzen: statt einer Start-Linie → Sequenz von Gates

## Umsetzungs-Reihenfolge
1. [ ] `gates.py`: nur Kreise + Linien-Pairing, Visualisierung im `main.py` (Taste `g`)
2. [ ] Rotation der Ziffern-Crops korrekt, visuell prüfen
3. [ ] `train_digits.py` + `digits.py` (MNIST, klein)
4. [ ] Klassifikation in Pipeline einbinden
5. [ ] `track.json` persistieren
6. [ ] Lap-Counter über Gate-Sequenz
7. [ ] Tests mit Zeichnungen auf Papier

## Offene Fragen
- Konvention Ziffer-Seite: links oder rechts in Fahrtrichtung? → **v1: links**
- Hintergrundkontrast: weißes Papier mit schwarzem Stift angenommen
- Mindestgröße Kreise / Ziffer im Bild: muss experimentell ermittelt werden
