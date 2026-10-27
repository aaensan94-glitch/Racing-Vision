# Configs

## `cars.json` — HSV-Bereiche pro Auto

Jedes Auto braucht einen HSV-Farbbereich. Format:
```json
"name": { "hsv_lower": [H, S, V], "hsv_upper": [H, S, V] }
```

### OpenCV-HSV-Skala (wichtig!)
OpenCV nutzt **andere Bereiche** als die üblichen Farbrad-Werte:

| Kanal | Bereich   | Bedeutung                                    |
|-------|-----------|----------------------------------------------|
| H     | 0–**180** | Farbton (nicht 0–360!)                       |
| S     | 0–255     | Sättigung: 0 = grau, 255 = volle Farbe       |
| V     | 0–255     | Helligkeit: 0 = schwarz, 255 = weiß/hell     |

### Hue-Zonen (grobe Richtwerte)

```
  0 ─── 10 ── 25 ── 35 ────── 85 ── 100 ───── 130 ────── 170 ── 180
  │ rot │ or. │ gelb │  grün  │ cyan │  blau   │ magenta  │ rot  │
```

| Farbe     | H-Bereich       |
|-----------|-----------------|
| Rot       | 0–10 **und** 170–180 (wickelt um!) |
| Orange    | 10–25           |
| Gelb      | 25–35           |
| Grün      | 35–85           |
| Cyan      | 85–100          |
| Blau      | 100–130         |
| Magenta   | 130–170         |

**Rot ist Sonderfall**: zwei Bereiche nötig oder Marker leicht orange/magenta wählen.

### S und V — Faust-Regeln
- **`S_lower`**: meist 60–100. Tiefer → auch verwaschene Töne; zu tief → Weißabgleich flackert.
- **`V_lower`**: meist 60–100. Tiefer → auch dunkle Ecken werden erkannt; zu tief → Rauschen.
- **Obergrenzen** typisch beide 255.

### Kalibrier-Workflow
1. `python main.py` starten, Marker in Bild halten
2. **`h`** drücken → HSV-Mittelwert im Terminal
3. In `cars.json` Range **H ±10**, **S/V großzügig** eintragen
4. Neu starten, prüfen

### Neues Auto hinzufügen
Einfach Zeile ergänzen:
```json
"gelb": { "hsv_lower": [25, 80, 80], "hsv_upper": [35, 255, 255] }
```
