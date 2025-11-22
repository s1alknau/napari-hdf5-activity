# AVI File Integration for napari-hdf5-activity

## Overview

Das napari-hdf5-activity Plugin unterstützt jetzt auch AVI-Videoformate zusätzlich zu HDF5-Dateien. Dies ermöglicht die Analyse von Videos aus dem bestehenden MATLAB-basierten Workflow (ActivityExtractorPolyp_v20230105.m).

## Kompatibilität

### MATLAB Workflow Parameter
- **frameRateOffline**: 5 (analysiert jeden 5. Frame)
- **frameRateOnline**: 5 Hz (5 Frames pro Sekunde)
- **ROI-Typen**: Ellipse oder Polygon
- **Bewegungsmetrik**: Pixel-Differenz zwischen Frames

### Python Plugin
Das Plugin verwendet die gleiche Analyse-Pipeline wie für HDF5-Dateien:
- ROI-basierte Bewegungserkennung
- Hysterese-Algorithmus
- Schwellwert-Methoden (Baseline, Calibration, Adaptive)
- Lighting Conditions Plots mit LED-Daten

## Verwendung

### 1. AVI-Datei laden

```python
# In napari
File > Open Files > video.avi auswählen
```

Das Plugin erkennt automatisch AVI-Dateien und lädt sie.

### 2. Metadata-Datei erstellen (optional)

Neben der AVI-Datei kann eine JSON-Metadata-Datei platziert werden:

**Dateien:**
```
experiment_video_001.avi
experiment_video_001.json  # Metadata
```

oder

```
Videos/
  ├── video_001.avi
  ├── video_002.avi
  └── metadata.json  # Gemeinsame Metadata für alle Videos
```

### 3. Metadata-Format

#### Minimale Metadata (metadata.json):
```json
{
  "fps": 5,
  "sampling_rate": 5,
  "led_schedule": {
    "type": "custom",
    "light_periods": [
      {"start_hour": 7, "end_hour": 19}
    ]
  }
}
```

#### Vollständige Metadata mit LED-Timeseries:
```json
{
  "experiment_name": "Melatonin_WT_100uM_ZT3_20230831",
  "fps": 5,
  "frame_interval": 0.2,
  "sampling_rate": 5,
  "resolution": {
    "width": 1920,
    "height": 1080
  },
  "videos": [
    {
      "filename": "video_001.avi",
      "excluded_animals": [3, 7],
      "duration_minutes": 60
    }
  ],
  "led_schedule": {
    "type": "custom",
    "light_periods": [
      {"start_hour": 7, "end_hour": 19},
      {"start_hour": 31, "end_hour": 43}
    ]
  },
  "timeseries": {
    "timestamps": [0, 0.2, 0.4, 0.6, ...],
    "led_white_power_percent": [0, 0, 100, 100, ...],
    "led_ir_power_percent": [100, 100, 100, 100, ...]
  }
}
```

### 4. LED-Daten

**WICHTIG**: AVI-Videos wurden bei **durchgehender IR-Beleuchtung** aufgenommen!
- IR LED: Immer an (100%)
- Weiße LED: Nur während Lichtphasen an
- Video sichtbar: Immer (wegen IR)
- Tiere sehen: Nur weiße LED-Phasen

Das Plugin unterstützt drei Modi für Lighting Conditions:

**Modus 1: Timeseries-Daten** (am genauesten)
```json
"timeseries": {
  "timestamps": [0, 0.2, 0.4, ...],
  "led_white_power_percent": [0, 100, 100, ...],
  "led_ir_power_percent": [100, 100, 100, ...]
}
```

**Modus 2: Light Schedule** (zeitbasiert)
```json
"led_schedule": {
  "type": "custom",
  "light_periods": [
    {"start_hour": 7, "end_hour": 19}
  ]
}
```

**Modus 3: Legacy 12h Zyklen** (Fallback, wenn keine Metadata vorhanden)
- Automatisch: 7:00-19:00 Uhr Licht, Rest dunkel

### 5. Metadata-Template erstellen

```python
from napari_hdf5_activity._avi_reader import create_metadata_template

# Template erstellen
create_metadata_template(
    output_path="C:/Users/user/Desktop/20240627_Nema/metadata.json",
    experiment_name="Melatonin_WT_100uM"
)
```

## Workflow

### Standard-Analyse (wie HDF5)

1. **AVI laden**: File > Open > video.avi
2. **ROI detektieren**: "Detect ROIs" Button
3. **Parameter einstellen**:
   - Frame Interval: 0.2 (für 5 Hz)
   - Threshold-Methode wählen
4. **Analyse starten**: "Process Data" Button
5. **Plots generieren**: Lighting Conditions Plot wählen

### Unterschiede zu HDF5

| Feature | HDF5 | AVI |
|---------|------|-----|
| Frame-by-frame Zugriff | ✅ Lazy loading | ✅ opencv VideoCapture |
| Metadata | ✅ Eingebettet | ⚠️ Separate JSON-Datei |
| LED-Daten | ✅ Timeseries | ⚠️ JSON oder Schedule |
| Timestamps | ✅ Pro Frame | ⚠️ Berechnet aus FPS |
| ROI-Masken | ✅ Labels Layer | ✅ Labels Layer |
| Sampling | ✅ Variable | ✅ frameRateOffline |

## Konvertierung von MATLAB zu Python

### MATLAB ROI.mat → napari Labels

```python
from napari_hdf5_activity._avi_reader import convert_matlab_roi_to_napari
import scipy.io

# ROI.mat laden
rois = convert_matlab_roi_to_napari("path/to/experiment_ROI.mat")

# In napari verwenden:
# 1. AVI laden
# 2. Manuell ROIs zeichnen (aktuell)
# Zukünftig: Automatischer Import von .mat Dateien
```

### MATLAB Excel → JSON Metadata

**MATLAB Excel-Dateien:**
- `experiment_Names.xlsx`: Video-Dateinamen
- `experiment_nOut.xlsx`: Ausgeschlossene Tiere

**Python JSON:**
```json
{
  "videos": [
    {
      "filename": "video_001.avi",
      "excluded_animals": [3, 7]
    }
  ]
}
```

## Fehlerbehebung

### Problem: "opencv-python not available"
```bash
pip install opencv-python
```

### Problem: "No metadata found"
- Plugin verwendet automatisch Fallback (Legacy 12h Zyklen)
- Erstelle metadata.json neben der AVI-Datei

### Problem: "Frame rate mismatch"
- Überprüfe FPS in metadata.json
- Setze korrekten frame_interval im Plugin

### Problem: "ROIs nicht gespeichert"
- ROIs müssen pro Session neu gezeichnet werden
- MATLAB .mat Import in Entwicklung

## Beispiel-Workflow

```python
# 1. Video-Verzeichnis
C:/Users/user/Desktop/20240627_Nema/
├── Videos/
│   ├── video_001.avi
│   ├── video_002.avi
│   └── video_003.avi
└── metadata.json

# 2. Metadata erstellen
{
  "experiment_name": "20240627_Nema",
  "fps": 5,
  "sampling_rate": 5,
  "led_schedule": {
    "type": "custom",
    "light_periods": [
      {"start_hour": 7, "end_hour": 19}
    ]
  }
}

# 3. In napari öffnen
# File > Open > video_001.avi

# 4. ROIs detektieren und analysieren
# 5. Lighting Conditions Plot zeigt automatisch Hell/Dunkel-Phasen
```

## Nächste Schritte / TODO

- [ ] Batch-Processing für mehrere AVI-Dateien
- [ ] MATLAB .mat ROI-Import
- [ ] Excel-zu-JSON Konverter-Skript
- [ ] GUI für Metadata-Editor
- [ ] Automatische Frame-Rate-Erkennung
- [ ] Multi-Video-Zeitreihen-Analyse

## Support

Bei Fragen oder Problemen:
1. Überprüfe Metadata-Format
2. Teste mit Beispiel-Video
3. Prüfe opencv-python Installation
4. Siehe Log-Ausgaben im Plugin

## Vergleich: MATLAB vs Python Plugin

| MATLAB ActivityExtractor | Python napari Plugin |
|--------------------------|----------------------|
| Batch-Processing | ✅ Single file (aktuell) |
| ROI aus .mat | ✅ Manuelle Detektion |
| Excel Metadata | ✅ JSON Metadata |
| Offline sampling | ✅ Sampling rate |
| Pixel change | ✅ Hysteresis detection |
| Frame-by-frame | ✅ opencv/HDF5 |
| Output: .mat files | ✅ Output: Excel/CSV/Plots |

## Migration von MATLAB

1. **Videos behalten**: AVI-Dateien direkt verwendbar
2. **Metadata konvertieren**: Excel → JSON (einmalig)
3. **ROIs neu zeichnen**: Oder .mat Import (in Entwicklung)
4. **Parameter übertragen**:
   - frameRateOffline → sampling_rate
   - frameRateOnline → fps
5. **Analyse**: Gleiche Ergebnisse mit erweiterten Plots
