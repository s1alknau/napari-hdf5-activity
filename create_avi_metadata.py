#!/usr/bin/env python3
"""
create_avi_metadata.py - Einfacher Generator für AVI Metadata

Erstellt eine metadata.json für AVI-Video-Analyse mit Standard-Parametern.

Usage:
    python create_avi_metadata.py

    # Oder mit Parametern:
    python create_avi_metadata.py --light-start 7 --light-end 19 --days 3
"""

import argparse
import json


def create_simple_metadata(
    output_path: str = "metadata.json",
    fps: float = 5.0,
    frame_interval: float = 5.0,
    light_start_hour: int = 7,
    light_end_hour: int = 19,
    n_days: int = 3,
):
    """
    Erstellt eine einfache metadata.json für AVI-Videos.

    Args:
        output_path: Wo die Datei gespeichert werden soll
        fps: Frames per second des Videos (Standard: 5)
        frame_interval: Sekunden zwischen analysierten Frames (Standard: 5.0s)
        light_start_hour: Beginn der Lichtphase (Standard: 7 = 07:00 Uhr)
        light_end_hour: Ende der Lichtphase (Standard: 19 = 19:00 Uhr)
        n_days: Anzahl Tage mit Lichtzyklen (Standard: 3)
    """

    # Erstelle Lichtphasen für mehrere Tage
    light_periods = []
    for day in range(n_days):
        start = light_start_hour + (day * 24)
        end = light_end_hour + (day * 24)
        light_periods.append(
            {
                "start_hour": start,
                "end_hour": end,
                "description": f"Tag {day + 1}: {light_start_hour:02d}:00-{light_end_hour:02d}:00 Uhr",
            }
        )

    metadata = {
        "_comment": "Metadata für AVI-Video-Analyse mit napari-hdf5-activity",
        "_wichtig": "AVI-Videos wurden bei durchgehender IR-Beleuchtung aufgenommen!",
        "video_settings": {
            "fps": fps,
            "frame_interval": frame_interval,
            "resolution": {"width": 1920, "height": 1080},
            "camera": "IR camera",
            "illumination": "continuous IR, white LED for light phases",
        },
        "lighting_conditions": {
            "ir_led": {
                "status": "always_on",
                "power_percent": 100,
                "note": "IR LED durchgehend an für Video-Aufnahme",
            },
            "white_led": {
                "status": "scheduled",
                "schedule_type": "custom",
                "light_periods": light_periods,
            },
        },
    }

    # Speichern
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata erstellt: {output_path}")
    print("\nEinstellungen:")
    print(f"  - Video FPS: {fps}")
    print(f"  - Frame Intervall: {frame_interval}s (1 Frame alle {frame_interval}s)")
    print(f"  - Effektive FPS: {1.0/frame_interval:.3f} FPS")
    print(
        f"  - Lichtzyklen: {n_days} Tage, {light_start_hour:02d}:00-{light_end_hour:02d}:00 Uhr"
    )
    print("\nLege diese Datei neben deine AVI-Videos:")
    print("  Videos/")
    print("    ├── video.avi")
    print("    └── metadata.json  <- Diese Datei")


def main():
    parser = argparse.ArgumentParser(
        description="Erstelle metadata.json für AVI-Videos"
    )
    parser.add_argument(
        "--output",
        default="metadata.json",
        help="Output-Datei (Standard: metadata.json)",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0, help="Video FPS (Standard: 5)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Frame-Intervall in Sekunden (Standard: 5.0s, wie HDF5)",
    )
    parser.add_argument(
        "--light-start",
        type=int,
        default=7,
        help="Lichtphase Start-Stunde (Standard: 7)",
    )
    parser.add_argument(
        "--light-end", type=int, default=19, help="Lichtphase End-Stunde (Standard: 19)"
    )
    parser.add_argument(
        "--days", type=int, default=3, help="Anzahl Tage mit Lichtzyklen (Standard: 3)"
    )

    args = parser.parse_args()

    create_simple_metadata(
        output_path=args.output,
        fps=args.fps,
        frame_interval=args.interval,
        light_start_hour=args.light_start,
        light_end_hour=args.light_end,
        n_days=args.days,
    )


if __name__ == "__main__":
    main()
