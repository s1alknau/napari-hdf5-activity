#!/usr/bin/env python3
"""
process_avi_batch.py - Batch-Processing für mehrere AVI-Videos als Zeitreihe

Verarbeitet mehrere AVI-Dateien als zusammenhängende Zeitreihe mit gleichem
Frame-Intervall wie HDF5-Dateien.

Usage:
    # Einfach: Alle AVIs im Verzeichnis
    python process_avi_batch.py --dir "C:/path/to/videos"

    # Mit spezifischen Videos
    python process_avi_batch.py --videos video1.avi video2.avi video3.avi

    # Mit custom Frame-Intervall
    python process_avi_batch.py --dir "C:/path/to/videos" --interval 0.2
"""

import argparse
from pathlib import Path
from typing import List
import sys


def find_avi_files(directory: str, pattern: str = "*.avi") -> List[str]:
    """Finde alle AVI-Dateien im Verzeichnis, sortiert nach Namen."""
    path = Path(directory)
    if not path.exists():
        print(f"Error: Directory does not exist: {directory}")
        return []

    avi_files = sorted(path.glob(pattern))
    return [str(f) for f in avi_files]


def process_avi_batch_napari(video_paths: List[str], frame_interval: float = 5.0):
    """
    Öffne mehrere AVIs in napari als Batch.

    Args:
        video_paths: Liste von AVI-Datei-Pfaden (in zeitlicher Reihenfolge!)
        frame_interval: Target Frame-Intervall in Sekunden (Standard: 5.0s = 0.2 FPS)
    """
    try:
        import napari
    except ImportError:
        print("Error: napari not installed. Install with: pip install napari[all]")
        sys.exit(1)

    if not video_paths:
        print("Error: No video files specified")
        sys.exit(1)

    print("\nBatch-Processing Configuration:")
    print(f"  Videos: {len(video_paths)}")
    print(f"  Frame interval: {frame_interval}s ({1/frame_interval:.1f} FPS effective)")
    print("\nVideo files in order:")
    for i, path in enumerate(video_paths, 1):
        print(f"  {i}. {Path(path).name}")

    # Napari öffnen mit mehreren AVIs
    # napari wird automatisch load_avi_batch_timeseries verwenden
    print("\nOpening in napari...")
    viewer = napari.Viewer()

    # AVIs als Liste übergeben - Reader erkennt Batch-Modus
    viewer.open(video_paths, plugin="napari-hdf5-activity")

    print("\n✓ Videos loaded in napari")
    print("\nNext steps:")
    print("  1. Use the plugin widget to detect ROIs")
    print("  2. Process data with same settings as HDF5")
    print("  3. Generate plots (Lighting Conditions will show LED phases)")

    napari.run()


def process_avi_batch_standalone(video_paths: List[str], frame_interval: float = 5.0):
    """
    Verarbeite AVIs ohne napari GUI (standalone).

    Args:
        video_paths: Liste von AVI-Datei-Pfaden
        frame_interval: Target Frame-Intervall in Sekunden
    """
    try:
        from napari_hdf5_activity._avi_reader import load_avi_batch_timeseries
    except ImportError:
        print("Error: napari-hdf5-activity not installed")
        sys.exit(1)

    print(f"\nLoading {len(video_paths)} videos...")
    frames, metadata = load_avi_batch_timeseries(video_paths, frame_interval)

    print("\n✓ Batch processing complete!")
    print("\nResults:")
    print(f"  Total frames: {len(frames)}")
    print(
        f"  Total duration: {metadata['total_duration']:.1f}s ({metadata['total_duration']/60:.1f} min)"
    )
    print(f"  Effective FPS: {metadata['effective_fps']:.2f}")
    print(f"  Frame shape: {frames.shape}")

    print("\nVideos processed:")
    for video_info in metadata["videos"]:
        print(f"  - {Path(video_info['path']).name}")
        print(
            f"    Time: {video_info['time_start']:.1f}s - {video_info['time_end']:.1f}s"
        )
        print(
            f"    Frames: {video_info['sampled_frames']} (sampled from {video_info['frame_count']})"
        )

    return frames, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Batch-Processing für mehrere AVI-Videos als Zeitreihe"
    )

    # Input-Optionen
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dir", help="Verzeichnis mit AVI-Dateien (werden alphabetisch sortiert)"
    )
    input_group.add_argument(
        "--videos",
        nargs="+",
        help="Spezifische AVI-Dateien (in zeitlicher Reihenfolge!)",
    )

    # Processing-Optionen
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Frame-Intervall in Sekunden (Standard: 5.0 = 0.2 FPS, wie HDF5)",
    )
    parser.add_argument(
        "--pattern", default="*.avi", help="Datei-Pattern für --dir (Standard: *.avi)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Ohne napari GUI verarbeiten (nur Daten laden)",
    )

    args = parser.parse_args()

    # Sammle Video-Dateien
    if args.dir:
        print(f"Searching for AVI files in: {args.dir}")
        video_paths = find_avi_files(args.dir, args.pattern)
        if not video_paths:
            print(f"No AVI files found matching pattern: {args.pattern}")
            sys.exit(1)
        print(f"Found {len(video_paths)} AVI files")
    else:
        video_paths = args.videos

    # Verarbeite
    if args.no_gui:
        frames, metadata = process_avi_batch_standalone(video_paths, args.interval)
    else:
        process_avi_batch_napari(video_paths, args.interval)


if __name__ == "__main__":
    main()
