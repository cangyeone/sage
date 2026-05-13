#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick checks for the SeismicX-Cont mini release package."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import h5py


def h5_stats(path: Path):
    segments = 0
    stations = set()
    channels = set()
    with h5py.File(path, "r") as h5:
        for year_id in h5:
            year = h5[year_id]
            for day_id in year:
                day = year[day_id]
                if "stations" not in day:
                    continue
                for station_id, station_grp in day["stations"].items():
                    stations.add(station_id)
                    waveform = station_grp.get("waveform")
                    if waveform is None:
                        continue
                    for channel, channel_grp in waveform.items():
                        channels.add(channel)
                        segments += sum(1 for _ in channel_grp.keys())
    return {"segments": segments, "stations": len(stations), "channels": len(channels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root
    h5_files = sorted((root / "data" / "hdf5").glob("continuous_waveform_usa_*.h5"))
    label_json = root / "data" / "label" / "annotations_mini_two_hours.json"
    db_path = root / "data" / "index" / "waveform_index.sqlite"

    print("HDF5 files:")
    for path in h5_files:
        stats = h5_stats(path)
        print(f"  {path.name}: {path.stat().st_size / 1024**2:.1f} MiB, {stats}")

    with label_json.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    print("Annotation summary:")
    print(json.dumps(labels.get("summary", {}), ensure_ascii=False, indent=2))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    n_files = cur.execute("SELECT COUNT(*) FROM hdf5_files").fetchone()[0]
    n_segments = cur.execute("SELECT COUNT(*) FROM waveform_segments").fetchone()[0]
    n_stations = cur.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    conn.close()
    print("SQLite index:")
    print(f"  files={n_files}, waveform_segments={n_segments}, stations={n_stations}")


if __name__ == "__main__":
    main()
