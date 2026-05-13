#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import h5py

ROOT = Path(__file__).resolve().parents[1]


def collect_h5_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(list(path.glob("*.h5")) + list(path.glob("*.hdf5")))


def scan_one_hdf5(h5_path):
    combo_counter = Counter()
    combo_examples = defaultdict(list)

    with h5py.File(h5_path, "r") as h5:
        for year_id in h5.keys():
            year_grp = h5[year_id]
            if not isinstance(year_grp, h5py.Group):
                continue

            for day_id in year_grp.keys():
                day_grp = year_grp[day_id]
                if not isinstance(day_grp, h5py.Group):
                    continue

                if "stations" not in day_grp:
                    continue

                stations_grp = day_grp["stations"]

                for station_id in stations_grp.keys():
                    station_grp = stations_grp[station_id]
                    if "waveform" not in station_grp:
                        continue

                    waveform_grp = station_grp["waveform"]

                    channels = sorted([
                        ch for ch in waveform_grp.keys()
                        if isinstance(waveform_grp[ch], h5py.Group)
                    ])

                    if not channels:
                        continue

                    combo = tuple(channels)
                    combo_counter[combo] += 1

                    if len(combo_examples[combo]) < 5:
                        combo_examples[combo].append(
                            f"{h5_path.name} | {year_id} | {day_id} | {station_id}"
                        )

    return combo_counter, combo_examples


def main():
    parser = argparse.ArgumentParser(
        description="Count channel combinations in HDF5 waveform files."
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/hdf5"),
        help="HDF5 file or directory containing .h5/.hdf5 files.",
    )
    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="Show up to 5 examples for each channel combination.",
    )
    args = parser.parse_args()

    h5_files = collect_h5_files(args.input)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found: {args.input}")

    total_counter = Counter()
    total_examples = defaultdict(list)

    for h5_file in h5_files:
        print(f"[INFO] Scanning {h5_file}")
        counter, examples = scan_one_hdf5(h5_file)

        total_counter.update(counter)

        for combo, exs in examples.items():
            rest = max(0, 5 - len(total_examples[combo]))
            total_examples[combo].extend(exs[:rest])

    print("\n========== Channel combination summary ==========")
    print(f"Total HDF5 files scanned: {len(h5_files)}")
    print(f"Total unique channel combinations: {len(total_counter)}")
    print(f"Total station-day records: {sum(total_counter.values())}")

    print("\nCount | Channel number | Channels")
    print("-" * 80)

    for combo, count in total_counter.most_common():
        combo_str = ",".join(combo)
        print(f"{count:6d} | {len(combo):14d} | {combo_str}")

        if args.show_examples:
            for ex in total_examples[combo]:
                print(f"       example: {ex}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
