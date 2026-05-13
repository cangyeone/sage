#!/usr/bin/env python3
"""
Minimal HDF5WaveformDataset usage example.

Run:
    python example_dataloader.py --h5_input path/to/data.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hdf5_waveform_dataset import HDF5WaveformDataset, waveform_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_input", default=str(ROOT / "data/hdf5/continuous_waveform_usa_*.h5"),
                        help="HDF5 file, directory, or glob pattern")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples to print")
    parser.add_argument("--response_json", default=str(ROOT / "data/response/instrument_responses_mini.json"),
                        help="Instrument-response JSON used with --remove_response")
    parser.add_argument("--remove_response", action="store_true",
                        help="Remove the native instrument response before resampling")
    parser.add_argument("--response_output", default="VEL",
                        help="Physical output unit for response removal: DISP, VEL, or ACC")
    parser.add_argument("--response_pre_filt", nargs=4, type=float, default=None,
                        metavar=("F1", "F2", "F3", "F4"),
                        help="Four-corner pre-filter passed to ObsPy remove_response")
    parser.add_argument("--response_water_level", type=float, default=60.0,
                        help="ObsPy water level; use a negative value to pass None")
    args = parser.parse_args()
    water_level = None if args.response_water_level < 0 else args.response_water_level

    # ── 1. Build dataset ───────────────────────────────────────────────────
    dataset = HDF5WaveformDataset(
        h5_file=args.h5_input,
        mode="three",                        # returns [T, 3] waveform per station-day
        allowed_families=("HH", "BH", "EH", "HN"),
        allowed_z_only_channels=("EHZ",),
        allow_z_only=True,
        replicate_z_only=True,               # Z-only → [Z, Z, Z]
        target_sampling_rate=100.0,          # resample everything to 100 Hz
        instrument_response_json=args.response_json if args.remove_response else None,
        remove_instrument_response=args.remove_response,
        response_output=args.response_output,
        response_pre_filt=tuple(args.response_pre_filt) if args.response_pre_filt else None,
        response_water_level=water_level,
    )

    print(f"HDF5 files  : {len(dataset.h5_files)}")
    print(f"Total samples: {len(dataset)}")
    print()

    # ── 2. Build DataLoader ────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,           # 0 = single-process, safest for h5py
        collate_fn=waveform_collate_fn,
    )

    # ── 3. Iterate and print ───────────────────────────────────────────────
    for i, batch in enumerate(loader):
        if i >= args.n_samples:
            break

        item = batch[0]          # batch_size=1, so one item per batch

        w = item["waveform"]     # torch.Tensor [T, 3]
        sr = item["sampling_rate"]
        duration_sec = w.shape[0] / sr if sr and sr > 0 else float("nan")
        
        print(f"── Sample {i + 1} ──────────────────────────────────────────")
        print(f"  station_id    : {item['station_id']}")
        print(f"  network       : {item['station_info'].get('network', '')}."
              f"{item['station_info'].get('station', '')}")
        print(f"  channels      : {item['channels']}")
        print(f"  starttime     : {item['starttime']}")
        print(f"  sampling_rate : {sr} Hz")
        print(f"  waveform shape: {tuple(w.shape)}  "
              f"({duration_sec:.1f} s  ×  3 components)")
        print(f"  waveform dtype: {w.dtype}")
        print(f"  Z-only        : {item.get('is_z_only', False)}")
        if args.remove_response:
            print(f"  response      : {item.get('instrument_processing', {})}")
        print(f"  location      : "
              f"lon={item['station_info'].get('longitude', float('nan')):.4f}  "
              f"lat={item['station_info'].get('latitude',  float('nan')):.4f}")
        # Quick per-channel stats
        for c, name in enumerate(["E/1", "N/2", "Z/3"]):
            ch = w[:, c].numpy()
            print(f"  ch[{name}]  "
                  f"min={float(np.min(ch)):+.3e}  "
                  f"max={float(np.max(ch)):+.3e}  "
                  f"std={float(np.std(ch)):.3e}")
        print()

    dataset.close()


if __name__ == "__main__":
    main()
