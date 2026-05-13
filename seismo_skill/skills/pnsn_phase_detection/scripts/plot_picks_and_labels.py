#!/usr/bin/env python3
"""Plot continuous waveform picks against manual/curated labels.

This script is intentionally stored inside the pnsn_phase_detection skill so it
can be used even when a demo folder such as publish_mini is not shipped.
It supports SeismicX-Cont style annotation JSON and picker JSONL files, and it
falls back to generic HDF5 traversal when the project dataloader is unavailable.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np


PHASE_COLORS = {
    "P": "tab:red",
    "Pg": "tab:red",
    "Pn": "crimson",
    "S": "tab:blue",
    "Sg": "tab:blue",
    "Sn": "navy",
}


@dataclass
class Pick:
    station_id: str
    phase: str
    time: float
    source: str
    confidence: float | None = None
    event_id: str | None = None


@dataclass
class WaveformPanel:
    station_id: str
    h5_file: str
    start: float
    end: float
    sampling_rate: float
    waveform: np.ndarray
    channels: list[str]


def parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def relpath_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def iter_h5_files(project_root: Path, pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        base = pattern_path.parent
        name = pattern_path.name
        return sorted(p for p in base.glob(name) if p.is_file())
    return sorted(p for p in project_root.glob(pattern) if p.is_file())


def load_label_picks(label_json: Path) -> list[Pick]:
    if not label_json.exists():
        return []
    obj = json.loads(label_json.read_text(encoding="utf-8"))
    picks: list[Pick] = []
    years = obj.get("years", {})
    for year in years.values():
        for day in year.get("days", {}).values():
            for event_id, event in day.get("events", {}).items():
                stations = event.get("stations", {})
                station_values = stations.values() if isinstance(stations, dict) else stations
                for station in station_values:
                    station_id = station.get("station_id") or ".".join(
                        str(station.get(k, "--")) for k in ("network", "station", "location")
                    )
                    for pick in station.get("picks", []):
                        t = parse_time(pick.get("time") or pick.get("phase_time"))
                        phase = str(pick.get("phase") or pick.get("phase_name") or "").strip()
                        if t is None or not phase:
                            continue
                        score = pick.get("score")
                        picks.append(
                            Pick(
                                station_id=station_id,
                                phase=phase,
                                time=t,
                                source=str(pick.get("status") or "label"),
                                confidence=float(score) if isinstance(score, (int, float)) else None,
                                event_id=str(event_id),
                            )
                        )
    return picks


def load_auto_picks(jsonl: Path, min_confidence: float = 0.0) -> list[Pick]:
    if not jsonl or not jsonl.exists():
        return []
    picks: list[Pick] = []
    with jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = str(rec.get("phase_name") or rec.get("phase") or "").strip()
            t = parse_time(rec.get("phase_time") or rec.get("time"))
            prob = rec.get("phase_prob", rec.get("prob", rec.get("confidence")))
            conf = float(prob) if isinstance(prob, (int, float)) else None
            if t is None or not phase:
                continue
            if conf is not None and conf < min_confidence:
                continue
            station_info = rec.get("station_info") or {}
            station_id = rec.get("station_id") or station_info.get("station_id")
            if not station_id:
                net = station_info.get("network") or rec.get("network") or "--"
                sta = station_info.get("station") or rec.get("station") or "--"
                loc = station_info.get("location") or rec.get("location") or "--"
                station_id = f"{net}.{sta}.{loc}"
            picks.append(
                Pick(
                    station_id=str(station_id),
                    phase=phase,
                    time=t,
                    source="auto",
                    confidence=conf,
                    event_id=str(rec.get("event_id")) if rec.get("event_id") else None,
                )
            )
    return picks


def station_aliases(station_id: str) -> set[str]:
    parts = station_id.split(".")
    aliases = {station_id}
    if len(parts) >= 2:
        aliases.add(".".join(parts[:2]))
    if len(parts) >= 3:
        aliases.add(".".join(parts[:3]))
        if parts[2] == "00":
            aliases.add(".".join(parts[:2] + ["--"]))
        if parts[2] == "--":
            aliases.add(".".join(parts[:2] + ["00"]))
    return aliases


def normalize_waveform(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 6:
        arr = arr.T
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)
    return arr


def read_attr_time(attrs: Any, names: Iterable[str]) -> float | None:
    for name in names:
        if name in attrs:
            value = attrs[name]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            t = parse_time(value)
            if t is not None:
                return t
    return None


def find_waveform_datasets(h5: h5py.File) -> list[tuple[str, h5py.Dataset]]:
    out: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        if obj.ndim not in (1, 2):
            return
        if max(obj.shape) < 200:
            return
        out.append((name, obj))

    h5.visititems(visit)
    return out


def panel_from_dataset(h5_path: Path, name: str, ds: h5py.Dataset) -> WaveformPanel | None:
    arr = normalize_waveform(ds[()])
    attrs = dict(ds.attrs)
    parent = ds.parent
    while parent is not None:
        for key, value in parent.attrs.items():
            attrs.setdefault(key, value)
        if parent.name == "/":
            break
        parent = parent.parent
    start = read_attr_time(attrs, ["starttime", "start_time", "start", "begin_time", "utc_start"])
    sr = None
    for key in ("sampling_rate", "sample_rate", "fs", "sps"):
        if key in attrs:
            try:
                sr = float(attrs[key])
                break
            except Exception:
                pass
    if start is None or not sr or sr <= 0:
        return None
    npts = arr.shape[0]
    end = start + npts / sr
    station_id = None
    for key in ("station_id", "station", "id"):
        if key in attrs:
            station_id = attrs[key]
            if isinstance(station_id, bytes):
                station_id = station_id.decode("utf-8", errors="replace")
            station_id = str(station_id)
            break
    if not station_id:
        tokens = [x for x in name.split("/") if x]
        station_id = tokens[-2] if len(tokens) >= 2 else tokens[-1]
    channels = ["E", "N", "Z"][: arr.shape[1]]
    return WaveformPanel(str(station_id), str(h5_path), start, end, sr, arr, channels)


def load_panels_generic(h5_files: list[Path], max_panels: int) -> list[WaveformPanel]:
    panels: list[WaveformPanel] = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            for name, ds in find_waveform_datasets(h5):
                panel = panel_from_dataset(h5_path, name, ds)
                if panel is None:
                    continue
                panels.append(panel)
                if len(panels) >= max_panels:
                    return panels
    return panels


def load_panels_with_project_dataset(
    project_root: Path,
    h5_pattern: str,
    max_panels: int,
    target_sampling_rate: float,
    picks_by_station: dict[str, list[Pick]] | None = None,
) -> list[WaveformPanel]:
    utils_root = project_root / "utils"
    if not utils_root.exists():
        return []
    import importlib.util

    module_path = utils_root / "hdf5_waveform_dataset.py"
    if not module_path.exists():
        return []
    spec = importlib.util.spec_from_file_location("skill_hdf5_waveform_dataset", module_path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_cls = getattr(module, "HDF5WaveformDataset", None)
    if dataset_cls is None:
        return []
    dataset = dataset_cls(
        h5_file=str(project_root / h5_pattern) if not Path(h5_pattern).is_absolute() else h5_pattern,
        target_sampling_rate=target_sampling_rate,
        keep_h5_open=False,
    )
    panels: list[WaveformPanel] = []
    fallback: list[WaveformPanel] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        wave = sample.get("waveform")
        if hasattr(wave, "detach"):
            wave = wave.detach().cpu().numpy()
        arr = normalize_waveform(np.asarray(wave))
        start = parse_time(sample.get("starttime"))
        end = parse_time(sample.get("endtime"))
        sr = float(sample.get("sampling_rate") or target_sampling_rate)
        if start is None:
            continue
        if end is None:
            end = start + arr.shape[0] / sr
        panel = WaveformPanel(
            station_id=str(sample.get("station_id", "unknown")),
            h5_file=str(sample.get("h5_file", "")),
            start=float(start),
            end=float(end),
            sampling_rate=sr,
            waveform=arr,
            channels=list(sample.get("channels") or ["E", "N", "Z"])[: arr.shape[1]],
        )
        if len(fallback) < max_panels:
            fallback.append(panel)
        if picks_by_station:
            matched = False
            for alias in station_aliases(panel.station_id):
                for pick in picks_by_station.get(alias, []):
                    if panel.start <= pick.time <= panel.end:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                continue
        panels.append(panel)
        if len(panels) >= max_panels:
            break
    return panels or fallback


def choose_panels_with_picks(
    panels: list[WaveformPanel],
    picks_by_station: dict[str, list[Pick]],
    max_panels: int,
) -> list[WaveformPanel]:
    scored: list[tuple[int, WaveformPanel]] = []
    for panel in panels:
        aliases = station_aliases(panel.station_id)
        count = 0
        for alias in aliases:
            for pick in picks_by_station.get(alias, []):
                if panel.start <= pick.time <= panel.end:
                    count += 1
        scored.append((count, panel))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for count, p in scored[:max_panels] if count > 0] or [p for _, p in scored[:max_panels]]


def plot_panel(panel: WaveformPanel, picks: list[Pick], out_path: Path, window_seconds: float | None) -> None:
    arr = panel.waveform
    sr = panel.sampling_rate
    if window_seconds:
        n = min(arr.shape[0], int(window_seconds * sr))
        if picks:
            first = min(p.time for p in picks)
            center_idx = int((first - panel.start) * sr)
            lo = max(0, center_idx - n // 4)
            hi = min(arr.shape[0], lo + n)
            lo = max(0, hi - n)
        else:
            lo, hi = 0, n
    else:
        lo, hi = 0, arr.shape[0]
    t = np.arange(lo, hi) / sr
    fig, axes = plt.subplots(arr.shape[1], 1, figsize=(13, 2.4 * arr.shape[1]), sharex=True)
    if arr.shape[1] == 1:
        axes = [axes]
    offset_start = panel.start + lo / sr
    for i, ax in enumerate(axes):
        y = arr[lo:hi, i]
        scale = np.nanmax(np.abs(y))
        if not math.isfinite(scale) or scale == 0:
            scale = 1.0
        ax.plot(t, y / scale, color="0.2", lw=0.7)
        ax.set_ylabel(panel.channels[i] if i < len(panel.channels) else f"C{i+1}")
        ax.grid(True, alpha=0.2)
        for pick in picks:
            rel = pick.time - offset_start
            if rel < 0 or rel > (hi - lo) / sr:
                continue
            color = PHASE_COLORS.get(pick.phase, "black")
            style = "-" if pick.source == "auto" else "--"
            ax.axvline(rel, color=color, linestyle=style, alpha=0.85, lw=1.1)
            label = pick.phase if pick.source != "auto" else f"{pick.phase} auto"
            ax.text(rel, 0.92, label, color=color, rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
    axes[-1].set_xlabel("Time since panel start (s)")
    title_time = datetime.fromtimestamp(offset_start, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    fig.suptitle(f"{panel.station_id} | {title_time} | labels dashed, auto solid", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root containing data/ and optional utils/.")
    parser.add_argument("--h5-input", default="data/hdf5/*.h5", help="HDF5 glob relative to project root.")
    parser.add_argument("--label-json", default="data/label/annotations_mini_two_hours.json")
    parser.add_argument("--auto-jsonl", default="", help="Optional picker JSONL output.")
    parser.add_argument("--outdir", default="", help="Output directory; defaults to SAGE_OUTDIR/annotation_plots or project outputs.")
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--window-seconds", type=float, default=180.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--target-sampling-rate", type=float, default=100.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    label_json = Path(args.label_json)
    if not label_json.is_absolute():
        label_json = project_root / label_json
    auto_jsonl = Path(args.auto_jsonl) if args.auto_jsonl else None
    if auto_jsonl and not auto_jsonl.is_absolute():
        auto_jsonl = project_root / auto_jsonl
    outdir = Path(args.outdir) if args.outdir else Path(os.environ.get("SAGE_OUTDIR", project_root / "outputs" / "pnsn_annotation_plots"))
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    label_picks = load_label_picks(label_json)
    auto_picks = load_auto_picks(auto_jsonl, min_confidence=args.min_confidence) if auto_jsonl else []
    all_picks = label_picks + auto_picks
    picks_by_station: dict[str, list[Pick]] = defaultdict(list)
    for pick in all_picks:
        for alias in station_aliases(pick.station_id):
            picks_by_station[alias].append(pick)

    panels = load_panels_with_project_dataset(
        project_root,
        args.h5_input,
        args.max_panels,
        args.target_sampling_rate,
        picks_by_station=picks_by_station,
    )
    backend = "project_dataset"
    if not panels:
        h5_files = iter_h5_files(project_root, args.h5_input)
        panels = load_panels_generic(h5_files, max(args.max_panels * 8, args.max_panels))
        backend = "generic_hdf5"

    chosen = choose_panels_with_picks(panels, picks_by_station, args.max_panels)
    manifest = {
        "project_root": str(project_root),
        "backend": backend,
        "label_json": str(label_json),
        "auto_jsonl": str(auto_jsonl) if auto_jsonl else "",
        "n_label_picks": len(label_picks),
        "n_auto_picks": len(auto_picks),
        "n_panels_found": len(panels),
        "figures": [],
    }
    for i, panel in enumerate(chosen, 1):
        aliases = station_aliases(panel.station_id)
        panel_picks = []
        seen = set()
        for alias in aliases:
            for pick in picks_by_station.get(alias, []):
                key = (pick.station_id, pick.phase, pick.time, pick.source)
                if key in seen:
                    continue
                seen.add(key)
                if panel.start <= pick.time <= panel.end:
                    panel_picks.append(pick)
        panel_picks.sort(key=lambda p: (p.time, p.phase, p.source))
        out_path = outdir / f"annotation_panel_{i:03d}_{panel.station_id.replace('.', '_')}.png"
        plot_panel(panel, panel_picks, out_path, args.window_seconds)
        manifest["figures"].append(
            {
                "path": str(out_path),
                "station_id": panel.station_id,
                "h5_file": panel.h5_file,
                "n_picks": len(panel_picks),
                "start": datetime.fromtimestamp(panel.start, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(panel.end, tz=timezone.utc).isoformat(),
            }
        )
        print(f"[FIGURE] {out_path} picks={len(panel_picks)}")

    manifest_path = outdir / "annotation_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAGE_TEST] annotation plotting complete: figures={len(manifest['figures'])}")
    print(f"[SAGE_TEST] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
