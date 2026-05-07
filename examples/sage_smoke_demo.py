#!/usr/bin/env python3
"""Deterministic end-to-end SAGE smoke demo.

This demo intentionally avoids online services and heavy models. It creates a
small synthetic seismic sequence, performs simple threshold phase picking,
associates the picks into one event, writes a catalog, generates a figure, and
records the run in ~/.seismicx/runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for p in (WEB_APP_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_records import append_event, finish_run, start_run


def _gaussian(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _pick_phase(t: np.ndarray, y: np.ndarray, min_t: float, max_t: float) -> tuple[float, float]:
    mask = (t >= min_t) & (t <= max_t)
    idx_local = np.argmax(np.abs(y[mask]))
    idx = np.where(mask)[0][idx_local]
    noise = np.std(y[t < 4.0]) + 1e-6
    confidence = min(0.99, float(abs(y[idx]) / (8 * noise)))
    return float(t[idx]), confidence


def run_smoke_demo(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    run_id = start_run(
        "smoke_demo",
        request="synthetic waveform -> picks -> event catalog -> figure -> report",
        metadata={"output_dir": str(output_dir)},
    )
    append_event(run_id, "start", "Creating synthetic waveform sequence")

    rng = np.random.default_rng(20260507)
    dt = 0.02
    t = np.arange(0, 20, dt)
    stations = [
        {"station": "STA01", "distance_km": 12.0, "p": 6.20, "s": 10.10},
        {"station": "STA02", "distance_km": 18.5, "p": 7.05, "s": 11.30},
        {"station": "STA03", "distance_km": 25.0, "p": 7.90, "s": 12.55},
    ]

    waveform_rows = []
    picks = []
    for sta in stations:
        y = rng.normal(0, 0.03, size=t.shape)
        y += _gaussian(t, sta["p"], 0.06, 1.0)
        y += _gaussian(t, sta["s"], 0.10, 0.75)
        y += 0.02 * np.sin(2 * np.pi * 0.7 * t)

        p_time, p_conf = _pick_phase(t, y, sta["p"] - 0.4, sta["p"] + 0.4)
        s_time, s_conf = _pick_phase(t, y, sta["s"] - 0.5, sta["s"] + 0.5)
        picks.extend([
            {"station": sta["station"], "phase": "P", "time_s": p_time, "confidence": p_conf},
            {"station": sta["station"], "phase": "S", "time_s": s_time, "confidence": s_conf},
        ])
        for ti, yi in zip(t, y):
            waveform_rows.append({"station": sta["station"], "time_s": float(ti), "amplitude": float(yi)})

    append_event(run_id, "picking", f"Picked {len(picks)} phases from {len(stations)} stations")

    waveforms_csv = output_dir / "synthetic_waveforms.csv"
    with waveforms_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "time_s", "amplitude"])
        writer.writeheader()
        writer.writerows(waveform_rows)

    picks_csv = output_dir / "phase_picks.csv"
    with picks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "phase", "time_s", "confidence"])
        writer.writeheader()
        writer.writerows(picks)

    p_times = [p["time_s"] for p in picks if p["phase"] == "P"]
    s_times = [p["time_s"] for p in picks if p["phase"] == "S"]
    origin_time = float(np.median(p_times) - 3.2)
    mean_sp = float(np.mean(np.array(s_times) - np.array(p_times)))
    magnitude_proxy = float(1.2 + np.log10(max(1e-3, mean_sp)))

    catalog = {
        "event_id": "SMOKE001",
        "origin_time_s": round(origin_time, 3),
        "n_picks": len(picks),
        "mean_s_minus_p_s": round(mean_sp, 3),
        "magnitude_proxy": round(magnitude_proxy, 3),
    }
    catalog_csv = output_dir / "event_catalog.csv"
    with catalog_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(catalog.keys()))
        writer.writeheader()
        writer.writerow(catalog)

    append_event(run_id, "association", "Associated picks into one synthetic event", catalog)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(stations), 1, figsize=(9, 5), sharex=True)
    for ax, sta in zip(axes, stations):
        rows = [r for r in waveform_rows if r["station"] == sta["station"]]
        yy = np.array([r["amplitude"] for r in rows])
        ax.plot(t, yy, color="#243b53", lw=0.9)
        for pick in [p for p in picks if p["station"] == sta["station"]]:
            color = "#d64545" if pick["phase"] == "P" else "#2f9e44"
            ax.axvline(pick["time_s"], color=color, lw=1.2, alpha=0.9)
            ax.text(pick["time_s"] + 0.05, 0.75, pick["phase"], color=color, fontsize=9)
        ax.set_ylabel(sta["station"])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SAGE Smoke Demo: synthetic waveform picks")
    fig.tight_layout()
    figure_png = output_dir / "smoke_waveform_picks.png"
    fig.savefig(figure_png, dpi=160)
    plt.close(fig)

    report_md = output_dir / "smoke_report.md"
    report_md.write_text(
        "\n".join([
            "# SAGE Smoke Demo Report",
            "",
            f"- Event: `{catalog['event_id']}`",
            f"- Picks: `{catalog['n_picks']}`",
            f"- Estimated origin time: `{catalog['origin_time_s']} s`",
            f"- Mean S-P: `{catalog['mean_s_minus_p_s']} s`",
            f"- Magnitude proxy: `{catalog['magnitude_proxy']}`",
            "",
            "Artifacts:",
            f"- `{waveforms_csv.name}`",
            f"- `{picks_csv.name}`",
            f"- `{catalog_csv.name}`",
            f"- `{figure_png.name}`",
        ]),
        encoding="utf-8",
    )

    artifacts = [waveforms_csv, picks_csv, catalog_csv, figure_png, report_md]
    append_event(run_id, "report", "Generated figure and report")
    finish_run(
        run_id,
        "succeeded",
        result=catalog,
        artifacts=[str(p) for p in artifacts],
    )

    return {
        "ok": True,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "catalog": catalog,
        "artifacts": [str(p) for p in artifacts],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic SAGE smoke demo")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "smoke_demo" / datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    args = parser.parse_args()
    result = run_smoke_demo(Path(args.output_dir).expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
