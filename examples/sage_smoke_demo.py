#!/usr/bin/env python3
"""Deterministic end-to-end SAGE smoke demo.

This demo intentionally avoids online services and heavy models. It creates a
small synthetic seismic sequence, performs simple threshold phase picking,
associates the picks into one event, writes a catalog, generates a figure, and
records the run in SAGE_RUN_RECORD_DIR, ~/.seismicx/runs, or web_app/outputs/runs.
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

    seed = 20260507
    rng = np.random.default_rng(seed)
    dt = 0.02
    t = np.arange(0, 20, dt)
    noise_std = 0.03
    background_amp = 0.02
    background_hz = 0.7
    p_width = 0.06
    s_width = 0.10
    p_amp = 1.0
    s_amp = 0.75
    stations = [
        {"station": "STA01", "distance_km": 12.0, "p": 6.20, "s": 10.10},
        {"station": "STA02", "distance_km": 18.5, "p": 7.05, "s": 11.30},
        {"station": "STA03", "distance_km": 25.0, "p": 7.90, "s": 12.55},
    ]
    station_truth = [
        {
            "station": sta["station"],
            "distance_km": sta["distance_km"],
            "true_p_s": sta["p"],
            "true_s_s": sta["s"],
            "p_window_start_s": round(sta["p"] - 0.4, 3),
            "p_window_end_s": round(sta["p"] + 0.4, 3),
            "s_window_start_s": round(sta["s"] - 0.5, 3),
            "s_window_end_s": round(sta["s"] + 0.5, 3),
            "p_amp": p_amp,
            "s_amp": s_amp,
            "p_width_s": p_width,
            "s_width_s": s_width,
            "noise_std": noise_std,
        }
        for sta in stations
    ]
    generation = {
        "description": "Deterministic synthetic waveform smoke test with one toy event observed by three stations.",
        "random_seed": seed,
        "sample_interval_s": dt,
        "duration_s": 20.0,
        "noise": f"Gaussian white noise N(0, {noise_std})",
        "background": f"{background_amp} * sin(2*pi*{background_hz}*t)",
        "p_pulse": {"shape": "Gaussian", "width_s": p_width, "amplitude": p_amp},
        "s_pulse": {"shape": "Gaussian", "width_s": s_width, "amplitude": s_amp},
        "picking_method": (
            "Ground-truth-window picker: choose max |amplitude| in "
            "[true_p-0.4, true_p+0.4] and [true_s-0.5, true_s+0.5]. "
            "This is a deterministic smoke check, not a trained AI picker."
        ),
        "stations": station_truth,
    }
    run_id = start_run(
        "smoke_demo",
        request="synthetic waveform -> picks -> event catalog -> figure -> report",
        metadata={"output_dir": str(output_dir), "generation": generation},
    )
    append_event(run_id, "start", "Creating synthetic waveform sequence", generation)
    steps = []

    waveform_rows = []
    picks = []
    for sta in stations:
        y = rng.normal(0, noise_std, size=t.shape)
        y += _gaussian(t, sta["p"], p_width, p_amp)
        y += _gaussian(t, sta["s"], s_width, s_amp)
        y += background_amp * np.sin(2 * np.pi * background_hz * t)

        p_time, p_conf = _pick_phase(t, y, sta["p"] - 0.4, sta["p"] + 0.4)
        s_time, s_conf = _pick_phase(t, y, sta["s"] - 0.5, sta["s"] + 0.5)
        picks.extend([
            {"station": sta["station"], "phase": "P", "time_s": p_time, "confidence": p_conf},
            {"station": sta["station"], "phase": "S", "time_s": s_time, "confidence": s_conf},
        ])
        for ti, yi in zip(t, y):
            waveform_rows.append({"station": sta["station"], "time_s": float(ti), "amplitude": float(yi)})

    append_event(run_id, "picking", f"Picked {len(picks)} phases from {len(stations)} stations")
    steps.append({
        "id": "synthesize_and_pick",
        "title": "Synthetic waveforms and phase picks",
        "status": "done",
        "n_stations": len(stations),
        "n_picks": len(picks),
    })

    waveforms_csv = output_dir / "synthetic_waveforms.csv"
    with waveforms_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "time_s", "amplitude"])
        writer.writeheader()
        writer.writerows(waveform_rows)

    truth_csv = output_dir / "station_truth.csv"
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(station_truth[0].keys()))
        writer.writeheader()
        writer.writerows(station_truth)

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
    steps.append({
        "id": "associate_catalog",
        "title": "Associate picks into an event catalog",
        "status": "done",
        "catalog": catalog,
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(stations), 1, figsize=(9, 5), sharex=True)
    for ax, sta in zip(axes, stations):
        rows = [r for r in waveform_rows if r["station"] == sta["station"]]
        yy = np.array([r["amplitude"] for r in rows])
        ax.plot(t, yy, color="#243b53", lw=0.9)
        ax.axvline(sta["p"], color="#d64545", lw=0.8, ls="--", alpha=0.45)
        ax.axvline(sta["s"], color="#2f9e44", lw=0.8, ls="--", alpha=0.45)
        for pick in [p for p in picks if p["station"] == sta["station"]]:
            color = "#d64545" if pick["phase"] == "P" else "#2f9e44"
            ax.axvline(pick["time_s"], color=color, lw=1.2, alpha=0.9)
            ax.text(pick["time_s"] + 0.05, 0.75, pick["phase"], color=color, fontsize=9)
        ax.set_ylabel(sta["station"])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SAGE Smoke Demo: synthetic waveform picks (dashed truth, solid pick)")
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
            "## Synthetic generation",
            "",
            f"- Random seed: `{seed}`",
            f"- Sampling: `dt={dt} s`, duration `{generation['duration_s']} s`",
            f"- Noise: `{generation['noise']}`",
            f"- Background: `{generation['background']}`",
            f"- P pulse: Gaussian, width `{p_width} s`, amplitude `{p_amp}`",
            f"- S pulse: Gaussian, width `{s_width} s`, amplitude `{s_amp}`",
            "",
            "## Ground truth arrivals",
            "",
            "| station | distance_km | true_p_s | true_s_s |",
            "|---|---:|---:|---:|",
            *[
                f"| {row['station']} | {row['distance_km']} | {row['true_p_s']} | {row['true_s_s']} |"
                for row in station_truth
            ],
            "",
            "## Picking method",
            "",
            generation["picking_method"],
            "",
            "Artifacts:",
            f"- `{waveforms_csv.name}`",
            f"- `{truth_csv.name}`",
            f"- `{picks_csv.name}`",
            f"- `{catalog_csv.name}`",
            f"- `{figure_png.name}`",
        ]),
        encoding="utf-8",
    )

    artifacts = [waveforms_csv, truth_csv, picks_csv, catalog_csv, figure_png, report_md]
    append_event(run_id, "report", "Generated figure and report")
    steps.append({
        "id": "report_artifacts",
        "title": "Generate report and artifacts",
        "status": "done",
        "artifacts": [p.name for p in artifacts],
    })
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
        "steps": steps,
        "generation": generation,
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
