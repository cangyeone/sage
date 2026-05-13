#!/usr/bin/env python3
"""Generate ESSD manuscript figures from released index and summary files."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DB_PATH = ROOT / "data" / "index" / "waveform_index.sqlite"
EVAL_DIR = ROOT / "eval_picks"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLORS = {
    "blue": "#3b6fb6",
    "teal": "#3f9f93",
    "green": "#6ba35b",
    "orange": "#d68b39",
    "red": "#bf5b5b",
    "purple": "#8060a8",
    "gray": "#6e7580",
    "light_gray": "#e9edf2",
    "dark": "#252b33",
}


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def draw_box(ax, xy, width, height, title, body, color, alpha=0.045):
    x, y = xy
    shadow = patches.FancyBboxPatch(
        (x + 0.025, y - 0.025),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.055",
        linewidth=0,
        facecolor="#eef2f5",
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.055",
        linewidth=0.8,
        edgecolor="#cfd6de",
        facecolor=mpl.colors.to_rgba(color, alpha),
        zorder=2,
    )
    ax.add_patch(box)
    ax.add_patch(
        patches.Rectangle(
            (x + 0.09, y + 0.13),
            0.055,
            height - 0.26,
            facecolor=color,
            edgecolor="none",
            alpha=0.9,
            zorder=3,
        )
    )
    ax.text(
        x + 0.28,
        y + height - 0.23,
        title,
        ha="left",
        va="top",
        color=COLORS["dark"],
        fontsize=9.3,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + 0.28,
        y + height - 0.55,
        body,
        ha="left",
        va="top",
        color="#3f4852",
        fontsize=7.8,
        linespacing=1.25,
        zorder=4,
    )


def draw_arrow(ax, start, end, color=COLORS["gray"], rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.0,
            color=color,
            shrinkA=6,
            shrinkB=6,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=10,
        ),
    )


def draw_group_label(ax, x, y, label):
    ax.text(
        x,
        y,
        label.upper(),
        ha="left",
        va="bottom",
        fontsize=7.1,
        color="#6f7782",
        fontweight="bold",
    )
    ax.plot([x, x + 2.6], [y - 0.10, y - 0.10], color="#d8dde3", lw=0.8)


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)

    draw_group_label(ax, 0.42, 4.80, "Source inputs")
    draw_group_label(ax, 4.25, 4.80, "Released data products")
    draw_group_label(ax, 8.55, 4.80, "Validation and reuse")

    draw_box(
        ax,
        (0.42, 3.25),
        2.85,
        1.16,
        "Source waveform archives",
        "SCEDC / NCEDC MiniSEED\nCI, BK, NC networks\nnative timing and amplitudes",
        COLORS["blue"],
    )
    draw_box(
        ax,
        (0.42, 1.66),
        2.85,
        1.16,
        "Source event metadata",
        "CEED-derived events and picks\nmanual / automatic status\nstation-level phase records",
        COLORS["orange"],
    )

    draw_box(
        ax,
        (4.25, 3.58),
        3.10,
        1.00,
        "Waveform product",
        "14 daily HDF5 files\n159,916 indexed segments\n387 GiB compressed",
        COLORS["teal"],
    )
    draw_box(
        ax,
        (4.25, 2.30),
        3.10,
        1.00,
        "Coverage product",
        "SQLite segment inventory\nstation-time lookup\ncoverage-aware denominators",
        COLORS["green"],
    )
    draw_box(
        ax,
        (4.25, 1.02),
        3.10,
        1.00,
        "Annotation product",
        "7,519 events\n290,329 P/S labels\nprovenance preserved",
        COLORS["orange"],
    )

    draw_box(
        ax,
        (8.55, 3.27),
        3.05,
        1.16,
        "Validation products",
        "inventory checks\ncoverage summaries\nbaseline picker sanity checks",
        COLORS["purple"],
    )
    draw_box(
        ax,
        (8.55, 1.49),
        3.05,
        1.16,
        "Reusable benchmark",
        "continuous picker evaluation\nfalse-pick burden\nassociation-ready streams",
        COLORS["red"],
    )

    draw_arrow(ax, (3.27, 3.84), (4.25, 4.08))
    draw_arrow(ax, (3.27, 3.48), (4.25, 2.80), rad=-0.05)
    draw_arrow(ax, (3.27, 2.24), (4.25, 1.52))
    draw_arrow(ax, (7.35, 4.08), (8.55, 3.94))
    draw_arrow(ax, (7.35, 2.80), (8.55, 3.58), rad=0.04)
    draw_arrow(ax, (7.35, 1.52), (8.55, 2.07))
    draw_arrow(ax, (10.07, 3.27), (10.07, 2.65))

    ax.plot([0.42, 11.60], [0.55, 0.55], color="#d8dde3", lw=0.8)
    ax.text(
        0.42,
        0.30,
        "Added value",
        ha="left",
        va="center",
        fontsize=7.8,
        color="#6f7782",
        fontweight="bold",
    )
    value_items = [
        "continuous daily streams",
        "explicit source lineage",
        "label-coverage audit",
        "reproducible evaluation target",
    ]
    x = 1.65
    for i, item in enumerate(value_items):
        ax.text(x, 0.30, item, ha="left", va="center", fontsize=8.0, color=COLORS["dark"])
        x += [2.35, 2.05, 2.25, 0][i]
        if i < len(value_items) - 1:
            ax.text(x - 0.18, 0.30, "|", ha="center", va="center", fontsize=8.0, color="#a0a7b0")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.08)
    savefig(fig, "data_product_workflow")


def load_coverage_matrix():
    query = """
        SELECT substr(starttime, 1, 10) AS day,
               network,
               substr(channel, 1, 2) AS family,
               COUNT(DISTINCT station_key) AS stations,
               SUM(end_epoch - start_epoch) / 86400.0 AS component_days
        FROM waveform_segments
        WHERE substr(channel, 1, 2) IN ('HH', 'BH', 'EH', 'HN')
        GROUP BY day, network, family
        ORDER BY day, network, family;
    """
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(query).fetchall()
    con.close()
    days = sorted({r[0] for r in rows})
    row_keys = sorted({(r[1], r[2]) for r in rows}, key=lambda x: (x[0], {"BH": 0, "EH": 1, "HH": 2, "HN": 3}[x[1]]))
    values = np.full((len(row_keys), len(days)), np.nan)
    stations = np.zeros((len(row_keys), len(days)))
    idx_r = {k: i for i, k in enumerate(row_keys)}
    idx_c = {d: i for i, d in enumerate(days)}
    for day, net, fam, nsta, comp_days in rows:
        values[idx_r[(net, fam)], idx_c[day]] = comp_days
        stations[idx_r[(net, fam)], idx_c[day]] = nsta
    return days, row_keys, values, stations


def figure_coverage_matrix() -> None:
    days, rows, values, stations = load_coverage_matrix()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "coverage",
        ["#f2f4f7", "#cfe8df", "#73b5a7", "#2f7f79", "#174e5a"],
    )
    period_slices = [
        ("2019 Ridgecrest week", [i for i, d in enumerate(days) if d.startswith("2019")], "Jul"),
        ("2021 quiet week", [i for i, d in enumerate(days) if d.startswith("2021")], "Nov"),
    ]
    vmax = np.nanmax(values)
    im = None
    for ax, (title, cols, month) in zip(axes, period_slices):
        sub_values = values[:, cols]
        sub_stations = stations[:, cols]
        masked = np.ma.masked_invalid(sub_values)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        sub_days = [days[i] for i in cols]
        day_labels = [f"{month} {int(d[-2:])}" for d in sub_days]
        ax.set_xticks(np.arange(len(sub_days)))
        ax.set_xticklabels(day_labels, rotation=35, ha="right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Monitoring day")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([f"{net}-{fam}" for net, fam in rows])
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        for i in range(sub_values.shape[0]):
            for j in range(sub_values.shape[1]):
                if np.isfinite(sub_values[i, j]) and sub_values[i, j] >= 60:
                    color = "white" if sub_values[i, j] > vmax * 0.56 else COLORS["dark"]
                    ax.text(j, i, f"{int(round(sub_stations[i, j]))}", ha="center", va="center", fontsize=6.5, color=color)

    axes[0].set_ylabel("Network-channel family")
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.018)
    cbar.set_label("component-days per cell")
    axes[0].text(
        0.0,
        -0.23,
        "Numbers inside cells are unique station keys. Blank cells indicate no indexed HH/BH/EH/HN data for that network-family-day.",
        transform=axes[0].transAxes,
        fontsize=8,
        color=COLORS["gray"],
        ha="left",
    )
    savefig(fig, "waveform_coverage_matrix")


def load_label_coverage():
    counts = {
        2019: {("manual", "P"): [0, 0], ("automatic", "P"): [0, 0], ("manual", "S"): [0, 0]},
        2021: {("manual", "P"): [0, 0], ("automatic", "P"): [0, 0], ("manual", "S"): [0, 0]},
    }
    with (EVAL_DIR / "eval_phasenet" / "matches.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            subset = rec.get("subset")
            phase = rec.get("label_phase")
            key = (subset, phase)
            if key not in counts[2019]:
                continue
            year = 2019 if rec["label_time_epoch"] < 1600000000 else 2021
            counts[year][key][0] += 1
            if rec.get("has_waveform"):
                counts[year][key][1] += 1
    labels = [("Manual P", ("manual", "P")), ("Automatic P", ("automatic", "P")), ("Manual S", ("manual", "S"))]
    return {
        year: [(label, counts[year][key][0], counts[year][key][1]) for label, key in labels]
        for year in (2019, 2021)
    }


def figure_label_coverage() -> None:
    by_year = load_label_coverage()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=False, gridspec_kw={"wspace": 0.32})
    for ax, year, title in zip(axes, (2019, 2021), ("2019 Ridgecrest week", "2021 quiet week")):
        rows = by_year[year]
        labels = [r[0] for r in rows]
        totals = np.array([r[1] for r in rows], dtype=float)
        covered = np.array([r[2] for r in rows], dtype=float)
        uncovered = totals - covered
        xmax = max(totals.max() * 1.28, 1)
        y = np.arange(len(rows))
        ax.barh(y, covered, color=COLORS["teal"], label="covered by released CI/BK/NC waveforms")
        ax.barh(y, uncovered, left=covered, color="#dfe4ea", edgecolor="white", label="retained label without released waveform")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("number of phase labels")
        ax.set_title(f"{title}\n{int(totals.sum()):,} labels", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        for i, (c, u, t) in enumerate(zip(covered, uncovered, totals)):
            pct = c / t * 100 if t else 0
            txt_color = "white" if c > xmax * 0.18 else COLORS["dark"]
            ax.text(max(c * 0.5, xmax * 0.035), i, f"{int(c):,}\n{pct:.1f}%", ha="center", va="center", color=txt_color, fontsize=7.3, fontweight="bold")
            if u > xmax * 0.08:
                ax.text(c + u * 0.5, i, f"{int(u):,}", ha="center", va="center", color=COLORS["gray"], fontsize=7.3)
            elif u > 0:
                ax.text(c + u + xmax * 0.018, i, f"+{int(u):,}", ha="left", va="center", color=COLORS["gray"], fontsize=7.3)
        ax.set_xlim(0, xmax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)
    axes[0].text(
        0.0,
        -0.28,
        "Each panel uses its own x-axis scale. Coverage is evaluated at the station and pick time using the waveform index.",
        transform=axes[0].transAxes,
        fontsize=8,
        color=COLORS["gray"],
        ha="left",
    )
    fig.subplots_adjust(bottom=0.24)
    savefig(fig, "label_coverage_qc")


def nice_picker_name(path_name: str) -> str:
    names = {
        "eval_lppnm": "LPPNM",
        "eval_phasenet": "PhaseNet",
        "eval_pnsn_v1": "PNSN v1",
        "eval_pnsn_v3": "PNSN v3",
        "eval_pnsn_v3_5120": "PNSN v3 5120",
        "eval_pnsn_v3_diff": "PNSN v3 diff",
        "eval_seismicxm": "SeismicXM",
        "eval_seist": "SeisT",
        "eval_skynet": "SkyNet",
    }
    return names.get(path_name, path_name.replace("eval_", "").replace("_", " "))


def load_baseline_points():
    points = {2019: [], 2021: []}
    for p in sorted(EVAL_DIR.glob("eval_*/summary.json")):
        counts = {
            2019: {"P": [0, 0], "S": [0, 0]},
            2021: {"P": [0, 0], "S": [0, 0]},
        }
        matches = p.parent / "matches.jsonl"
        with matches.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("subset") != "all" or not rec.get("has_waveform"):
                    continue
                phase = rec["label_phase"]
                if phase not in ("P", "S"):
                    continue
                year = 2019 if rec["label_time_epoch"] < 1600000000 else 2021
                counts[year][phase][0] += 1
                if rec.get("matched"):
                    counts[year][phase][1] += 1
        for year in (2019, 2021):
            p_den, p_tp = counts[year]["P"]
            s_den, s_tp = counts[year]["S"]
            c_den = p_den + s_den
            c_tp = p_tp + s_tp
            points[year].append(
                {
                    "name": nice_picker_name(p.parent.name),
                    "combined": c_tp / c_den if c_den else math.nan,
                    "p_recall": p_tp / p_den if p_den else math.nan,
                    "s_recall": s_tp / s_den if s_den else math.nan,
                    "covered_labels": c_den,
                    "tp": c_tp,
                }
            )
    return points


def figure_baseline_space() -> None:
    by_year = load_baseline_points()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9), sharey=False, gridspec_kw={"wspace": 0.36})
    colors = {"P": COLORS["blue"], "S": COLORS["orange"]}
    for ax, year, title in zip(axes, (2019, 2021), ("2019 Ridgecrest week", "2021 quiet week")):
        points = sorted(by_year[year], key=lambda x: x["combined"], reverse=True)
        names = [p["name"] for p in points]
        y = np.arange(len(points))
        height = 0.35
        ax.barh(y - height / 2, [p["p_recall"] for p in points], height=height, color=colors["P"], label="P")
        ax.barh(y + height / 2, [p["s_recall"] for p in points], height=height, color=colors["S"], label="S")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("coverage-corrected recall")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        for i, p in enumerate(points):
            ax.text(0.965, i, f"{p['combined']:.2f}", ha="right", va="center", fontsize=7.5, color=COLORS["dark"])
    for ax in axes:
        ax.set_ylabel("Picker run")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False, title="Phase")
    axes[0].text(
        0.0,
        -0.25,
        "Rows are sorted independently by period using combined P/S coverage-corrected recall; values at right are combined P/S recall.",
        transform=axes[0].transAxes,
        fontsize=8,
        color=COLORS["gray"],
        ha="left",
    )
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "baseline_reuse_space")


def load_residual_samples(picker_dir: str = "eval_phasenet"):
    samples = {(2019, "P"): [], (2019, "S"): [], (2021, "P"): [], (2021, "S"): []}
    path = EVAL_DIR / picker_dir / "matches.jsonl"
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("subset") != "all" or not rec.get("has_waveform") or not rec.get("matched"):
                continue
            residual = rec.get("residual_s")
            phase = rec.get("label_phase")
            if residual is None or phase not in ("P", "S") or abs(float(residual)) > 5:
                continue
            year = 2019 if rec["label_time_epoch"] < 1600000000 else 2021
            samples[(year, phase)].append(float(residual))
    return {key: np.asarray(vals, dtype=float) for key, vals in samples.items()}


def figure_residual_diagnostic() -> None:
    samples = load_residual_samples("eval_phasenet")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.9), sharex=True, sharey=True)
    x = np.linspace(-5, 5, 800)
    bins = np.linspace(-5, 5, 85)
    titles = {
        (2019, "P"): "2019 Ridgecrest P",
        (2019, "S"): "2019 Ridgecrest S",
        (2021, "P"): "2021 quiet P",
        (2021, "S"): "2021 quiet S",
    }
    for ax, key in zip(axes.flat, [(2019, "P"), (2019, "S"), (2021, "P"), (2021, "S")]):
        vals = samples[key]
        ax.hist(vals, bins=bins, density=True, color="#b8c6d8", edgecolor="white", linewidth=0.35)
        if len(vals) > 10:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd > 0:
                gaussian_pdf = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                ax.plot(x, gaussian_pdf, color=COLORS["gray"], lw=1.2, ls="--", label="Gaussian")
            if scipy_stats is not None:
                df, loc, scale = scipy_stats.t.fit(vals)
                ax.plot(x, scipy_stats.t.pdf(x, df, loc=loc, scale=scale), color=COLORS["red"], lw=1.35, label="Student-t")
                note = f"n={len(vals):,}; nu={df:.2f}"
            else:
                note = f"n={len(vals):,}"
            ax.text(0.04, 0.92, note, transform=ax.transAxes, fontsize=8, color=COLORS["dark"], ha="left", va="top")
        ax.set_title(titles[key], loc="left", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(axis="both", color="#e5e7eb", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(-5, 5)
        ax.set_ylim(1e-4, 25)
    axes[1, 0].set_xlabel("residual relative to reference arrival (s)")
    axes[1, 1].set_xlabel("residual relative to reference arrival (s)")
    axes[0, 0].set_ylabel("density (log scale)")
    axes[1, 0].set_ylabel("density (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, loc="lower left", frameon=False)
    fig.subplots_adjust(bottom=0.13, hspace=0.28, wspace=0.18)
    savefig(fig, "residual_heavytail_diagnostic")


def main() -> None:
    figure_workflow()
    figure_coverage_matrix()
    figure_label_coverage()
    figure_baseline_space()
    figure_residual_diagnostic()
    print("Generated ESSD QC figures:")
    for name in (
        "data_product_workflow",
        "waveform_coverage_matrix",
        "label_coverage_qc",
        "baseline_reuse_space",
        "residual_heavytail_diagnostic",
    ):
        print(f"  figures/{name}.pdf")


if __name__ == "__main__":
    main()
