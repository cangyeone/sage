"""
plotting.py — Seismological analysis figures.

Functions
---------
plot_gr(result, output_path)
    Gutenberg-Richter frequency-magnitude distribution plot.

plot_temporal(catalog, output_path)
    Earthquake rate / cumulative events over time.

plot_spatial(catalog, output_path)
    Epicentre map (longitude vs latitude, colour-coded by depth or time).

plot_all(result, catalog, output_prefix)
    Convenience wrapper: call all three plots and return file paths.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .bvalue import BvalueResult
from .catalog_loader import CatalogData


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.size":        11,
}


def _apply_style():
    if HAS_MATPLOTLIB:
        for k, v in _STYLE.items():
            plt.rcParams[k] = v


# ---------------------------------------------------------------------------
# G-R frequency-magnitude plot
# ---------------------------------------------------------------------------

def plot_gr(
    result: BvalueResult,
    output_path: str | Path,
    title: Optional[str] = None,
) -> str:
    """
    Plot the Gutenberg-Richter frequency-magnitude distribution.

    Parameters
    ----------
    result : BvalueResult
        Output from calc_bvalue_mle or calc_bvalue_lsq.
    output_path : str or Path
        File path for the saved figure (PNG).
    title : str, optional
        Custom plot title.

    Returns
    -------
    str
        Absolute path of the saved figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.asarray(result.mag_bins)
    cumul = np.asarray(result.cumulative_n, dtype=float)
    incr  = np.asarray(result.incremental_n, dtype=float)

    # Observed cumulative
    mask_c = cumul > 0
    ax.semilogy(bins[mask_c], cumul[mask_c], "ko-", ms=5, label="Cumulative (observed)")

    # Observed incremental
    mask_i = incr > 0
    ax.semilogy(bins[mask_i], incr[mask_i], "s", color="#888888", ms=4,
                label="Incremental (observed)", alpha=0.7)

    # Fitted G-R line (only above Mc)
    x_fit = bins[bins >= result.mc - 0.05]
    y_fit = 10 ** (result.a_value - result.b_value * x_fit)
    ax.semilogy(x_fit, y_fit, "r-", lw=2,
                label=f"G-R fit (b={result.b_value:.3f}±{result.b_uncertainty:.3f})")

    # Mc line
    ax.axvline(result.mc, color="blue", ls="--", lw=1.5, label=f"Mc = {result.mc:.2f}")

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Cumulative Number  N(≥ M)")
    ax.set_title(title or "Gutenberg-Richter Frequency-Magnitude Distribution")
    ax.legend(fontsize=9)

    # Annotation
    info = (
        f"b = {result.b_value:.3f} ± {result.b_uncertainty:.3f}\n"
        f"a = {result.a_value:.3f}\n"
        f"Mc = {result.mc:.2f}  ({result.mc_method})\n"
        f"N = {result.n_events}  ({result.method.upper()})"
    )
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Temporal distribution plot
# ---------------------------------------------------------------------------

def plot_temporal(
    catalog: CatalogData,
    output_path: str | Path,
    title: Optional[str] = None,
    bin_hours: float = 24.0,
) -> str:
    """
    Plot earthquake frequency over time (rate per bin + cumulative).

    Parameters
    ----------
    catalog : CatalogData
    output_path : str or Path
    title : str, optional
    bin_hours : float
        Width of time bins in hours. Default 24 (daily).

    Returns
    -------
    str
        Absolute path of the saved figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting.")
    if not catalog.times:
        raise ValueError("CatalogData has no times.")

    _apply_style()
    import matplotlib.dates as mdates
    from datetime import timedelta

    times = sorted(catalog.times)
    t0, t1 = times[0], times[-1]
    total_hours = max((t1 - t0).total_seconds() / 3600, bin_hours)

    # Build bins
    n_bins = max(int(total_hours / bin_hours), 1)
    bin_edges = [t0 + timedelta(hours=i * bin_hours) for i in range(n_bins + 1)]
    counts = np.zeros(n_bins, dtype=int)
    for t in times:
        idx = min(int((t - t0).total_seconds() / 3600 / bin_hours), n_bins - 1)
        counts[idx] += 1

    bin_centres = [t0 + timedelta(hours=(i + 0.5) * bin_hours) for i in range(n_bins)]
    cumulative = np.cumsum(counts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Rate
    ax1.bar(bin_centres, counts, width=timedelta(hours=bin_hours * 0.85),
            color="#4C78A8", alpha=0.8, label=f"Events per {bin_hours:.0f}h")
    ax1.set_ylabel("Event Count")
    ax1.set_title(title or "Seismicity Rate over Time")
    ax1.legend(fontsize=9)

    # Cumulative
    ax2.step(bin_centres, cumulative, where="mid", color="#E45756", lw=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Events")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=9)

    # Date formatting
    total_days = total_hours / 24
    if total_days <= 3:
        fmt = mdates.DateFormatter("%m-%d %H:%M")
    elif total_days <= 60:
        fmt = mdates.DateFormatter("%Y-%m-%d")
    else:
        fmt = mdates.DateFormatter("%Y-%m")
    ax2.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Spatial distribution plot
# ---------------------------------------------------------------------------

def plot_spatial(
    catalog: CatalogData,
    output_path: str | Path,
    title: Optional[str] = None,
    color_by: str = "depth",
) -> str:
    """
    Plot epicentre map (longitude vs latitude).

    Parameters
    ----------
    catalog : CatalogData
    output_path : str or Path
    title : str, optional
    color_by : {'depth', 'time', 'magnitude'}
        Variable used for colour coding. Falls back gracefully if not available.

    Returns
    -------
    str
        Absolute path of the saved figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting.")
    if not catalog.has_locations:
        raise ValueError(
            "CatalogData has no longitude/latitude information. "
            "Load a catalog file with location columns."
        )

    _apply_style()

    lons = np.asarray(catalog.longitudes)
    lats = np.asarray(catalog.latitudes)

    # Choose colour variable
    c_vals = None
    c_label = ""
    cmap = "plasma_r"

    if color_by == "depth" and catalog.depths:
        c_vals = np.asarray(catalog.depths[:len(lons)])
        c_label = "Depth (km)"
        cmap = "plasma_r"
    elif color_by == "time" and catalog.times:
        import matplotlib.dates as mdates
        c_vals = np.array([t.timestamp() for t in catalog.times[:len(lons)]])
        c_label = "Time"
        cmap = "viridis"
    elif color_by == "magnitude" and catalog.magnitudes:
        c_vals = np.asarray(catalog.magnitudes[:len(lons)])
        c_label = "Magnitude"
        cmap = "YlOrRd"

    # Marker size — scale with magnitude if available
    if catalog.magnitudes and len(catalog.magnitudes) >= len(lons):
        mag = np.asarray(catalog.magnitudes[:len(lons)])
        sizes = 10 * (2 ** (mag - mag.min()))
        sizes = np.clip(sizes, 5, 300)
    else:
        sizes = 20

    fig, ax = plt.subplots(figsize=(8, 7))

    if c_vals is not None and len(c_vals) == len(lons):
        sc = ax.scatter(lons, lats, c=c_vals, s=sizes, cmap=cmap,
                        alpha=0.7, edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(c_label)
        if color_by == "time" and catalog.times:
            import matplotlib.dates as mdates
            from datetime import datetime as _dt
            tick_vals = cbar.get_ticks()
            cbar.set_ticklabels(
                [_dt.fromtimestamp(v).strftime("%Y-%m-%d") for v in tick_vals],
                fontsize=7
            )
    else:
        ax.scatter(lons, lats, s=sizes, color="#4C78A8", alpha=0.6, edgecolors="none")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title or f"Epicentre Distribution  (N={len(lons)})")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    result: Optional[BvalueResult],
    catalog: CatalogData,
    output_prefix: str | Path,
) -> dict:
    """
    Generate all available plots and return a dict of {plot_type: file_path}.

    Parameters
    ----------
    result : BvalueResult or None
        B-value analysis result. If None, the G-R plot is skipped.
    catalog : CatalogData
        Earthquake catalog.
    output_prefix : str or Path
        Path prefix (without extension). Each plot appends a suffix.

    Returns
    -------
    dict
        Keys: 'gr', 'temporal', 'spatial' (only for generated plots).
    """
    prefix = Path(output_prefix)
    generated = {}

    if result is not None:
        try:
            p = plot_gr(result, str(prefix) + "_gr.png")
            generated["gr"] = p
        except Exception as e:
            generated["gr_error"] = str(e)

    if catalog.times:
        try:
            p = plot_temporal(catalog, str(prefix) + "_temporal.png")
            generated["temporal"] = p
        except Exception as e:
            generated["temporal_error"] = str(e)

    if catalog.has_locations:
        try:
            p = plot_spatial(catalog, str(prefix) + "_spatial.png")
            generated["spatial"] = p
        except Exception as e:
            generated["spatial_error"] = str(e)

    return generated
