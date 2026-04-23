"""
bvalue.py — Gutenberg-Richter b-value and completeness magnitude (Mc) estimation.

Methods
-------
calc_mc_maxcurvature(magnitudes)
    Estimate Mc via the Maximum Curvature method (Wiemer & Wyss 2000).

calc_mc_gof(magnitudes, ...)
    Estimate Mc via Goodness-of-Fit (Wiemer & Wyss 2000).

calc_bvalue_mle(magnitudes, mc, mag_bin)
    Estimate b via Maximum Likelihood Estimation (Aki 1965; Utsu 1965).

calc_bvalue_lsq(magnitudes, mc, mag_bin)
    Estimate b via least-squares regression on the FMD.

BvalueResult
    Dataclass returned by all b-value functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BvalueResult:
    """Results of Gutenberg-Richter b-value analysis."""
    mc: float                   # Completeness magnitude
    b_value: float              # G-R b-value
    b_uncertainty: float        # Standard error of b
    a_value: float              # G-R a-value (log10 N = a - b*M)
    n_events: int               # Number of events used (M >= Mc)
    mean_magnitude: float       # Mean magnitude used
    method: str                 # 'mle' or 'lsq'
    mc_method: str              # 'maxcurvature' or 'gof'

    # Frequency-magnitude distribution (for plotting)
    mag_bins: List[float] = None
    cumulative_n: List[int] = None
    incremental_n: List[int] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  Gutenberg-Richter b-value Analysis",
            "=" * 50,
            f"  Mc (completeness magnitude) : {self.mc:.2f}  [{self.mc_method}]",
            f"  b-value                     : {self.b_value:.3f} ± {self.b_uncertainty:.3f}  [{self.method}]",
            f"  a-value                     : {self.a_value:.3f}",
            f"  Events used (M ≥ Mc)        : {self.n_events}",
            f"  Mean magnitude              : {self.mean_magnitude:.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frequency-magnitude distribution (FMD)
# ---------------------------------------------------------------------------

def _build_fmd(
    magnitudes: List[float],
    mag_bin: float = 0.1,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build frequency-magnitude distribution.

    Returns
    -------
    bins : ndarray
        Magnitude bin centres.
    incremental : ndarray
        Number of events per bin.
    cumulative : ndarray
        Cumulative number of events (N ≥ M).
    """
    mags = np.asarray(magnitudes, dtype=float)
    if min_mag is None:
        min_mag = np.floor(mags.min() / mag_bin) * mag_bin
    if max_mag is None:
        max_mag = np.ceil(mags.max() / mag_bin) * mag_bin

    bins = np.arange(min_mag, max_mag + mag_bin * 0.5, mag_bin)
    bins = np.round(bins, 10)  # avoid float artefacts

    incremental = np.array(
        [np.sum((mags >= b - mag_bin / 2) & (mags < b + mag_bin / 2)) for b in bins],
        dtype=int,
    )
    # Cumulative: N(M >= bin_centre)
    cumulative = np.array(
        [np.sum(mags >= b - mag_bin / 2) for b in bins],
        dtype=int,
    )
    return bins, incremental, cumulative


# ---------------------------------------------------------------------------
# Completeness magnitude — Maximum Curvature
# ---------------------------------------------------------------------------

def calc_mc_maxcurvature(
    magnitudes: List[float],
    mag_bin: float = 0.1,
    correction: float = 0.2,
) -> float:
    """
    Estimate completeness magnitude Mc via the Maximum Curvature method.

    The method locates the peak of the incremental frequency-magnitude
    distribution. A correction of +0.2 is commonly applied (Woessner &
    Wiemer 2005).

    Parameters
    ----------
    magnitudes : list of float
    mag_bin : float, optional
        Magnitude bin width. Default 0.1.
    correction : float, optional
        Bias correction added to raw Mc. Default 0.2.

    Returns
    -------
    float
        Estimated Mc.
    """
    bins, incremental, _ = _build_fmd(magnitudes, mag_bin)
    idx_peak = np.argmax(incremental)
    mc = float(bins[idx_peak]) + correction
    return round(mc, 2)


# ---------------------------------------------------------------------------
# Completeness magnitude — Goodness-of-Fit
# ---------------------------------------------------------------------------

def calc_mc_gof(
    magnitudes: List[float],
    mag_bin: float = 0.1,
    r_threshold: float = 95.0,
) -> float:
    """
    Estimate Mc via the Goodness-of-Fit Test (Wiemer & Wyss 2000).

    For each candidate Mc (bins above the FMD peak), fit a G-R line to
    events with M >= Mc and compute the synthetic FMD. Mc is the lowest
    candidate for which the residual R <= (100 - r_threshold) %.

    Falls back to max-curvature if no Mc satisfies the threshold.

    Parameters
    ----------
    magnitudes : list of float
    mag_bin : float, optional
    r_threshold : float, optional
        Required percentage of data fit (default 95 %).

    Returns
    -------
    float
        Estimated Mc.
    """
    bins, incremental, cumulative = _build_fmd(magnitudes, mag_bin)
    mags = np.asarray(magnitudes, dtype=float)
    idx_peak = int(np.argmax(incremental))

    best_mc = None
    for i in range(idx_peak, len(bins)):
        mc_cand = bins[i]
        subset = mags[mags >= mc_cand - mag_bin / 2]
        n = len(subset)
        if n < 10:
            break
        # MLE b-value for this subset
        mean_m = subset.mean()
        b = math.log10(math.e) / (mean_m - (mc_cand - mag_bin / 2))
        a = math.log10(n) + b * mc_cand

        # Synthetic cumulative counts
        synth = np.array([10 ** (a - b * m) for m in bins[i:]])
        obs = cumulative[i:].astype(float)
        # Residual R (Wiemer & Wyss eq. 3)
        r = 100 * np.sum(np.abs(obs - synth)) / np.sum(obs) if np.sum(obs) > 0 else 100
        if r <= (100 - r_threshold):
            best_mc = float(mc_cand)
            break

    if best_mc is None:
        # Fall back to max curvature
        best_mc = calc_mc_maxcurvature(magnitudes, mag_bin)
    return round(best_mc, 2)


# ---------------------------------------------------------------------------
# b-value — Maximum Likelihood Estimation
# ---------------------------------------------------------------------------

def calc_bvalue_mle(
    magnitudes: List[float],
    mc: Optional[float] = None,
    mag_bin: float = 0.1,
    mc_method: str = "maxcurvature",
) -> BvalueResult:
    """
    Estimate b-value via Maximum Likelihood Estimation (Aki 1965).

    Formula:  b = log10(e) / (mean(M) - (Mc - mag_bin/2))
    Uncertainty (Shi & Bolt 1982):
              σ_b = 2.3 * b² * std(M) / sqrt(N*(N-1))

    Parameters
    ----------
    magnitudes : list of float
        All event magnitudes (filtering to M >= Mc is done internally).
    mc : float, optional
        Completeness magnitude. Estimated automatically if not given.
    mag_bin : float, optional
        Magnitude bin width. Default 0.1.
    mc_method : {'maxcurvature', 'gof'}
        Method used to estimate Mc when not supplied.

    Returns
    -------
    BvalueResult
    """
    mags = np.asarray(magnitudes, dtype=float)

    if mc is None:
        if mc_method == "gof":
            mc = calc_mc_gof(mags.tolist(), mag_bin)
        else:
            mc = calc_mc_maxcurvature(mags.tolist(), mag_bin)
        mc_method_used = mc_method
    else:
        mc_method_used = "user-defined"

    # Filter to complete part
    subset = mags[mags >= mc - mag_bin / 2]
    n = len(subset)
    if n < 3:
        raise ValueError(
            f"Only {n} events with M >= {mc:.2f}. Need at least 3 for b-value estimation."
        )

    mean_m = float(subset.mean())
    # Aki (1965) formula
    b = math.log10(math.e) / (mean_m - (mc - mag_bin / 2))

    # Shi & Bolt (1982) uncertainty
    std_m = float(subset.std(ddof=1)) if n > 1 else 0.0
    b_unc = 2.3 * b ** 2 * std_m / math.sqrt(n * (n - 1)) if n > 1 else float("nan")

    # a-value: log10(N_cumulative at Mc)
    a = math.log10(n) + b * mc

    # Build FMD for plotting
    bins, incremental, cumulative = _build_fmd(mags.tolist(), mag_bin)

    return BvalueResult(
        mc=mc,
        b_value=round(b, 4),
        b_uncertainty=round(b_unc, 4),
        a_value=round(a, 4),
        n_events=n,
        mean_magnitude=round(mean_m, 4),
        method="mle",
        mc_method=mc_method_used,
        mag_bins=bins.tolist(),
        cumulative_n=cumulative.tolist(),
        incremental_n=incremental.tolist(),
    )


# ---------------------------------------------------------------------------
# b-value — Least-Squares Regression
# ---------------------------------------------------------------------------

def calc_bvalue_lsq(
    magnitudes: List[float],
    mc: Optional[float] = None,
    mag_bin: float = 0.1,
    mc_method: str = "maxcurvature",
) -> BvalueResult:
    """
    Estimate b-value via least-squares linear regression on log10(N) vs M.

    Parameters are the same as :func:`calc_bvalue_mle`.

    Returns
    -------
    BvalueResult
    """
    mags = np.asarray(magnitudes, dtype=float)

    if mc is None:
        if mc_method == "gof":
            mc = calc_mc_gof(mags.tolist(), mag_bin)
        else:
            mc = calc_mc_maxcurvature(mags.tolist(), mag_bin)
        mc_method_used = mc_method
    else:
        mc_method_used = "user-defined"

    bins, incremental, cumulative = _build_fmd(mags.tolist(), mag_bin)

    # Only use bins >= Mc with at least 1 event
    mask = (bins >= mc - mag_bin / 2) & (cumulative > 0)
    x = bins[mask]
    y = np.log10(cumulative[mask].astype(float))

    n_events = int(cumulative[mask][0]) if mask.any() else 0
    if len(x) < 2:
        raise ValueError(f"Too few magnitude bins above Mc={mc:.2f} for regression.")

    # Linear regression: y = a - b*x
    coeffs = np.polyfit(x, y, 1)
    b = float(-coeffs[0])
    a = float(coeffs[1])

    # Uncertainty from regression
    y_pred = np.polyval(coeffs, x)
    residuals = y - y_pred
    n = len(x)
    se = math.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
    sxx = float(np.sum((x - x.mean()) ** 2))
    b_unc = se / math.sqrt(sxx) if sxx > 0 else float("nan")

    mean_m = float(mags[mags >= mc - mag_bin / 2].mean()) if n_events > 0 else mc

    return BvalueResult(
        mc=mc,
        b_value=round(b, 4),
        b_uncertainty=round(b_unc, 4),
        a_value=round(a, 4),
        n_events=n_events,
        mean_magnitude=round(mean_m, 4),
        method="lsq",
        mc_method=mc_method_used,
        mag_bins=bins.tolist(),
        cumulative_n=cumulative.tolist(),
        incremental_n=incremental.tolist(),
    )
