---
name: b_value_analysis
category: statistics
keywords: b value, b-value, Gutenberg-Richter, frequency-magnitude distribution, FMD, minimum magnitude of completeness, Mc, seismic statistics, calc_bvalue_mle, calc_bvalue_lsq, load_catalog_file, plot_gr, b值, Gutenberg-Richter, 频度震级分布, 最小完整性震级, 地震统计
---

# B-value Analysis

## Description

Calculate the b-value from the Gutenberg-Richter relationship using earthquake catalogs, estimate the magnitude of completeness (Mc), and plot frequency-magnitude distribution.

---

## Main Functions

### `load_catalog_file(path)`

Auto-detect format and load earthquake catalog.

**Parameters:**
- `path` : str — Path to catalog file, supports `.csv` / `.json` / `.txt` (phase pick format)

**Returns:** `CatalogData` object with attributes `.magnitudes`, `.depths`, `.lats`, `.lons`, `.times`, etc.

```python
from seismo_stats.catalog_loader import load_catalog_file
catalog = load_catalog_file("/data/catalog.csv")
print(catalog.summary())
# Output: CatalogData: 1523 events, mag range [0.2, 5.1], ...
```

---

### `calc_mc_maxcurvature(magnitudes, bin_width=0.1)`

Estimate minimum magnitude of completeness (Mc) using maximum curvature method.

**Parameters:**
- `magnitudes` : array-like — List of magnitudes
- `bin_width` : float — Magnitude bin width, default 0.1

**Returns:** float — Estimated Mc value

```python
from seismo_stats.bvalue import calc_mc_maxcurvature
import numpy as np
mags = catalog.magnitudes
Mc = calc_mc_maxcurvature(mags)
print(f"Maximum curvature Mc = {Mc:.1f}")
```

---

### `calc_mc_gof(magnitudes, bin_width=0.1, conf_level=0.9)`

Estimate Mc using goodness-of-fit method, more robust than maximum curvature.

**Parameters:**
- `magnitudes` : array-like
- `bin_width` : float — Default 0.1
- `conf_level` : float — Confidence level, default 0.9 (90%)

**Returns:** `(Mc, R_best)` — Mc value and optimal fit residual

```python
from seismo_stats.bvalue import calc_mc_gof
Mc, R = calc_mc_gof(mags, conf_level=0.9)
print(f"Goodness-of-fit Mc = {Mc:.1f}  (residual R = {R:.3f})")
```

---

### `calc_bvalue_mle(magnitudes, Mc, bin_width=0.1)`

Calculate b-value using maximum likelihood estimation (MLE / Aki 1965), highest precision.

**Parameters:**
- `magnitudes` : array-like
- `Mc` : float — Magnitude of completeness
- `bin_width` : float — Default 0.1

**Returns:** `BvalueResult` object

**BvalueResult attributes:**
- `.b` — b-value
- `.sigma_b` — Standard error
- `.a` — a-value (G-R relationship intercept)
- `.n_events` — Number of events used for calculation
- `.Mc` — Mc value used

```python
from seismo_stats.bvalue import calc_bvalue_mle
result = calc_bvalue_mle(mags, Mc=Mc)
print(f"b = {result.b:.3f} ± {result.sigma_b:.3f}")
print(f"a = {result.a:.3f}")
print(f"Events used: {result.n_events}")
```

---

### `calc_bvalue_lsq(magnitudes, Mc, bin_width=0.1)`

Calculate b-value using least-squares fitting of G-R relationship, suitable for method comparison.

**Parameters:** Same as `calc_bvalue_mle`

**Returns:** `BvalueResult`

```python
from seismo_stats.bvalue import calc_bvalue_lsq
result_lsq = calc_bvalue_lsq(mags, Mc=Mc)
print(f"LSQ b = {result_lsq.b:.3f}")
```

---

### `plot_gr(catalog_or_mags, Mc=None, outfile=None)`

Plot frequency-magnitude distribution (G-R plot) with automatic fitting and b-value annotation.

**Parameters:**
- `catalog_or_mags` : CatalogData or array-like (list of magnitudes)
- `Mc` : float — If None, auto-estimated
- `outfile` : str — Save path

```python
from seismo_stats.plotting import plot_gr
fig = plot_gr(catalog, Mc=Mc, outfile="fmd.png")
print("F-M distribution plot saved: fmd.png")
```

---

## Complete Example

```python
from seismo_stats.catalog_loader import load_catalog_file
from seismo_stats.bvalue import calc_mc_maxcurvature, calc_mc_gof, calc_bvalue_mle
from seismo_stats.plotting import plot_gr, plot_temporal, plot_all

# 1. Load catalog
catalog = load_catalog_file("/data/catalog.csv")
print(catalog.summary())
mags = catalog.magnitudes

# 2. Estimate Mc (compare two methods)
Mc_mc = calc_mc_maxcurvature(mags)
Mc_gof, R = calc_mc_gof(mags)
print(f"Maximum curvature Mc = {Mc_mc:.1f}")
print(f"Goodness-of-fit Mc = {Mc_gof:.1f}  (R = {R:.3f})")
Mc = Mc_gof  # Use goodness-of-fit result

# 3. Calculate b-value
result = calc_bvalue_mle(mags, Mc=Mc)
print(f"\nb-value (MLE) = {result.b:.3f} ± {result.sigma_b:.3f}")
print(f"a-value       = {result.a:.3f}")
print(f"N (M>=Mc)   = {result.n_events}")

# 4. Plot
plot_gr(catalog, Mc=Mc, outfile="fmd.png")
plot_temporal(catalog, outfile="temporal.png")

print("\nAnalysis complete, plots saved.")
```

---

## Notes

- MLE method (Aki 1965) is statistically superior to LSQ; recommended as primary method
- When event count < 50, b-value is unstable; results for reference only
- When using `bin_width=0.1`, ensure catalog magnitude precision is 0.1
