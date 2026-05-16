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

### `calc_mc_maxcurvature(magnitudes, mag_bin=0.1, correction=0.2)`

Estimate minimum magnitude of completeness (Mc) using maximum curvature method.

**Parameters:**
- `magnitudes` : array-like — List of magnitudes
- `mag_bin` : float — Magnitude bin width, default 0.1
- `correction` : float — Bias correction added to raw maximum-curvature Mc, default 0.2

**Returns:** float — Estimated Mc value

```python
from seismo_stats.bvalue import calc_mc_maxcurvature
import numpy as np
mags = catalog.magnitudes
Mc = calc_mc_maxcurvature(mags, mag_bin=0.1)
print(f"Maximum curvature Mc = {Mc:.1f}")
```

---

### `calc_mc_gof(magnitudes, mag_bin=0.1, r_threshold=95.0)`

Estimate Mc using goodness-of-fit method, more robust than maximum curvature.

**Parameters:**
- `magnitudes` : array-like
- `mag_bin` : float — Magnitude bin width, default 0.1
- `r_threshold` : float — Required percentage of data fit, default 95.0

**Returns:** float — Estimated Mc value

```python
from seismo_stats.bvalue import calc_mc_gof
Mc = calc_mc_gof(mags, mag_bin=0.1, r_threshold=95.0)
print(f"Goodness-of-fit Mc = {Mc:.1f}")
```

---

### `calc_bvalue_mle(magnitudes, mc=None, mag_bin=0.1, mc_method="maxcurvature")`

Calculate b-value using maximum likelihood estimation (MLE / Aki 1965), highest precision.

**Parameters:**
- `magnitudes` : array-like
- `mc` : float or None — Magnitude of completeness; estimated automatically when None
- `mag_bin` : float — Magnitude bin width, default 0.1
- `mc_method` : `"maxcurvature"` or `"gof"` — Method used when `mc` is None

**Returns:** `BvalueResult` object

**BvalueResult attributes:**
- `.b_value` — b-value
- `.b_uncertainty` — Standard error of b (Shi & Bolt 1982)
- `.a_value` — a-value (G-R relationship intercept)
- `.n_events` — Number of events used for calculation
- `.mc` — Mc value used
- `.mean_magnitude` — Mean magnitude of complete events
- `.method` — `"mle"` or `"lsq"`
- `.mc_method` — Mc source/method

```python
from seismo_stats.bvalue import calc_bvalue_mle
result = calc_bvalue_mle(mags, mc=Mc, mag_bin=0.1)
print(f"b = {result.b_value:.3f} ± {result.b_uncertainty:.3f}")
print(f"a = {result.a_value:.3f}")
print(f"Events used: {result.n_events}")
```

---

### `calc_bvalue_lsq(magnitudes, mc=None, mag_bin=0.1, mc_method="maxcurvature")`

Calculate b-value using least-squares fitting of G-R relationship, suitable for method comparison.

**Parameters:** Same as `calc_bvalue_mle`

**Returns:** `BvalueResult`

```python
from seismo_stats.bvalue import calc_bvalue_lsq
result_lsq = calc_bvalue_lsq(mags, mc=Mc, mag_bin=0.1)
print(f"LSQ b = {result_lsq.b_value:.3f}")
```

---

### `plot_gr(result, output_path, title=None)`

Plot frequency-magnitude distribution (G-R plot) from a `BvalueResult` with b-value annotation.

**Parameters:**
- `result` : `BvalueResult` returned by `calc_bvalue_mle` or `calc_bvalue_lsq`
- `output_path` : str — Save path
- `title` : str or None — Optional plot title

```python
from seismo_stats.plotting import plot_gr
plot_gr(result, "fmd.png")
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
Mc_mc = calc_mc_maxcurvature(mags, mag_bin=0.1)
Mc_gof = calc_mc_gof(mags, mag_bin=0.1, r_threshold=95.0)
print(f"Maximum curvature Mc = {Mc_mc:.1f}")
print(f"Goodness-of-fit Mc = {Mc_gof:.1f}")
Mc = Mc_gof  # Use goodness-of-fit result

# 3. Calculate b-value
result = calc_bvalue_mle(mags, mc=Mc, mag_bin=0.1)
print(f"\nb-value (MLE) = {result.b_value:.3f} ± {result.b_uncertainty:.3f}")
print(f"a-value       = {result.a_value:.3f}")
print(f"N (M>=Mc)   = {result.n_events}")

# 4. Plot
plot_gr(result, "fmd.png")
plot_temporal(catalog, "temporal.png")
plot_all(result, catalog, "catalog_summary")

print("\nAnalysis complete, plots saved.")
```

---

## Notes

- MLE method (Aki 1965) is statistically superior to LSQ; recommended as primary method
- When event count < 50, b-value is unstable; results for reference only
- When using `mag_bin=0.1`, ensure catalog magnitude precision is 0.1
- The MLE uncertainty follows Shi & Bolt (1982)
