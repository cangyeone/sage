---
name: b_value_analysis
category: statistics
keywords: b值, b-value, Gutenberg-Richter, 频度震级分布, FMD, 最小完整性震级, Mc, 地震统计, calc_bvalue_mle, calc_bvalue_lsq, load_catalog_file, plot_gr
---

# b 值分析

## 描述

从地震目录计算 Gutenberg-Richter 关系中的 b 值，估算最小完整性震级 Mc，绘制频度-震级分布图。

---

## 主要函数

### `load_catalog_file(path)`

自动识别格式并加载地震目录。

**参数：**
- `path` : str — 目录文件路径，支持 `.csv` / `.json` / `.txt`（震相拾取格式）

**返回：** `CatalogData` 对象，含 `.magnitudes`、`.depths`、`.lats`、`.lons`、`.times` 等属性

```python
from seismo_stats.catalog_loader import load_catalog_file
catalog = load_catalog_file("/data/catalog.csv")
print(catalog.summary())
# 输出: CatalogData: 1523 events, mag range [0.2, 5.1], ...
```

---

### `calc_mc_maxcurvature(magnitudes, bin_width=0.1)`

用最大曲率法估算最小完整性震级 Mc。

**参数：**
- `magnitudes` : array-like — 震级列表
- `bin_width` : float — 震级分箱宽度，默认 0.1

**返回：** float — Mc 估算值

```python
from seismo_stats.bvalue import calc_mc_maxcurvature
import numpy as np
mags = catalog.magnitudes
Mc = calc_mc_maxcurvature(mags)
print(f"最大曲率法 Mc = {Mc:.1f}")
```

---

### `calc_mc_gof(magnitudes, bin_width=0.1, conf_level=0.9)`

用拟合优度法（Goodness-of-fit）估算 Mc，比最大曲率法更稳健。

**参数：**
- `magnitudes` : array-like
- `bin_width` : float — 默认 0.1
- `conf_level` : float — 置信水平，默认 0.9（90%）

**返回：** `(Mc, R_best)` — Mc 值和最优拟合残差

```python
from seismo_stats.bvalue import calc_mc_gof
Mc, R = calc_mc_gof(mags, conf_level=0.9)
print(f"拟合优度法 Mc = {Mc:.1f}  (残差 R = {R:.3f})")
```

---

### `calc_bvalue_mle(magnitudes, Mc, bin_width=0.1)`

用最大似然估计（MLE / Aki 1965）计算 b 值，精度最高。

**参数：**
- `magnitudes` : array-like
- `Mc` : float — 最小完整性震级
- `bin_width` : float — 默认 0.1

**返回：** `BvalueResult` 对象

**BvalueResult 属性：**
- `.b` — b 值
- `.sigma_b` — 标准误差
- `.a` — a 值（G-R 关系截距）
- `.n_events` — 用于计算的事件数
- `.Mc` — 使用的 Mc

```python
from seismo_stats.bvalue import calc_bvalue_mle
result = calc_bvalue_mle(mags, Mc=Mc)
print(f"b = {result.b:.3f} ± {result.sigma_b:.3f}")
print(f"a = {result.a:.3f}")
print(f"使用事件数: {result.n_events}")
```

---

### `calc_bvalue_lsq(magnitudes, Mc, bin_width=0.1)`

用最小二乘法拟合 G-R 关系，适用于比较不同方法。

**参数：** 同 `calc_bvalue_mle`

**返回：** `BvalueResult`

```python
from seismo_stats.bvalue import calc_bvalue_lsq
result_lsq = calc_bvalue_lsq(mags, Mc=Mc)
print(f"LSQ b = {result_lsq.b:.3f}")
```

---

### `plot_gr(catalog_or_mags, Mc=None, outfile=None)`

绘制频度-震级分布图（G-R 图），自动拟合并标注 b 值。

**参数：**
- `catalog_or_mags` : CatalogData 或 array-like（震级列表）
- `Mc` : float — 若 None 则自动估算
- `outfile` : str — 保存路径

```python
from seismo_stats.plotting import plot_gr
fig = plot_gr(catalog, Mc=Mc, outfile="fmd.png")
print("F-M 分布图已保存: fmd.png")
```

---

## 完整示例

```python
from seismo_stats.catalog_loader import load_catalog_file
from seismo_stats.bvalue import calc_mc_maxcurvature, calc_mc_gof, calc_bvalue_mle
from seismo_stats.plotting import plot_gr, plot_temporal, plot_all

# 1. 加载目录
catalog = load_catalog_file("/data/catalog.csv")
print(catalog.summary())
mags = catalog.magnitudes

# 2. 估算 Mc（两种方法对比）
Mc_mc = calc_mc_maxcurvature(mags)
Mc_gof, R = calc_mc_gof(mags)
print(f"最大曲率法 Mc = {Mc_mc:.1f}")
print(f"拟合优度法 Mc = {Mc_gof:.1f}  (R = {R:.3f})")
Mc = Mc_gof  # 使用拟合优度法结果

# 3. 计算 b 值
result = calc_bvalue_mle(mags, Mc=Mc)
print(f"\nb 值 (MLE) = {result.b:.3f} ± {result.sigma_b:.3f}")
print(f"a 值       = {result.a:.3f}")
print(f"N (M≥Mc)   = {result.n_events}")

# 4. 绘图
plot_gr(catalog, Mc=Mc, outfile="fmd.png")
plot_temporal(catalog, outfile="temporal.png")

print("\n分析完成，图像已保存。")
```

---

## 注意事项

- MLE 方法（Aki 1965）比 LSQ 统计上更优，推荐作为主要方法
- 事件数 < 50 时 b 值不稳定，结果仅供参考
- 使用 `bin_width=0.1` 时须确保目录震级精度为 0.1
