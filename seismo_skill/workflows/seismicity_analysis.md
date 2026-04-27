---
name: seismicity_analysis
title: 地震目录分析与震中分布图工作流
version: "1.0"
description: 从地震目录到震中分布图、深度剖面和 b 值分析的完整工作流，涵盖数据读取、过滤、GMT 绘图和统计分析。
keywords:
  - seismicity
  - earthquake catalog
  - GMT
  - epicenter
  - 震中分布
  - 地震目录
  - b值
  - b value
  - magnitude
  - depth
  - focal mechanism
  - 震源机制
skills:
  - name: gmt_plotting
    role: 震中分布图、深度剖面图、断层叠加
  - name: tabular_io
    role: 地震目录 CSV/TXT 读取、列名规范化、时间解析
  - name: b_value_analysis
    role: Gutenberg-Richter b 值、Mc 估算、频度-震级关系
steps:
  - id: load_catalog
    skill: tabular_io
    description: 读取并规范化地震目录（列名、时间格式、Mc 过滤）
  - id: epicenter_map
    skill: gmt_plotting
    description: GMT 绘制震中分布图（按深度着色）
    depends_on: [load_catalog]
  - id: time_series
    skill: tabular_io
    description: 震级-深度时间序列图（matplotlib）
    depends_on: [load_catalog]
  - id: b_value
    skill: b_value_analysis
    description: 最大曲率法估算 Mc，拟合 Gutenberg-Richter b 值
    depends_on: [load_catalog]
  - id: cross_section
    skill: gmt_plotting
    description: 沿测线投影深度剖面图
    depends_on: [load_catalog, b_value]
---

## 地震目录分析工作流

---

### Step 1: 读取并规范化地震目录

用 `tabular_io` skill 指导 code engine 生成读取代码。关键操作：

- `pd.read_csv()` / `pd.read_table()` 读取目录文件
- 统一列名为 `lon`, `lat`, `depth`, `magnitude`, `time`
- `pd.to_datetime()` 解析时间列
- 按完整性震级 Mc 过滤（`df[df.magnitude >= Mc]`）

```python
import pandas as pd

df = pd.read_csv("catalog.csv")
df = df.rename(columns={"longitude":"lon","latitude":"lat","depth_km":"depth","ml":"magnitude"})
df["time"] = pd.to_datetime(df["time"])
Mc = 2.0
df = df[df["magnitude"] >= Mc].copy()
print(f"筛选后: {len(df)} 条地震，M≥{Mc}")
```

---

### Step 2: 震中分布图（GMT classic mode）

用 `gmt_plotting` skill 指导 code engine 生成 GMT 脚本，通过 `run_gmt()` 执行。

```bash
# 按深度着色，按震级缩放圆圈大小
gmt makecpt -Chot -T0/100/5 -Z -I > depth.cpt
gmt pscoast -R100/130/20/50 -JM15c -Df -W0.5p -G#f5f5dc -S#a0c8f0 -Baf -BWSen -K > seismicity.ps
awk '{print $1,$2,$3,$4*0.1}' epicenter.txt | gmt psxy -R -J -Scc -Cdepth.cpt -Wfaint,black -O -K >> seismicity.ps
gmt psscale -Cdepth.cpt -Dx1c/1c+w5c/0.3c+jBL+h -Bxa20f5+l"深度 (km)" -O -K >> seismicity.ps
echo | gmt psxy -R -J -O >> seismicity.ps
gmt ps2raster seismicity.ps -A -Tg -E300
```

---

### Step 3: 深度-震级时间序列图

```python
import matplotlib.pyplot as plt, matplotlib.dates as mdates

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].scatter(df["time"], df["magnitude"], s=2, alpha=0.5)
axes[0].set_ylabel("震级 M"); axes[0].set_title("震级时间序列")
axes[1].scatter(df["time"], df["depth"], s=2, alpha=0.5, color="orange")
axes[1].invert_yaxis(); axes[1].set_ylabel("深度 (km)")
df_s = df.sort_values("time")
axes[2].plot(df_s["time"], range(1, len(df_s)+1)); axes[2].set_ylabel("累积次数")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout(); plt.savefig("timeseries.png", dpi=200)
```

---

### Step 4: b 值分析（Gutenberg-Richter）

用 `b_value_analysis` skill 指导 code engine。流程：最大曲率法估算 Mc → 最小二乘拟合 G-R 关系 → 绘制频度-震级图。

```python
import numpy as np, matplotlib.pyplot as plt

mags = df["magnitude"].values
bins = np.arange(mags.min(), mags.max()+0.1, 0.1)
counts, edges = np.histogram(mags, bins=bins)
Mc = edges[np.argmax(counts)]
m_above = mags[mags >= Mc]
b = np.log10(np.e) / (np.mean(m_above) - (Mc - 0.05))
a = np.log10(len(m_above)) + b * Mc
print(f"Mc={Mc:.1f}  b={b:.2f}  a={a:.2f}")
```

---

### Step 5: 深度剖面（沿测线投影）

```python
import numpy as np

p1, p2 = (104.0, 25.0), (110.0, 35.0)
dx,dy = p2[0]-p1[0], p2[1]-p1[1]
L = np.hypot(dx,dy); ux,uy = dx/L, dy/L
df["dist_km"] = ((df.lon-p1[0])*ux + (df.lat-p1[1])*uy)*111.0
perp = (df.lon-p1[0])*(-uy) + (df.lat-p1[1])*ux
df_cross = df[np.abs(perp*111.0) <= 50].copy()
fig,ax = plt.subplots(figsize=(12,4))
sc = ax.scatter(df_cross.dist_km, df_cross.depth, c=df_cross.magnitude, cmap="hot_r", s=10)
ax.invert_yaxis(); ax.set_xlabel("距离 (km)"); ax.set_ylabel("深度 (km)")
plt.colorbar(sc,ax=ax,label="震级"); plt.savefig("cross_section.png",dpi=200)
```

---

## 调度注意事项

- **Mc 完整性**是 b 值可靠的前提，step 4 依赖 step 1 的 Mc 过滤结果
- **深度剖面半宽**根据研究区地震带宽度调整（通常 20–100 km）
- **时间序列**建议全程使用 UTC，避免时区混乱
- step 2 的 GMT 脚本由 code engine 生成并通过 `run_gmt()` 工具执行
