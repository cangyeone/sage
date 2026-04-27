---
name: gmt_terrain_map
title: GMT 地形图绘制工作流
version: "1.0"
description: 使用 GMT classic mode 绘制带色彩地形底图的完整参考流程，包含 DEM 渲染、海岸线叠加、等高线和图例。
keywords:
  - GMT
  - terrain
  - topography
  - 地形图
  - 地形
  - terrain map
  - 地形底图
  - China map
  - 中国地形
  - topo
  - DEM
  - grdimage
  - coast
skills:
  - name: gmt_plotting
    role: 地图底图、海岸线、比例尺等通用绘图层
  - name: _gen_gmt_docs_6_5
    role: GMT 6.5 官方 API 文档 RAG 参考
steps:
  - id: prepare_cpt
    skill: gmt_plotting
    description: 生成地形色标（CPT 文件）
  - id: cut_dem
    skill: gmt_plotting
    description: 裁剪 DEM 数据到研究区域
    depends_on: [prepare_cpt]
  - id: render_terrain
    skill: gmt_plotting
    description: grdimage 渲染地形底图（首层，仅加 -K）
    depends_on: [cut_dem]
  - id: add_coast
    skill: gmt_plotting
    description: pscoast 叠加海岸线与国界（-O -K）
    depends_on: [render_terrain]
  - id: add_contours
    skill: gmt_plotting
    description: grdcontour 叠加等高线（可选，-O -K）
    depends_on: [render_terrain]
  - id: add_scale_legend
    skill: gmt_plotting
    description: psscale + psbasemap 添加色标、比例尺、指北针（-O -K）
    depends_on: [add_coast]
  - id: close_and_export
    skill: gmt_plotting
    description: 关闭 PostScript 末层（-O，无 -K）并 ps2raster 转 PNG
    depends_on: [add_scale_legend, add_contours]
---

## GMT 地形图绘制步骤

> **模式选择**：本工作流使用 **classic mode**（`gmt pscoast`、`-K -O` 标志）。
> 若需 modern mode，使用 `gmt begin/end` 并去掉模块名的 `ps` 前缀（如 `gmt coast`）。
> **严禁混用两种模式**。

---

### Step 1: 准备地形色标

```bash
# geo 色表：海洋蓝 + 陆地绿棕，适合全球/区域地形
gmt makecpt -Cgeo -T-8000/5000/100 -Z > topo.cpt

# 仅陆地地形可用 topo 或 etopo1
# gmt makecpt -Ctopo -T0/5000/50 -Z > topo.cpt
```

---

### Step 2: 下载/裁剪 DEM 数据

```bash
# 使用 GMT 内置远程 DEM（需联网，第一次下载后缓存）
# 1 弧分 (~1.8km): @earth_relief_01m
# 15 弧秒 (~450m): @earth_relief_15s
# 3 弧秒 (~90m):   @earth_relief_03s

# 裁剪到研究区域（中国: 70/140/10/55）
gmt grdcut @earth_relief_01m -R70/140/10/55 -Gchina_topo.grd
```

---

### Step 3: 渲染地形底图（grdimage）

```bash
# classic mode — 第一层，需要 -K，不加 -O
gmt grdimage china_topo.grd -Ctopo.cpt -JM15c -R70/140/10/55 \
    -Baf -BWSen -K > map.ps
```

---

### Step 4: 叠加海岸线和国界

```bash
# classic mode 用 pscoast；加 -O（续接）-K（后续还有层）
gmt pscoast -R -J -Df -W0.5p,black -N1/0.8p,gray40 \
    -S#a0d0f0 -O -K >> map.ps
```

---

### Step 5: 叠加等高线（可选）

```bash
gmt grdcontour china_topo.grd -R -J \
    -C1000 -A2000+f7p,Helvetica,black -Wc0.3p,gray50 \
    -O -K >> map.ps
```

---

### Step 6: 叠加震点/站点数据（可选）

```bash
# 震中分布：lon lat depth mag
gmt psxy epicenters.txt -R -J -Sc0.15c -Cquake_depth.cpt \
    -Wfaint -O -K >> map.ps
```

---

### Step 7: 添加色标、比例尺、指北针

```bash
# 色标（地形）
gmt psscale -Ctopo.cpt -Dx1c/1c+w6c/0.3c+jBL+h \
    -Bxa2000f500+l"海拔 (m)" -O -K >> map.ps

# 比例尺
gmt psbasemap -R -J -Lx13c/1c+c30N+w500k+lkm+f -O -K >> map.ps
```

---

### Step 8: 关闭 PostScript 并转换格式

```bash
# 最后一层不加 -K，只加 -O
echo | gmt psxy -R -J -O >> map.ps

# 转 PNG（-A 裁白边，-E300 300dpi）
gmt ps2raster map.ps -A -Tg -E300
```

---

## 常见错误与修复

| 错误信息 | 原因 | 修复方法 |
|---|---|---|
| `gmt coast: unrecognized option` | 混用了 modern mode 模块名 | classic mode 用 `pscoast`，不是 `coast` |
| `No title: psbasemap` | 缺少 `-B` | 加 `-Baf` 或 `-B+t"标题"` |
| 输出空白 | `-K -O` 顺序错误 | 首层只加 `-K`；中间层加 `-O -K`；末层只加 `-O` |
| DEM 全为 NaN | 裁剪范围超出数据覆盖 | 检查 `-R` 范围与 DEM 覆盖范围 |
| 色标偏色 | CPT 范围与数据不符 | 用 `gmt grdinfo topo.grd` 查实际高程范围再设 `-T` |
