---
name: source_parameters
category: analysis
keywords: 震级, 矩震级, 地震矩, 震源参数, 角频率, 应力降, ML震级, 拐角频率, estimate_magnitude_ml, estimate_corner_freq, estimate_seismic_moment, moment_to_mw, estimate_stress_drop
---

# 震源参数估算

## 描述

估算本地震级（ML）、地震矩（M₀）、矩震级（Mw）、角频率（fc）和应力降（Δσ），基于 ObsPy Trace 直接计算。

---

## 主要函数

### `estimate_magnitude_ml(tr, dist_km, station_correction=0.0)`

基于 Wood-Anderson 仿真计算本地震级（ML，Richter 1935）。

**参数：**
- `tr` : obspy.Trace — 速度记录（单分量）
- `dist_km` : float — 震中距（千米）
- `station_correction` : float — 台站校正值，默认 0

**返回：** float — ML 震级

```python
tr = st.select(channel="HHZ")[0]
dist_km = 85.3  # 震中距

ml = estimate_magnitude_ml(tr, dist_km=dist_km)
print(f"ML = {ml:.2f}")
```

---

### `estimate_corner_freq(tr, dist_km, method="brune", density=2700, vp=6000)`

用 Brune 震源模型估算角频率（拐角频率）fc。

**参数：**
- `tr` : obspy.Trace — 速度记录（P 波窗口内）
- `dist_km` : float — 震中距（千米）
- `method` : str — 震源模型，`"brune"`
- `density` : float — 介质密度（kg/m³），默认 2700
- `vp` : float — P 波速度（m/s），默认 6000

**返回：** `(fc, omega0)` — 角频率（Hz）和低频谱振幅 Ω₀

```python
fc, omega0 = estimate_corner_freq(tr, dist_km=dist_km)
print(f"角频率 fc = {fc:.2f} Hz")
print(f"低频谱振幅 Ω₀ = {omega0:.3e}")
```

---

### `estimate_seismic_moment(tr, dist_km, density=2700, vp=6000, radiation=0.52)`

用远场位移谱估算地震矩 M₀。

**参数：**
- `tr` : obspy.Trace — 位移记录（已去仪器响应）
- `dist_km` : float — 震中距（千米）
- `density` : float — 介质密度（kg/m³）
- `vp` : float — P 波速度（m/s）
- `radiation` : float — 辐射花样系数，P 波默认 0.52

**返回：** float — 地震矩 M₀（N·m）

```python
M0 = estimate_seismic_moment(tr, dist_km=dist_km)
print(f"地震矩 M₀ = {M0:.3e} N·m")
```

---

### `moment_to_mw(M0)`

将地震矩转换为矩震级。

**公式：** Mw = (2/3) × log₁₀(M₀) − 6.07

**参数：**
- `M0` : float — 地震矩（N·m）

**返回：** float — 矩震级 Mw

```python
Mw = moment_to_mw(M0)
print(f"矩震级 Mw = {Mw:.2f}")
```

---

### `estimate_stress_drop(M0, fc, vs=3500)`

用 Brune 模型估算静态应力降 Δσ。

**公式：** Δσ = (7/16) × M₀ × (fc/vs)³

**参数：**
- `M0` : float — 地震矩（N·m）
- `fc` : float — 角频率（Hz）
- `vs` : float — S 波速度（m/s），默认 3500

**返回：** float — 应力降（MPa）

```python
delta_sigma = estimate_stress_drop(M0, fc, vs=3500)
print(f"应力降 Δσ = {delta_sigma:.2f} MPa")
```

---

## 完整震源参数分析示例

```python
from obspy import read_inventory

# 1. 读取波形并去仪器响应
st = read_stream("/data/event.mseed")
inv = read_inventory("/data/station.xml")

st_vel = st.copy()
st_vel = detrend_stream(st_vel)
st_vel = taper_stream(st_vel)
st_vel = remove_response(st_vel, inv, output="VEL", pre_filt=(0.01, 0.02, 45, 50))

st_disp = st.copy()
st_disp = detrend_stream(st_disp)
st_disp = taper_stream(st_disp)
st_disp = remove_response(st_disp, inv, output="DISP", pre_filt=(0.01, 0.02, 45, 50))

# 2. 取垂向分量
dist_km = 120.0
tr_vel  = st_vel.select(channel="*Z")[0]
tr_disp = st_disp.select(channel="*Z")[0]

# 3. 计算各震源参数
ML  = estimate_magnitude_ml(tr_vel, dist_km=dist_km)
fc, omega0 = estimate_corner_freq(tr_vel, dist_km=dist_km)
M0  = estimate_seismic_moment(tr_disp, dist_km=dist_km)
Mw  = moment_to_mw(M0)
ds  = estimate_stress_drop(M0, fc)

print(f"--- 震源参数汇总 ---")
print(f"本地震级  ML  = {ML:.2f}")
print(f"矩震级    Mw  = {Mw:.2f}")
print(f"地震矩    M₀  = {M0:.3e} N·m")
print(f"角频率    fc  = {fc:.2f} Hz")
print(f"应力降    Δσ  = {ds:.2f} MPa")
```

---

## 注意事项

- `estimate_magnitude_ml` 内部自动做 Wood-Anderson 仪器仿真，输入须为速度记录
- `estimate_seismic_moment` 要求输入为位移记录（已去仪器响应），否则结果偏差极大
- 单台估算结果不确定性较大，建议多台平均
