---
name: spectral_analysis
category: analysis
keywords: 频谱, 功率谱, 振幅谱, HVSR, 谱比法, 场地响应, 卓越周期, 傅里叶, FFT, compute_spectrum, compute_hvsr, plot_psd
---

# 频谱分析与 HVSR

## 描述

计算波形的振幅谱、功率谱，以及水平-垂直谱比（HVSR）用于场地响应和卓越频率分析。

---

## 主要函数

### `compute_spectrum(tr, method="fft")`

计算单道波形的振幅谱。

**参数：**
- `tr` : obspy.Trace
- `method` : str — `"fft"`（快速傅里叶）

**返回：** `(freqs, amps)` — numpy 数组，频率（Hz）和对应振幅

```python
tr = st.select(channel="HHZ")[0]
freqs, amps = compute_spectrum(tr)

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.semilogy(freqs, amps)
plt.xlabel("频率 (Hz)")
plt.ylabel("振幅")
plt.title("振幅谱")
plt.xlim(0.1, 50)
plt.grid(True, alpha=0.3)
plt.savefig("spectrum.png", dpi=150)
print(f"主频: {freqs[np.argmax(amps)]:.2f} Hz")
```

---

### `compute_hvsr(st, method="konno_ohmachi", f_min=0.1, f_max=20.0, smooth_coeff=40, window_length=20.0, overlap=0.5)`

计算水平-垂直谱比（HVSR / H/V），用于估算场地卓越频率和场地放大效应。

**参数：**
- `st` : obspy.Stream — 须含三分量（Z + 两水平）
- `method` : str — 平滑方法，`"konno_ohmachi"` / `"constant"` / `"proportional"`
- `f_min` : float — 分析频率下限（Hz），默认 0.1
- `f_max` : float — 分析频率上限（Hz），默认 20.0
- `smooth_coeff` : float — Konno-Ohmachi 平滑系数 b，默认 40
- `window_length` : float — 分窗长度（秒），默认 20.0
- `overlap` : float — 分窗重叠率，默认 0.5

**返回：** `(freqs, hvsr_mean, hvsr_std)` — 频率数组、H/V 均值、标准差

```python
st = read_stream_from_dir("/data/station/", pattern="**/*.mseed")
st = detrend_stream(st)
st = taper_stream(st)

freqs, hvsr, hvsr_std = compute_hvsr(
    st,
    method="konno_ohmachi",
    f_min=0.1,
    f_max=20.0,
    window_length=30.0,
    overlap=0.5,
)

# 找卓越频率
import numpy as np
f0_idx = np.argmax(hvsr)
f0 = freqs[f0_idx]
print(f"场地卓越频率 f0 = {f0:.3f} Hz  (T0 = {1/f0:.2f} s)")
print(f"卓越频率处 H/V = {hvsr[f0_idx]:.2f} ± {hvsr_std[f0_idx]:.2f}")

# 绘图
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(freqs, hvsr, "b-", lw=2, label="H/V 均值")
ax.fill_between(freqs, hvsr - hvsr_std, hvsr + hvsr_std, alpha=0.3, color="blue", label="±1σ")
ax.axvline(f0, color="red", ls="--", label=f"f₀ = {f0:.3f} Hz")
ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("H/V 比值")
ax.set_title("HVSR 谱比曲线")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("hvsr.png", dpi=150, bbox_inches="tight")
print("HVSR 图已保存: hvsr.png")
```

---

## PSD 与全球噪声模型对比

```python
# 绘制 PSD 并与 NLNM/NHNM 对比（内置参考模型）
tr = st.select(channel="*Z")[0]
fig = plot_psd(tr, outfile="psd_nlnm.png")
print("PSD 图已保存: psd_nlnm.png")
```

---

## 多台站频谱对比

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5))
for tr in st:
    freqs, amps = compute_spectrum(tr)
    ax.semilogy(freqs, amps, label=tr.id, alpha=0.8)

ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("振幅")
ax.set_title("多台站振幅谱对比")
ax.legend(fontsize=8)
ax.set_xlim(0.1, 50)
ax.grid(True, alpha=0.3)
plt.savefig("multi_spectrum.png", dpi=150, bbox_inches="tight")
```

---

## 注意事项

- HVSR 分析要求数据长度 ≥ 10 倍窗长，建议使用 ≥ 30 分钟的背景噪声记录
- 三分量采样率必须一致，否则先用 `resample_stream` 统一
- `f_min` 应 ≥ 3/记录长度，避免频率分辨率不足
