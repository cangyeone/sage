---
name: waveform_visualization
category: visualization
keywords: 画图, 绘图, 波形图, 可视化, 粒子运动, 质点运动, plot_stream, plot_spectrogram, plot_psd, plot_particle_motion, savefig
---

# 波形可视化

## 描述

绘制波形时序图、振幅谱图、功率谱密度图和质点运动图，支持震相标注叠加。

---

## 主要函数

### `plot_stream(st, title="", outfile=None, picks=None, show=False)`

绘制多分量波形图（每道一行，纵向排列）。

**参数：**
- `st` : obspy.Stream
- `title` : str — 图标题
- `outfile` : str — 保存路径，如 `"waveform.png"`；为 None 则不保存
- `picks` : list[dict] — 震相标注列表，每条含 `{"phase":"P","time_s":12.5,"confidence":0.92}`
- `show` : bool — 是否弹出交互窗口（服务器环境设 False）

**返回：** matplotlib.figure.Figure

```python
st = read_stream("/data/wave.mseed")
st = detrend_stream(st)
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

# 带震相标注的波形图
picks = [
    {"phase": "P", "time_s": 12.5, "confidence": 0.95},
    {"phase": "S", "time_s": 21.3, "confidence": 0.88},
]
fig = plot_stream(st, title="HHZ 三分量波形", outfile="waveform.png", picks=picks)
print("波形图已保存: waveform.png")
```

---

### `plot_spectrogram(tr, outfile=None, wlen=None, per_lap=0.9, show=False)`

绘制单道波形的时频谱图（短时傅里叶变换）。

**参数：**
- `tr` : obspy.Trace — 单道（从 Stream 中取 `st[0]`）
- `outfile` : str — 保存路径
- `wlen` : float — 时窗长度（秒），None 则自动
- `per_lap` : float — 时窗重叠率，默认 0.9（90%）

```python
st = read_stream("/data/wave.mseed")
tr = st.select(channel="HHZ")[0]  # 取垂直分量
fig = plot_spectrogram(tr, outfile="spectrogram.png")
```

---

### `plot_psd(tr, outfile=None, show=False)`

绘制功率谱密度（PSD）曲线，叠加 NLNM/NHNM 全球噪声模型作为参考。

**参数：**
- `tr` : obspy.Trace

```python
tr = st.select(channel="HHZ")[0]
fig = plot_psd(tr, outfile="psd.png")
```

---

### `plot_particle_motion(st, outfile=None, show=False)`

绘制质点运动图（水平面 N-E 投影 + 垂直面 Z-N 投影）。

**参数：**
- `st` : obspy.Stream — 需要包含三分量（Z/N/E 或 Z/1/2）

```python
st = read_stream_from_dir("/data/event_001/", pattern="**/*.mseed")
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)
fig = plot_particle_motion(st, outfile="particle_motion.png")
```

---

## 完整示例：三图联绘

```python
import os

st = read_stream("/data/event.mseed")
st = detrend_stream(st)
st = taper_stream(st)
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

os.makedirs("output", exist_ok=True)

# 1. 波形图
plot_stream(st, title="事件波形（1-10 Hz）", outfile="output/waveform.png")

# 2. 垂向分量频谱图
tr_z = st.select(channel="*Z")[0]
plot_spectrogram(tr_z, outfile="output/spectrogram.png")

# 3. PSD
plot_psd(tr_z, outfile="output/psd.png")

# 4. 质点运动
plot_particle_motion(st, outfile="output/particle_motion.png")

print("所有图像已保存到 output/")
```

---

## 注意事项

- 服务器/无界面环境须设 `show=False`，否则会卡住等待 GUI
- 绘制质点运动图需要三分量数据，缺分量时会跳过对应投影
- `outfile` 路径的目录须预先存在（用 `os.makedirs` 创建）
