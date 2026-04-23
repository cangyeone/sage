---
name: waveform_processing
category: waveform
keywords: 滤波, 带通滤波, 低通滤波, 高通滤波, 去趋势, 去均值, detrend, demean, 重采样, resample, taper, 仪器响应, 预处理, filter_stream, detrend_stream, resample_stream, trim_stream
---

# 波形预处理

## 描述

对 ObsPy Stream/Trace 进行去趋势、去均值、taper、滤波、重采样、截窗等标准预处理操作。

---

## 主要函数

### `detrend_stream(st, type="demean")`

去趋势或去均值。

**参数：**
- `st` : obspy.Stream
- `type` : str — `"demean"`（去均值）/ `"linear"`（去线性趋势）/ `"constant"`（同 demean）

**返回：** obspy.Stream（原地修改并返回）

```python
st = read_stream("/data/wave.mseed")
st = detrend_stream(st, type="demean")     # 去均值
st = detrend_stream(st, type="linear")    # 去线性趋势
```

---

### `taper_stream(st, max_percentage=0.05, type="cosine")`

对波形两端施加余弦锥形窗，减少频谱泄漏。

**参数：**
- `st` : obspy.Stream
- `max_percentage` : float — taper 占总长度的比例，默认 0.05（5%）
- `type` : str — 窗类型，`"cosine"` / `"hann"` / `"hamming"` 等

```python
st = taper_stream(st, max_percentage=0.05)
```

---

### `filter_stream(st, filter_type, freqmin=None, freqmax=None, corners=4, zerophase=True)`

对 Stream 进行频率域滤波。

**参数：**
- `st` : obspy.Stream
- `filter_type` : str — `"bandpass"` / `"lowpass"` / `"highpass"` / `"bandstop"`
- `freqmin` : float — 低截频（Hz），bandpass/highpass 必填
- `freqmax` : float — 高截频（Hz），bandpass/lowpass 必填
- `corners` : int — 滤波器阶数，默认 4
- `zerophase` : bool — 零相位滤波（前向+反向），默认 True

**返回：** obspy.Stream

```python
st = read_stream("/data/wave.mseed")
st = detrend_stream(st)
st = taper_stream(st)

# 带通滤波 1~10 Hz
st_bp = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

# 低通滤波 5 Hz
st_lp = filter_stream(st, "lowpass", freqmax=5.0)

# 高通滤波 0.5 Hz
st_hp = filter_stream(st, "highpass", freqmin=0.5)
```

---

### `resample_stream(st, sampling_rate)`

重采样到指定采样率。

**参数：**
- `st` : obspy.Stream
- `sampling_rate` : float — 目标采样率（Hz）

**返回：** obspy.Stream

```python
# 降采样到 50 Hz
st = resample_stream(st, sampling_rate=50.0)
```

---

### `trim_stream(st, starttime=None, endtime=None, pad=True)`

截取指定时间窗口。

**参数：**
- `st` : obspy.Stream
- `starttime` : obspy.UTCDateTime 或 str，如 `"2024-01-01T00:00:00"`
- `endtime` : obspy.UTCDateTime 或 str
- `pad` : bool — 不足时是否补零，默认 True

```python
from obspy import UTCDateTime
t0 = UTCDateTime("2024-03-15T08:30:00")
st = trim_stream(st, starttime=t0, endtime=t0 + 60)  # 截取 60 秒
```

---

### `merge_stream(st)`

合并同一通道的多段 Trace（填充 gap）。

**返回：** obspy.Stream

```python
st = merge_stream(st)
```

---

### `remove_response(st, inventory_or_paz, output="VEL", pre_filt=None)`

去除仪器响应，转换为物理量（位移 / 速度 / 加速度）。

**参数：**
- `st` : obspy.Stream
- `inventory_or_paz` : obspy.Inventory 或 PAZ dict
- `output` : str — `"DISP"`（位移）/ `"VEL"`（速度）/ `"ACC"`（加速度）
- `pre_filt` : tuple — 水波整形前置滤波器，如 `(0.005, 0.01, 45, 50)`

```python
from obspy import read_inventory
inv = read_inventory("station.xml")
st = remove_response(st, inv, output="VEL", pre_filt=(0.005, 0.01, 45, 50))
```

---

## 标准预处理流程

```python
st = read_stream("/data/wave.mseed")

# 标准四步预处理
st = detrend_stream(st, type="demean")
st = detrend_stream(st, type="linear")
st = taper_stream(st, max_percentage=0.05)
st_filtered = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

print("预处理完成")
stream_info(st_filtered)
```

---

## 注意事项

- 滤波前务必先去趋势和 taper，否则边缘效应会污染频谱
- `zerophase=True` 不引入相位偏移，地震学中强烈推荐
- 重采样前建议先做低通滤波（截频 < 新采样率/2），避免混叠
