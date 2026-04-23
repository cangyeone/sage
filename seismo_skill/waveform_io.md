---
name: waveform_io
category: waveform
keywords: 读取波形, 加载波形, mseed, sac, miniseed, seed, 波形文件, read_stream, stream_info, picks_to_dict
---

# 波形读取与信息查询

## 描述

读取本地波形文件（SAC / MiniSEED / SEED 等），获取流信息，加载震相拾取结果。

---

## 主要函数

### `read_stream(path)`

读取单个波形文件，返回 `obspy.Stream`。

**参数：**
- `path` : str — 波形文件路径（支持 .mseed / .sac / .seed / .msd 等）

**返回：** `obspy.Stream`

```python
st = read_stream("/data/station/HHZ.mseed")
print(st)
# 输出: 1 Trace(s) in Stream: NET.STA.LOC.HHZ ...
```

---

### `read_stream_from_dir(directory, pattern="**/*")`

递归读取目录下所有符合模式的波形文件，合并为一个 Stream。

**参数：**
- `directory` : str — 目录路径
- `pattern` : str — glob 模式，默认 `**/*`（所有文件）

**返回：** `obspy.Stream`

```python
st = read_stream_from_dir("/data/event_001/", pattern="**/*.mseed")
print(f"共加载 {len(st)} 条 Trace")
```

---

### `stream_info(st)`

打印 Stream 的详细统计信息（台站、通道、时间范围、采样率）。

**参数：**
- `st` : obspy.Stream

**返回：** str（信息字符串，同时打印）

```python
st = read_stream("/data/wave.mseed")
info = stream_info(st)
print(info)
```

---

### `picks_to_dict(picks_file)`

读取震相拾取结果文件（CSV 格式：phase, time_s, confidence, abs_time, snr, ...）。

**参数：**
- `picks_file` : str — 拾取结果文件路径

**返回：** `List[Dict]`，每条包含 `phase`、`time_s`、`confidence`、`abs_time`、`snr` 等字段

```python
picks = picks_to_dict("results/picks.csv")
for p in picks:
    print(f"{p['phase']}  t={p['time_s']:.2f}s  conf={p['confidence']:.2f}")
```

---

## 注意事项

- `read_stream` 底层调用 `obspy.read()`，支持所有 ObsPy 能读取的格式
- 若文件路径不存在会抛出 `FileNotFoundError`，建议用 `try/except` 保护
- 三分量文件（HHZ/HHN/HHE）建议放在同一目录并用 `read_stream_from_dir` 一次性加载
