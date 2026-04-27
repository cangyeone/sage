"""
toolkit.py — 内置地震学处理工具包

此模块在代码执行环境中通过 ``from seismo_code.toolkit import *`` 自动注入，
LLM 生成的代码可以直接调用这里的所有函数，而不需要重新实现基础功能。

函数分组
--------
1. 数据 I/O          read_stream, read_stream_from_dir
2. 波形处理          detrend_stream, taper_stream, filter_stream, resample_stream,
                     trim_stream, merge_stream
3. 仪器响应          remove_response
4. 可视化            plot_stream, plot_spectrogram, plot_psd, plot_particle_motion
5. 走时与到时        taup_arrivals, p_travel_time, s_travel_time
6. 频谱分析          compute_spectrum, compute_hvsr
7. 震源参数          estimate_magnitude_ml, estimate_corner_freq,
                     estimate_seismic_moment, estimate_stress_drop
8. 实用工具          stream_info, picks_to_dict
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ObsPy lazy import helpers
# ---------------------------------------------------------------------------

def _obspy():
    import obspy
    return obspy

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ---------------------------------------------------------------------------
# 1. 数据 I/O
# ---------------------------------------------------------------------------

def read_stream(path: str):
    """
    读取地震数据文件（MiniSEED / SAC / SEG2 等 ObsPy 支持的格式）。

    Parameters
    ----------
    path : str
        文件路径或目录路径（目录时读取所有支持格式文件）。

    Returns
    -------
    obspy.Stream
    """
    obspy = _obspy()
    p = Path(path)
    if p.is_dir():
        return read_stream_from_dir(str(p))
    return obspy.read(str(p))


def read_stream_from_dir(directory: str, pattern: str = "**/*") -> "obspy.Stream":
    """
    递归读取目录中所有地震数据文件，合并为一个 Stream。

    Parameters
    ----------
    directory : str
        目标目录。
    pattern : str
        glob 模式，默认递归匹配所有文件。

    Returns
    -------
    obspy.Stream
    """
    obspy = _obspy()
    EXTS = {".mseed", ".miniseed", ".seed", ".sac", ".SAC", ".MSEED"}
    st = obspy.Stream()
    for fpath in sorted(Path(directory).glob(pattern)):
        if fpath.suffix.lower() in {e.lower() for e in EXTS} or fpath.suffix == "":
            try:
                st += obspy.read(str(fpath))
            except Exception:
                pass
    st.merge(method=1, fill_value="interpolate")
    return st


# ---------------------------------------------------------------------------
# 2. 波形处理
# ---------------------------------------------------------------------------

def detrend_stream(st, type: str = "demean") -> "obspy.Stream":
    """去趋势（demean / linear / constant）"""
    st = st.copy()
    st.detrend(type)
    return st


def taper_stream(st, max_percentage: float = 0.05, type: str = "cosine") -> "obspy.Stream":
    """对 Stream 施加余弦/汉宁窗等尖灭处理。"""
    st = st.copy()
    st.taper(max_percentage=max_percentage, type=type)
    return st


def filter_stream(
    st,
    filter_type: str = "bandpass",
    freqmin: Optional[float] = 1.0,
    freqmax: Optional[float] = 10.0,
    corners: int = 4,
    zerophase: bool = True,
) -> "obspy.Stream":
    """
    对 Stream 应用滤波器。

    Parameters
    ----------
    st : obspy.Stream
    filter_type : {'bandpass', 'lowpass', 'highpass', 'bandstop'}
    freqmin : float
        低截频（bandpass / bandstop / highpass 使用）。
    freqmax : float
        高截频（bandpass / bandstop / lowpass 使用）。
    corners : int
        滤波器阶数，默认 4。
    zerophase : bool
        是否使用零相位滤波，默认 True。

    Returns
    -------
    obspy.Stream
    """
    st = st.copy()
    kwargs: Dict = dict(corners=corners, zerophase=zerophase)
    ft = filter_type.lower()
    if ft == "bandpass":
        st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, **kwargs)
    elif ft == "lowpass":
        st.filter("lowpass", freq=freqmax or freqmin, **kwargs)
    elif ft == "highpass":
        st.filter("highpass", freq=freqmin, **kwargs)
    elif ft == "bandstop":
        st.filter("bandstop", freqmin=freqmin, freqmax=freqmax, **kwargs)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    return st


def resample_stream(st, sampling_rate: float) -> "obspy.Stream":
    """重采样至目标采样率（Hz）。"""
    st = st.copy()
    st.resample(sampling_rate)
    return st


def trim_stream(st, starttime=None, endtime=None, pad: bool = True) -> "obspy.Stream":
    """
    裁剪 Stream 至 [starttime, endtime]。

    Parameters
    ----------
    starttime / endtime : str or obspy.UTCDateTime
        时间字符串（'2021-05-21T09:04:00'）或 UTCDateTime。
    pad : bool
        缺失部分用零填充，默认 True。
    """
    obspy = _obspy()
    st = st.copy()
    t0 = obspy.UTCDateTime(starttime) if isinstance(starttime, str) else starttime
    t1 = obspy.UTCDateTime(endtime) if isinstance(endtime, str) else endtime
    st.trim(starttime=t0, endtime=t1, pad=pad, fill_value=0)
    return st


def merge_stream(st) -> "obspy.Stream":
    """合并同台站同通道的多段数据。"""
    st = st.copy()
    st.merge(method=1, fill_value="interpolate")
    return st


# ---------------------------------------------------------------------------
# 3. 仪器响应去除
# ---------------------------------------------------------------------------

def remove_response(
    st,
    inventory_path: str,
    output: str = "VEL",
    pre_filt: Optional[Tuple] = (0.5, 1.0, 45.0, 50.0),
) -> "obspy.Stream":
    """
    去除仪器响应，将原始计数转为物理量。

    Parameters
    ----------
    st : obspy.Stream
    inventory_path : str
        StationXML 或 RESP 文件路径。
    output : {'DISP', 'VEL', 'ACC'}
        输出物理量：位移 / 速度 / 加速度。
    pre_filt : tuple of 4 floats, optional
        零相位余弦滤波频率（f1, f2, f3, f4）Hz。

    Returns
    -------
    obspy.Stream（单位：m, m/s, 或 m/s²）
    """
    obspy = _obspy()
    inv = obspy.read_inventory(inventory_path)
    st = st.copy()
    st.remove_response(inventory=inv, output=output, pre_filt=pre_filt)
    return st


# ---------------------------------------------------------------------------
# 4. 可视化
# ---------------------------------------------------------------------------

def plot_stream(
    st,
    title: str = "Waveform",
    outfile: Optional[str] = None,
    picks: Optional[List[Dict]] = None,
    normalize: bool = True,
    figsize: Tuple = (12, None),
) -> str:
    """
    绘制 Stream 中各道的波形图。

    Parameters
    ----------
    st : obspy.Stream
    title : str
        图标题。
    outfile : str, optional
        输出文件路径（PNG）。None 时自动生成。
    picks : list of dict, optional
        震相标注列表，每个 dict 含 {'time': UTCDateTime, 'phase': str, 'station': str}。
    normalize : bool
        各道独立归一化，默认 True。
    figsize : tuple
        图像尺寸（宽, 高），高为 None 时自动按道数计算。

    Returns
    -------
    str
        保存的图像路径。
    """
    plt = _plt()
    obspy = _obspy()

    n = len(st)
    if n == 0:
        raise ValueError("Stream is empty.")

    h = figsize[1] or max(3, n * 2.2)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], h), sharex=False)
    if n == 1:
        axes = [axes]

    for i, tr in enumerate(st):
        data = tr.data.astype(float)
        times = tr.times()
        if normalize and np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        axes[i].plot(times, data, "k-", lw=0.8)

        # Draw picks
        if picks:
            for pk in picks:
                sta = pk.get("station", "")
                if sta and sta.split(".")[-1] not in (tr.stats.station, sta):
                    continue
                pt = pk.get("time")
                if pt is None:
                    continue
                if isinstance(pt, str):
                    pt = obspy.UTCDateTime(pt)
                rel = float(pt - tr.stats.starttime)
                if 0 <= rel <= times[-1]:
                    color = "r" if pk.get("phase", "").startswith("P") else "b"
                    axes[i].axvline(rel, color=color, lw=1.2, ls="--", alpha=0.8)
                    axes[i].text(rel, 0.9, pk.get("phase", ""),
                                 transform=axes[i].get_xaxis_transform(),
                                 color=color, fontsize=8, ha="left")

        label = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
        axes[i].set_ylabel(label, fontsize=8, rotation=0, labelpad=60, va="center")
        axes[i].set_xlim(times[0], times[-1])
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel("Time (s)" if i == n - 1 else "")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "waveform.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return outfile


def plot_spectrogram(
    tr,
    title: str = "Spectrogram",
    outfile: Optional[str] = None,
    wlen: float = 1.0,
    per_lap: float = 0.9,
    dbscale: bool = True,
) -> str:
    """
    绘制单道波形的时频谱图（短时傅里叶变换）。

    Parameters
    ----------
    tr : obspy.Trace
    title : str
    outfile : str, optional
    wlen : float
        STFT 窗口长度（秒），默认 1.0。
    per_lap : float
        窗口重叠比例 0–1，默认 0.9。
    dbscale : bool
        使用 dB 色标，默认 True。

    Returns
    -------
    str  图像路径
    """
    from obspy.imaging.spectrogram import spectrogram as _sg
    plt = _plt()

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "spectrogram.png")

    fig = plt.figure(figsize=(10, 5))
    _sg(tr.data, tr.stats.sampling_rate, wlen=wlen, per_lap=per_lap,
        dbscale=dbscale, show=False, axes=fig.add_subplot(111))
    plt.title(f"{title}  —  {tr.id}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return outfile


def plot_psd(
    tr,
    title: str = "Power Spectral Density",
    outfile: Optional[str] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    计算并绘制功率谱密度（Welch 法）。

    Parameters
    ----------
    tr : obspy.Trace
    title : str
    outfile : str, optional
    window : str
        Welch 窗口类型（'hann', 'boxcar' 等），默认 'hann'。

    Returns
    -------
    freqs : ndarray
    psd : ndarray
    outfile : str
    """
    from scipy.signal import welch as _welch
    plt = _plt()

    sr = tr.stats.sampling_rate
    freqs, psd = _welch(tr.data.astype(float), fs=sr, window=window, nperseg=min(len(tr.data), int(sr * 10)))

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "psd.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    psd_db = 10 * np.log10(psd + 1e-30)
    ax.semilogx(freqs[1:], psd_db[1:], "b-", lw=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB rel. 1 (unit)²/Hz)")
    ax.set_title(f"{title}  —  {tr.id}")
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return freqs, psd, outfile


def plot_particle_motion(
    st,
    component_pairs: Optional[List[Tuple[str, str]]] = None,
    outfile: Optional[str] = None,
) -> str:
    """
    绘制质点运动轨迹图（Hodogram）。

    Parameters
    ----------
    st : obspy.Stream
        应含 E/N/Z 三分量。
    component_pairs : list of (str, str), optional
        要绘制的分量对，如 [('E','N'), ('Z','N')]。默认自动检测。
    outfile : str, optional

    Returns
    -------
    str  图像路径
    """
    plt = _plt()

    def _get_comp(comp: str):
        for tr in st:
            if tr.stats.channel[-1].upper() == comp.upper():
                return tr.data.astype(float)
        return None

    if component_pairs is None:
        component_pairs = [("E", "N"), ("Z", "N")]

    fig, axes = plt.subplots(1, len(component_pairs), figsize=(5 * len(component_pairs), 4))
    if len(component_pairs) == 1:
        axes = [axes]

    for ax, (c1, c2) in zip(axes, component_pairs):
        d1 = _get_comp(c1)
        d2 = _get_comp(c2)
        if d1 is None or d2 is None:
            ax.set_title(f"Missing {c1} or {c2}")
            continue
        n = min(len(d1), len(d2))
        sc = ax.scatter(d1[:n], d2[:n], c=np.arange(n), cmap="viridis", s=1, alpha=0.7)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_title(f"Particle motion ({c1}-{c2})")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Sample index")

    fig.suptitle("Particle Motion / Hodogram", fontsize=12)
    plt.tight_layout()

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "particle_motion.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return outfile


# ---------------------------------------------------------------------------
# 5. 走时与到时计算
# ---------------------------------------------------------------------------

def taup_arrivals(
    dist_deg: float,
    depth_km: float = 10.0,
    model: str = "iasp91",
    phases: Optional[List[str]] = None,
) -> List[Dict]:
    """
    用 TauPy 计算理论走时。

    Parameters
    ----------
    dist_deg : float
        震中距（度）。
    depth_km : float
        震源深度（km）。
    model : str
        速度模型名称（'iasp91', 'ak135', 'prem'）。
    phases : list of str, optional
        需要的震相列表，如 ['P', 'S', 'pP']。None 表示全部。

    Returns
    -------
    list of dict
        每个震相: {'phase': str, 'time_s': float, 'ray_param': float}
    """
    from obspy.taup import TauPyModel
    m = TauPyModel(model=model)
    arrivals = m.get_travel_times(
        source_depth_in_km=depth_km,
        distance_in_degree=dist_deg,
        phase_list=phases or ["P", "S", "pP", "sP", "PP", "SS", "Pn", "Sn"],
    )
    return [
        {"phase": a.name, "time_s": round(a.time, 3), "ray_param": round(a.ray_param_sec_degree, 4)}
        for a in arrivals
    ]


def p_travel_time(dist_km: float, depth_km: float = 10.0, model: str = "iasp91") -> float:
    """计算 P 波走时（秒），震中距以 km 给出。"""
    from obspy.geodetics import kilometer2degrees
    dist_deg = kilometer2degrees(dist_km)
    arr = taup_arrivals(dist_deg, depth_km, model, phases=["P", "Pn", "Pg"])
    return arr[0]["time_s"] if arr else float("nan")


def s_travel_time(dist_km: float, depth_km: float = 10.0, model: str = "iasp91") -> float:
    """计算 S 波走时（秒），震中距以 km 给出。"""
    from obspy.geodetics import kilometer2degrees
    dist_deg = kilometer2degrees(dist_km)
    arr = taup_arrivals(dist_deg, depth_km, model, phases=["S", "Sn", "Sg"])
    return arr[0]["time_s"] if arr else float("nan")


def plot_travel_time_curve(
    dist_range: Tuple[float, float] = (0, 30),
    depth_km: float = 10.0,
    model: str = "iasp91",
    phases: Optional[List[str]] = None,
    outfile: Optional[str] = None,
) -> str:
    """
    绘制走时曲线（T-Δ 图）。

    Parameters
    ----------
    dist_range : (min_deg, max_deg)
    depth_km : float
    model : str
    phases : list of str, optional
    outfile : str, optional

    Returns
    -------
    str  图像路径
    """
    from obspy.taup import TauPyModel
    plt = _plt()
    m = TauPyModel(model=model)
    dists = np.linspace(dist_range[0], dist_range[1], 60)
    phase_times: Dict[str, List] = {}
    phase_dists: Dict[str, List] = {}
    ph_list = phases or ["P", "S", "Pn", "Sn", "Pg", "Sg"]

    for d in dists:
        try:
            arrs = m.get_travel_times(depth_km, d, phase_list=ph_list)
        except Exception:
            continue
        for a in arrs:
            phase_times.setdefault(a.name, []).append(a.time)
            phase_dists.setdefault(a.name, []).append(d)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(phase_times)))
    for (ph, ts), col in zip(phase_times.items(), colors):
        ds = phase_dists[ph]
        order = np.argsort(ds)
        ax.plot(np.array(ds)[order], np.array(ts)[order], ".", ms=3, label=ph, color=col)

    ax.set_xlabel("Distance (°)")
    ax.set_ylabel("Travel time (s)")
    ax.set_title(f"Travel-time curves — {model}  depth={depth_km} km")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "travel_time_curve.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return outfile


# ---------------------------------------------------------------------------
# 6. 频谱分析
# ---------------------------------------------------------------------------

def compute_spectrum(
    tr,
    method: str = "fft",
    window: str = "hann",
    return_velocity: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算地震道的振幅谱。

    Parameters
    ----------
    tr : obspy.Trace
    method : {'fft', 'welch'}
    window : str
    return_velocity : bool
        True 时返回速度谱（频域积分），False 返回位移谱（频域积分两次）。

    Returns
    -------
    freqs : ndarray
    amplitudes : ndarray
    """
    data = tr.data.astype(float)
    sr = tr.stats.sampling_rate
    n = len(data)

    if method == "welch":
        from scipy.signal import welch
        freqs, psd = welch(data, fs=sr, window=window, nperseg=min(n, int(sr * 10)))
        return freqs, np.sqrt(psd)

    # FFT
    from scipy.signal.windows import get_window
    win = get_window(window, n) if window != "boxcar" else np.ones(n)
    spectrum = np.fft.rfft(data * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    amplitudes = np.abs(spectrum) / (n * sr)
    return freqs, amplitudes


def compute_hvsr(
    st,
    freqmin: float = 0.1,
    freqmax: float = 20.0,
    outfile: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    计算水平垂直谱比（HVSR / Nakamura 法）。

    Parameters
    ----------
    st : obspy.Stream  （需含 E/N/Z 三分量）
    freqmin / freqmax : float  分析频率范围
    outfile : str, optional

    Returns
    -------
    freqs : ndarray
    hvsr : ndarray
    outfile : str
    """
    plt = _plt()

    def _get(c):
        for tr in st:
            if tr.stats.channel[-1].upper() == c:
                return tr
        return None

    trZ = _get("Z")
    trN = _get("N") or _get("1")
    trE = _get("E") or _get("2")
    if trZ is None or trN is None or trE is None:
        raise ValueError("HVSR requires E, N, Z components.")

    sr = trZ.stats.sampling_rate
    freqs_z, amp_z = compute_spectrum(trZ)
    freqs_n, amp_n = compute_spectrum(trN)
    freqs_e, amp_e = compute_spectrum(trE)

    n = min(len(freqs_z), len(freqs_n), len(freqs_e))
    freqs = freqs_z[:n]
    amp_h = np.sqrt((amp_n[:n] ** 2 + amp_e[:n] ** 2) / 2)
    hvsr = np.where(amp_z[:n] > 0, amp_h / amp_z[:n], 0)

    # Frequency mask
    mask = (freqs >= freqmin) & (freqs <= freqmax)

    if outfile is None:
        import tempfile
        outdir = os.environ.get("SAGE_OUTDIR", tempfile.gettempdir())
        outfile = os.path.join(outdir, "hvsr.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freqs[mask], hvsr[mask], "b-", lw=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Ratio")
    ax.set_title("HVSR (Horizontal-to-Vertical Spectral Ratio)")
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] {outfile}")
    return freqs[mask], hvsr[mask], outfile


# ---------------------------------------------------------------------------
# 7. 震源参数
# ---------------------------------------------------------------------------

def estimate_magnitude_ml(
    tr,
    dist_km: float,
    station: str = "",
    wood_anderson_gain: float = 2080.0,
) -> float:
    """
    估算 Richter 近震震级 ML（仿 Wood-Anderson 地震仪）。

    Parameters
    ----------
    tr : obspy.Trace
        垂直或水平分量速度道（单位 m/s）。
    dist_km : float
        震中距（km）。
    station : str  （仅用于日志）
    wood_anderson_gain : float
        Wood-Anderson 放大系数，默认 2080。

    Returns
    -------
    float  ML 震级
    """
    # 模拟 Wood-Anderson 响应（简单高通+低通近似）
    from scipy.signal import lfilter
    sr = tr.stats.sampling_rate
    data = tr.data.astype(float) * wood_anderson_gain

    # Peak-to-peak amplitude in mm
    amp_mm = np.max(np.abs(data)) * 1000.0  # m → mm

    # Richter (1935) station correction table (approximate polynomial)
    log_dist = np.log10(max(dist_km, 1.0))
    station_corr = 0.0  # assume calibrated station
    ml = np.log10(max(amp_mm, 1e-10)) + 3 * log_dist - 2.92 + station_corr
    return round(float(ml), 2)


def estimate_corner_freq(
    tr,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    freqmin: float = 1.0,
    freqmax: float = 30.0,
) -> float:
    """
    用 Brune (1970) 震源谱模型估算拐角频率 fc。

    P 波窗口内的振幅谱用对数最小二乘法拟合 ω² 模型：
        A(f) = Ω₀ / (1 + (f/fc)²)

    Parameters
    ----------
    tr : obspy.Trace
    t_start / t_end : float, optional
        相对于道起始时刻的时间窗（秒）。
    freqmin / freqmax : float
        拟合的频率范围（Hz）。

    Returns
    -------
    float  拐角频率 fc（Hz）
    """
    from scipy.optimize import curve_fit

    sr = tr.stats.sampling_rate
    data = tr.data.astype(float)

    # Extract window
    i0 = int(t_start * sr) if t_start else 0
    i1 = int(t_end * sr) if t_end else len(data)
    data = data[i0:i1]

    n = len(data)
    if n < 16:
        return float("nan")

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec = np.abs(np.fft.rfft(data * np.hanning(n))) / n

    mask = (freqs >= freqmin) & (freqs <= freqmax)
    if mask.sum() < 5:
        return float("nan")

    f = freqs[mask]
    s = spec[mask] + 1e-30

    def brune(freq, omega0, fc):
        return omega0 / (1.0 + (freq / fc) ** 2)

    try:
        p0 = [s.max(), 5.0]
        popt, _ = curve_fit(brune, f, s, p0=p0,
                            bounds=([0, freqmin], [np.inf, freqmax]),
                            maxfev=5000)
        return round(float(popt[1]), 3)
    except Exception:
        return float("nan")


def estimate_seismic_moment(
    tr,
    dist_km: float,
    density: float = 2700.0,
    velocity: float = 3500.0,
    radiation: float = 0.63,
    free_surface: float = 2.0,
) -> float:
    """
    估算地震矩 M₀（N·m）。

    使用远场位移谱的低频平台值（Ω₀）：
        M₀ = 4π ρ v³ R Ω₀ / (radiation × free_surface)

    Parameters
    ----------
    tr : obspy.Trace  （位移道，单位 m）
    dist_km : float   震中距（km）
    density : float   密度（kg/m³），默认 2700
    velocity : float  P 或 S 波速度（m/s），默认 3500
    radiation : float 辐射花样系数，P 波取 0.52，S 波取 0.63
    free_surface : float 自由面修正，默认 2.0

    Returns
    -------
    float  M₀（N·m）
    """
    freqs, spec = compute_spectrum(tr)
    # Low-frequency plateau: median of spectrum below 2 Hz
    mask_low = (freqs > 0.1) & (freqs < 2.0)
    omega0 = float(np.median(spec[mask_low])) if mask_low.sum() > 0 else float(spec[1])

    R = dist_km * 1000.0  # m
    M0 = 4 * np.pi * density * velocity ** 3 * R * omega0 / (radiation * free_surface)
    return float(M0)


def moment_to_mw(M0: float) -> float:
    """将地震矩 M₀（N·m）转换为矩震级 Mw（Hanks & Kanamori 1979）。"""
    return (2.0 / 3.0) * np.log10(M0) - 6.07


def estimate_stress_drop(
    M0: float,
    fc: float,
    velocity: float = 3500.0,
    kappa: float = 0.37,
) -> float:
    """
    用 Brune (1970) 公式估算应力降（Pa）。

        Δσ = (7/16) * M₀ * (fc / kappa / velocity)³

    Parameters
    ----------
    M0 : float   地震矩（N·m）
    fc : float   拐角频率（Hz）
    velocity : float  剪切波速（m/s）
    kappa : float   Brune 模型常数，默认 0.37

    Returns
    -------
    float  应力降（Pa）
    """
    r = kappa * velocity / fc  # 震源半径（m）
    delta_sigma = (7.0 / 16.0) * M0 / r ** 3
    return float(delta_sigma)


# ---------------------------------------------------------------------------
# 8. 实用工具
# ---------------------------------------------------------------------------

def stream_info(st) -> str:
    """返回并打印 Stream 的详细信息字符串（采样率、时间范围、通道数等）。"""
    lines = [f"Stream: {len(st)} trace(s)"]
    for tr in st:
        s = tr.stats
        lines.append(
            f"  {tr.id}  |  {s.sampling_rate} Hz  |  "
            f"{s.starttime.strftime('%Y-%m-%dT%H:%M:%S')}  →  "
            f"{s.endtime.strftime('%Y-%m-%dT%H:%M:%S')}  |  "
            f"{len(tr.data)} samples"
        )
    result = "\n".join(lines)
    print(result)
    return result


def picks_to_dict(picks_file: str) -> List[Dict]:
    """
    读取 SAGE 震相拾取 .txt 文件，返回 list of dict。

    Fields: phase, rel_time, confidence, abs_time, snr, amplitude, station, polarity
    """
    import csv
    from datetime import datetime as _dt
    results = []
    with open(picks_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                results.append({
                    "phase": parts[0],
                    "rel_time": float(parts[1]),
                    "confidence": float(parts[2]),
                    "abs_time": _dt.strptime(parts[3], "%Y-%m-%d %H:%M:%S.%f"),
                    "snr": float(parts[4]),
                    "amplitude": float(parts[5]),
                    "station": parts[6],
                    "polarity": parts[7] if len(parts) > 7 else "N",
                })
            except (ValueError, IndexError):
                continue
    return results


# ---------------------------------------------------------------------------
# 9. GMT 绘图
# ---------------------------------------------------------------------------

def _extract_cjk_texts(script: str):
    """
    从 GMT 脚本中提取所有包含 CJK 字符的标题/标签文本，
    返回 (cleaned_script, list_of_(pattern, text)) 供后处理使用。

    处理以下模式：
      -BWSne+t"中文标题"     → 标题
      -B...+l"中文标签"      → 轴标签
      -t"中文"               → 整体标题
    """
    import re as _re

    CJK = r'[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef\u3000-\u303f]'
    found: list = []   # [(placeholder_key, original_text, annotation_type)]

    def _replace(m, ann_type):
        text = m.group(2)
        if _re.search(CJK, text):
            found.append((ann_type, text))
            return m.group(1) + '" "'   # 替换为空字符串占位
        return m.group(0)

    # +t"..." 标题
    out = _re.sub(r'(\+t")([^"]*")', lambda m: _replace(m, 'title'), script)
    # +l"..." 轴标签（x/y）
    out = _re.sub(r'(\+l")([^"]*")', lambda m: _replace(m, 'label'), out)
    # -t"..." 独立标题（某些 GMT 命令）
    out = _re.sub(r'(-t")([^"]*")',  lambda m: _replace(m, 'title'), out)

    return out, found


def _overlay_cjk_texts(img_path: str, texts: list) -> None:
    """
    用 matplotlib 将 CJK 文本叠加到 GMT 生成的 PNG 上。
    texts: list of (ann_type, text)  ann_type 为 'title'/'label'
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        titles = [t for tp, t in texts if tp == 'title']
        if not titles:
            return

        main_title = '  '.join(titles)

        img = mpimg.imread(img_path)
        h, w = img.shape[:2]
        dpi = 150
        extra_h = 52   # 顶部留给标题的像素高度

        fig = plt.figure(figsize=(w / dpi, (h + extra_h) / dpi), dpi=dpi)
        ax_img = fig.add_axes([0, 0, 1, h / (h + extra_h)])
        ax_img.imshow(img, aspect='auto')
        ax_img.axis('off')

        fig.text(
            0.5, 1 - 4 / (h + extra_h),
            main_title,
            ha='center', va='top',
            fontsize=15, fontweight='bold',
            transform=fig.transFigure,
        )

        plt.savefig(img_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', pad_inches=0.05)
        plt.close(fig)
    except Exception as e:
        print(f"警告：CJK 标题叠加失败: {e}")


def run_gmt(
    script: str,
    outname: str = "gmt_map",
    title: str = "GMT Map",
) -> str:
    """
    执行 GMT bash 脚本，返回输出 PNG 路径（兼容旧接口）。

    Thin wrapper around ``execute_bash()`` — kept for mixed Python+GMT scripts
    that need to pass data computed in Python into the bash script.

    For pure GMT tasks, prefer outputting a ``bash`` code block directly so the
    code engine runs it via ``execute_bash()`` without going through Python at all.

    CJK 中文标题自动处理：GMT PostScript 引擎不支持 CJK，
    此函数先用占位符替换中文文本，执行后再通过 matplotlib 叠加回 PNG。

    Parameters
    ----------
    script : str
        完整 GMT bash 脚本（GMT6 modern mode）。
        ``gmt begin`` 后的文件名会被自动替换为 ``outname``。
    outname : str
        输出文件基名（不含扩展名），默认 ``"gmt_map"``。
    title : str
        脚本注释标题，不影响图像内容。

    Returns
    -------
    str  生成的 PNG 绝对路径。
    """
    import re as _re, shutil as _shutil

    # ── 前置检查 ──────────────────────────────────────────────────────────
    if _shutil.which('gmt') is None:
        raise RuntimeError(
            "GMT 未安装或未在 PATH 中。请先安装 GMT >= 6.0：\n"
            "  macOS : brew install gmt\n"
            "  Linux : conda install -c conda-forge gmt\n"
            "          或 sudo apt install gmt"
        )

    # Strip .png / .PNG suffix from outname so we don't end up with double extensions
    # (LLM sometimes calls run_gmt(..., outname="terrain_map.png"))
    _outname_base = outname
    for _sfx in (".png", ".PNG", ".Png"):
        if _outname_base.endswith(_sfx):
            _outname_base = _outname_base[: -len(_sfx)]
            break
    outname = _outname_base

    outdir  = os.environ.get("SAGE_OUTDIR", os.getcwd())
    out_png = os.path.join(outdir, outname + ".png")
    out_sh  = os.path.join(outdir, outname + ".sh")

    # ── 锁定 gmt begin 的输出名和格式 ────────────────────────────────────
    if _re.search(r'gmt\s+begin', script, _re.IGNORECASE):
        script = _re.sub(
            r'(gmt\s+begin)\s+\S+(\s+\S+)?',
            r'\1 ' + outname + ' PNG',
            script, flags=_re.IGNORECASE,
        )

    # ── 提取 CJK 文本，替换为占位符（GMT PS 引擎不支持 CJK）────────────
    script_exec, cjk_texts = _extract_cjk_texts(script)

    # ── 保存原始脚本（含中文）供下载 ─────────────────────────────────────
    with open(out_sh, 'w', encoding='utf-8') as f:
        f.write(
            f"#!/bin/bash\n# GMT script — {title}\n# Generated by SAGE SeismicX\n\n"
            f"cd '{outdir}'\n\n" + script
        )
    os.chmod(out_sh, 0o755)

    # ── 执行（通过 safe_executor.execute_bash）────────────────────────────
    try:
        from seismo_code.safe_executor import execute_bash as _exec_bash
    except ImportError:
        from safe_executor import execute_bash as _exec_bash   # type: ignore

    full_script = f"cd '{outdir}'\n\n" + script_exec
    res = _exec_bash(full_script, project_root=str(Path(__file__).parent.parent),
                     timeout=180, keep_dir=True)

    if res.stdout:
        print(res.stdout, end='')

    if not res.success:
        # GMT sometimes returns non-zero due to warnings yet still creates the PNG.
        # Promote to error only if the PNG is also absent (checked below).
        _gmt_hard_fail = not os.path.exists(out_png)

    # GMT5 legacy：ps → png
    if not os.path.exists(out_png):
        import subprocess as _sp
        ps_file = os.path.join(outdir, outname + ".ps")
        if os.path.exists(ps_file):
            _sp.run(['gmt', 'psconvert', ps_file, '-Tg', '-A', '-P'],
                    cwd=outdir, capture_output=True)

    # Some GMT versions write terrain_map.PNG (uppercase) instead of .png
    if not os.path.exists(out_png):
        _upper = os.path.join(outdir, outname + ".PNG")
        if os.path.exists(_upper):
            import shutil as _sh2
            _sh2.copy2(_upper, out_png)

    # Last resort: search the bash tmpdir (exec_dir) for the PNG
    if not os.path.exists(out_png) and hasattr(res, 'exec_dir') and res.exec_dir:
        import glob as _glob
        _candidates = (
            _glob.glob(os.path.join(res.exec_dir, outname + ".png")) +
            _glob.glob(os.path.join(res.exec_dir, outname + ".PNG")) +
            _glob.glob(os.path.join(res.exec_dir, "*.png")) +
            _glob.glob(os.path.join(res.exec_dir, "*.PNG"))
        )
        if _candidates:
            import shutil as _sh3
            _sh3.copy2(_candidates[0], out_png)

    if not os.path.exists(out_png):
        # Now raise — combine success flag and absence of PNG for clear diagnosis
        _err_ctx = (res.stderr or res.error or '')[:600]
        if not res.success:
            raise RuntimeError(
                f"GMT 执行失败（返回码非零）:\n{_err_ctx}"
            )
        raise RuntimeError(
            f"GMT 执行成功但未找到输出图像 {out_png}\n"
            f"stderr: {_err_ctx}"
        )

    # If we reach here the PNG exists; still surface non-zero exit as a warning
    if not res.success:
        _warn_msg = (res.stderr or res.error or '').strip()
        if _warn_msg:
            print(f"[GMT WARNING] {_warn_msg[:300]}")

    # ── 叠加中文标题 ──────────────────────────────────────────────────────
    if cjk_texts:
        _overlay_cjk_texts(out_png, cjk_texts)

    print(f"[FIGURE] {out_png}")
    print(f"[GMT_SCRIPT] {out_sh}")
    return out_png
