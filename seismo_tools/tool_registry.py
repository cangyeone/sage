"""
tool_registry.py — 外部地震学工具注册表

内置知识库
----------
包含以下常用工具的完整接口模板，无需用户提供文档即可直接调用：
  HypoDD      双差相对重定位
  VELEST      1D 速度结构反演 + 震源定位
  NonLinLoc   非线性地震定位（概率密度）
  HYPOINVERSE USGS 标准地震定位
  focmec      震源机制解（P 波初动）

用户注册的工具
--------------
通过 DocParser.parse_text / parse_file 解析的自定义工具也存储在此注册表中。
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

# Registry storage directory
_REGISTRY_DIR = Path(__file__).parent / "registry"


# ---------------------------------------------------------------------------
# Built-in tool knowledge base
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: Dict[str, Dict] = {

    # ------------------------------------------------------------------
    # HypoDD — 双差相对重定位
    # ------------------------------------------------------------------
    "hypodd": {
        "name": "HypoDD",
        "executable": "hypoDD",
        "description": (
            "双差地震定位方法（Zhang & Fréchet 2003）。利用两个相邻地震在同一台站"
            "的震相到时差（双差），通过迭代最小二乘法精确求解相对震源位置。"
            "适用于余震序列、断层带精细定位。"
        ),
        "input_files": ["hypoDD.inp", "dt.ct", "dt.cc", "event.dat", "station.dat"],
        "input_format": textwrap.dedent("""\
            hypoDD.inp — 主控制文件（ASCII），关键行说明：
              行1: dt.cc dt.ct              ! 互相关和目录到时差文件
              行2: event.dat               ! 事件列表文件
              行3: station.dat             ! 台站文件（NET STA LOC LAT LON ELE）
              行4: hypoDD.loc hypoDD.reloc  ! 输出文件
              行5: 1 0 8 -999 -999 -999 -999 -999  ! 方法（1=SVD, 2=LSQR）
              ...

            dt.ct 格式（目录差分到时）:
              # EV1_ID EV2_ID  0.0
              STA  TP1  TP2  WEIGHT  PHASE
              ...

            station.dat 格式:
              NET_STA  LAT  LON  ELE(m)
        """),
        "input_template": textwrap.dedent("""\
            * Input file for hypoDD
            * Cross-correlation diff times:
            {{CC_FILE}}
            * Catalog diff times:
            {{CT_FILE}}
            * Initial locations:
            {{EVENT_FILE}}
            * Station information:
            {{STATION_FILE}}
            * Output files:
            hypoDD.loc hypoDD.reloc
            * Initial locations (SAVE):
            0
            * Min. number of pairs:
            8
            * CID_MIN RMS_MAX
            1 0.5
            * Distance/depth cutoffs:
            -999 -999 -999
            * Max number of iter:
            12
            * Convergence test:
            0.1
        """),
        "parameters": {
            "IDATA": "数据类型：1=dt.ct, 2=dt.cc, 3=both",
            "IPHA": "震相类型：1=P, 2=S, 3=P+S",
            "DIST": "台站到震中最大距离（km）",
            "OBSCC": "最小互相关对数",
            "OBSCT": "最小目录差分对数",
            "IITER": "迭代次数",
            "WRCC": "互相关数据权重",
            "WRCT": "目录数据权重",
        },
        "output_files": ["hypoDD.reloc", "hypoDD.loc", "hypoDD.sta", "hypoDD.res", "hypoDD.log"],
        "output_format": textwrap.dedent("""\
            hypoDD.reloc 格式（ASCII，空格分隔）:
              EV_ID  LAT  LON  DEPTH  X  Y  Z  EX  EY  EZ  YEAR  MON  DAY  HR  MIN  SEC  MAG  NC  RCC  RCT  CID
        """),
        "run_command": "hypoDD hypoDD.inp",
        "notes": "需要安装 hypoDD 可执行文件；建议先用 ph2dt 处理 phase.dat 生成 dt.ct",
    },

    # ------------------------------------------------------------------
    # VELEST — 1D 速度结构反演
    # ------------------------------------------------------------------
    "velest": {
        "name": "VELEST",
        "executable": "velest",
        "description": (
            "Kissling et al. (1994) 联合反演地震位置和一维 P/S 波速度结构。"
            "使用单纯形法或阻尼最小二乘反演，输出最小一维速度模型（Minimum 1D model）。"
        ),
        "input_files": ["velest.cmn", "events.cnv", "stations.sta", "vpmod.mod", "vsmod.mod"],
        "input_format": textwrap.dedent("""\
            velest.cmn — 主控制文件
            events.cnv — 地震事件文件（Nordic/CNV 格式）：
              年月日 时分秒 纬度 经度 深度 震级 ...
              STA  DIST  AIN  ARES  WTIME  PTIME  SP  IN  PH
              ...

            stations.sta — 台站坐标文件：
              STA  LAT  LON  ALT(m)  DELAY_P  DELAY_S

            vpmod.mod — P 波初始速度模型：
              DEPTH(km)  VP(km/s)
        """),
        "input_template": textwrap.dedent("""\
            velest.cmn template:
            Title
            {{TITLE}}
            VELEST control parameters
            {{PARAMS}}
        """),
        "parameters": {
            "imod": "速度模型约束：0=固定, 1=反演",
            "itmax": "最大迭代次数（通常 10-20）",
            "iabs": "绝对到时：1；差分到时：0",
            "neqs": "事件数量",
            "nshot": "爆破数量",
            "nsta": "台站数量",
            "istafix": "固定台站延迟：1；反演：0",
            "zmin": "震源最小深度（km）",
        },
        "output_files": ["velest.out", "velest.log", "final.mod", "final.sta", "final.cnv"],
        "output_format": "velest.out 包含反演后的速度结构和重定位结果；final.mod 为最终速度模型",
        "run_command": "velest velest.cmn",
        "notes": "CNV 格式对列宽要求严格，建议使用专用格式转换脚本",
    },

    # ------------------------------------------------------------------
    # NonLinLoc — 非线性地震定位
    # ------------------------------------------------------------------
    "nonlinloc": {
        "name": "NonLinLoc",
        "executable": "NLLoc",
        "description": (
            "Lomax et al. (2000) 基于概率密度函数（PDF）的非线性地震定位。"
            "使用 Oct-tree 或 Metropolis 采样在三维速度模型中搜索最优震源位置，"
            "输出完整的 PDF 以及置信椭球。"
        ),
        "input_files": ["nlloc.in", "phase.obs", "vp.LAYER.buf", "vs.LAYER.buf"],
        "input_format": textwrap.dedent("""\
            nlloc.in — 主控制文件（关键行）：
              LOCSIG  signature
              LOCFILES obs_dir/phase.obs NLLOC_OBS results/loc/event
              LOCGRID  NX NY NZ ORIGX ORIGY ORIGZ DX DY DZ PROB_DENSITY SAVE
              LOCMETH  EDT_OT_WT 9999 4 -1 -1 1.0 -1 0
              LOCPHASEID  P  P p px pX
              ...

            phase.obs (NLLOC_OBS 格式):
              STA  INST  COMP  PONSET  IPHASE  WEIGHT  DATE  HOUR_MIN  SEC  ERR  ERR_MAG  CODA_DUR  AMP  PRE_IMPORT
        """),
        "input_template": "",
        "parameters": {
            "LOCGRID": "定位网格设置（NX NY NZ 原点 间距 输出类型）",
            "LOCMETH": "定位方法（EDT, EDT_OT_WT, L1, LIN, OT_STACK）",
            "LOCQUAL2ERR": "RMS 到质量等级映射",
            "LOCPHSTAT": "震相统计计算控制",
            "LOCDELAY": "台站走时校正",
        },
        "output_files": ["event.hyp", "event.scat", "event.grid0.loc.hdr"],
        "output_format": textwrap.dedent("""\
            .hyp 文件（NLLOC_HYP 格式）：
              HYPOCENTER  x=DX  y=DY  z=DZ  OT=ORIGIN_TIME  ix=IX  iy=IY  iz=IZ
              GEOGRAPHIC  OT  LAT  LON  DEPTH
              STATISTICS  Exp... CovXX CovXY ...
              PHASE_STATISTICS ...
        """),
        "run_command": "NLLoc nlloc.in",
        "notes": "需先用 Vel2Grid 和 Grid2Time 生成走时网格文件（.buf）",
    },

    # ------------------------------------------------------------------
    # HYPOINVERSE — USGS 标准地震定位
    # ------------------------------------------------------------------
    "hypoinverse": {
        "name": "HYPOINVERSE",
        "executable": "hypoinv",
        "description": (
            "Klein (2002) USGS 标准地震定位程序。使用1D分层速度模型和台站时差，"
            "通过迭代最小二乘法求解震源位置和发震时刻。广泛应用于区域台网常规定位。"
        ),
        "input_files": ["hypoinv.inp", "sum.hyp", "arch.hyp"],
        "input_format": textwrap.dedent("""\
            HYPOINVERSE 命令文件格式：
              CRH 1  vp_model.crh    ! 速度模型
              STA station.sta        ! 台站文件
              PHS phase.pha          ! 震相文件
              SUM out.sum            ! 输出汇总
              HYP out.hyp            ! 输出详细

            phase.pha（HYPOINVERSE 格式）:
              YYYYMMDD HHMM SS.ss  LAT  LON  DEP  MAG
              SSSSII WHPHS WTRES ...
        """),
        "input_template": "",
        "parameters": {
            "RMS": "RMS 残差截止（秒）",
            "ERH": "水平误差截止（km）",
            "ERZ": "垂直误差截止（km）",
            "MIN": "最小震相数",
            "INS": "初始震源深度",
            "POS": "P 波速度比",
        },
        "output_files": ["out.sum", "out.arc", "out.hyp"],
        "output_format": "sum 文件每行一个地震；arc 文件含完整震相信息",
        "run_command": "hypoinv < hypoinv.inp",
        "notes": "台站坐标文件需与速度模型参考椭球一致",
    },

    # ------------------------------------------------------------------
    # focmec — P 波初动震源机制
    # ------------------------------------------------------------------
    "focmec": {
        "name": "FOCMEC",
        "executable": "focmec",
        "description": (
            "Snoke (2003) 基于 P 波初动极性和/或 SV/SH 振幅比的震源机制解搜索程序。"
            "穷举搜索断层面走向、倾角和滑动角参数空间，输出满足约束的所有解。"
        ),
        "input_files": ["focmec.inp", "focmec.pol", "focmec.amp"],
        "input_format": textwrap.dedent("""\
            focmec.pol — P 波初动极性文件：
              STA  AZM  TKO  POLARITY  (U=正 / D=负 / X=不确定)

            focmec.amp — 振幅比文件：
              STA  AZM  TKO  SV/P  SH/P

            focmec.inp — 控制文件：
              NCOUNT — 每次搜索的解数
              DANG   — 搜索角度步长（度）
              MAXOUT — 最大输出解数
        """),
        "input_template": "",
        "parameters": {
            "DANG": "走向/倾角/滑动角搜索步长（度），典型值 5",
            "MAXERROR": "允许的极性错误数",
            "AMPRAT": "振幅比权重",
        },
        "output_files": ["focmec.out", "focmec.ps"],
        "output_format": "focmec.out 每行一个解：STK DIP RAKE NPOLOK NBAD",
        "run_command": "focmec < focmec.inp",
        "notes": "takeoff angle 需根据速度模型和震源深度预先计算",
    },
}

# Fix typo in hypoinverse template
BUILTIN_TOOLS["hypoinverse"]["input_format"] = textwrap.dedent("""\
    HYPOINVERSE 命令文件格式：
      CRH 1  vp_model.crh    ! 速度模型
      STA station.sta        ! 台站文件
      PHS phase.pha          ! 震相文件
      SUM out.sum            ! 输出汇总
      HYP out.hyp            ! 输出详细

    phase.pha（HYPOINVERSE 格式）:
      YYYYMMDD HHMM SS.ss  LAT  LON  DEP  MAG
      SSSSII WHPHS WTRES ...
""")


# ---------------------------------------------------------------------------
# Registry interface
# ---------------------------------------------------------------------------

def list_tools() -> List[str]:
    """List all available tools (built-in + user-registered)."""
    tools = list(BUILTIN_TOOLS.keys())
    if _REGISTRY_DIR.exists():
        for f in _REGISTRY_DIR.glob("*.json"):
            key = f.stem
            if key not in tools:
                tools.append(key)
    return sorted(tools)


def get_tool(name: str) -> Optional[Dict]:
    """
    Get tool profile by name (case-insensitive).

    Checks user registry first, then built-in knowledge.
    """
    key = name.lower().replace(" ", "_").replace("-", "")
    # Also try common aliases
    aliases = {
        "hypodd": "hypodd",
        "velest": "velest",
        "nonlinloc": "nonlinloc",
        "nlloc": "nonlinloc",
        "hypoinverse": "hypoinverse",
        "hyp": "hypoinverse",
        "focmec": "focmec",
    }
    key = aliases.get(key, key)

    # Check user registry
    if _REGISTRY_DIR.exists():
        for f in _REGISTRY_DIR.glob("*.json"):
            if f.stem == key:
                try:
                    with open(f, encoding="utf-8") as fp:
                        return json.load(fp)
                except Exception:
                    pass

    # Check built-in
    return BUILTIN_TOOLS.get(key)


def register_tool(profile_dict: Dict) -> str:
    """Save a tool profile dict to the user registry. Returns saved path."""
    _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    import re
    name = profile_dict.get("name", "unknown")
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", name.lower())
    out = _REGISTRY_DIR / f"{safe}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profile_dict, f, ensure_ascii=False, indent=2)
    return str(out)


def generate_input_files(
    tool_name: str,
    data: Dict,
    output_dir: str,
    llm_config: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Use LLM to generate input files for an external tool based on user data.

    Parameters
    ----------
    tool_name : str
        Tool name (from registry).
    data : dict
        User data context: paths to picks file, station file, event catalog, etc.
    output_dir : str
        Directory to write generated input files.
    llm_config : dict, optional

    Returns
    -------
    dict  mapping filename → absolute path of generated file
    """
    import urllib.request
    import urllib.error

    profile = get_tool(tool_name)
    if profile is None:
        raise ValueError(f"Tool not found: {tool_name}. Available: {list_tools()}")

    if llm_config is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config_manager import get_config_manager
            llm_config = get_config_manager().get_llm_config()
        except Exception:
            llm_config = {"provider": "ollama", "model": "qwen2.5:7b",
                          "api_base": "http://localhost:11434"}

    tool_desc = json.dumps(profile, ensure_ascii=False, indent=2)
    data_desc = json.dumps(data, ensure_ascii=False, indent=2)

    prompt = f"""根据以下工具接口描述和用户数据，生成工具所需的所有输入文件内容。

工具描述:
{tool_desc}

用户数据 (文件路径和说明):
{data_desc}

请生成每个输入文件的完整内容。输出格式为 JSON:
{{
  "filename1.inp": "文件内容...",
  "filename2.dat": "文件内容...",
  ...
}}
只输出 JSON，不要有其他内容。"""

    messages = [
        {"role": "system", "content": "你是专业的地震学数据处理专家，熟悉各类地震定位和反演软件的输入格式。"},
        {"role": "user", "content": prompt},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.1, "num_predict": 4096}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 4096}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if provider == "ollama":
        raw = body.get("message", {}).get("content", "{}")
    else:
        raw = body.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    # Parse generated files
    import re as _re
    raw = _re.sub(r"```json\s*", "", raw)
    raw = _re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        files_dict = json.loads(raw)
    except json.JSONDecodeError:
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        files_dict = json.loads(m.group()) if m else {}

    # Write files to output directory
    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for fname, content in files_dict.items():
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        written[fname] = fpath

    return written


def run_tool(
    tool_name: str,
    input_dir: str,
    extra_args: Optional[str] = None,
    timeout: int = 300,
) -> Dict:
    """
    Run an external seismology tool in the given directory.

    Parameters
    ----------
    tool_name : str
    input_dir : str
        Working directory (should contain all input files).
    extra_args : str, optional
        Additional command-line arguments.
    timeout : int
        Maximum execution time (seconds).

    Returns
    -------
    dict
        {success, returncode, stdout, stderr}
    """
    profile = get_tool(tool_name)
    if profile is None:
        return {"success": False, "returncode": -1, "stdout": "", "stderr": f"Tool not found: {tool_name}"}

    executable = profile.get("executable", tool_name)
    run_cmd = profile.get("run_command", executable)

    # Build command
    cmd = run_cmd
    if extra_args:
        cmd += f" {extra_args}"

    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=input_dir,
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "stdout": "", "stderr": f"超时 (>{timeout}s)"}
    except FileNotFoundError:
        return {
            "success": False, "returncode": -1, "stdout": "",
            "stderr": f"找不到可执行文件 '{executable}'，请确认已安装并在 PATH 中"
        }
