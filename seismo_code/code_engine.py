"""
code_engine.py — LLM 驱动的地震学代码生成与执行引擎

工作流
------
1. 用户提出自然语言需求（"对这段波形做 1-10Hz 带通滤波并画图"）
2. CodeEngine 构建包含地震学上下文的系统提示
3. 调用本地 LLM（Ollama）或兼容 OpenAI API 的模型生成 Python 代码
4. 调用 safe_executor 在子进程中执行
5. 若执行失败，附带错误信息重试（最多 2 次）
6. 返回执行结果（输出、图像路径、错误信息）

对话上下文
----------
CodeEngine 维护多轮对话历史，LLM 可以跨轮次引用已生成的变量和文件路径。
"""

from __future__ import annotations

import json
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .safe_executor import ExecutionResult, execute_code

# seismo_skill 技能文档检索（可选依赖，不影响主流程）
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from seismo_skill import build_skill_context as _build_skill_context
except Exception:
    def _build_skill_context(query: str, **_kw) -> str:  # type: ignore
        return ""


# ---------------------------------------------------------------------------
# Seismology context injected into the LLM system prompt
# ---------------------------------------------------------------------------

_TOOLKIT_SUMMARY = """
## 内置地震学工具包（直接调用，无需 import）

### 数据读取
- `read_stream(path)` → obspy.Stream  读取 mseed/sac 等波形文件或目录
- `read_stream_from_dir(directory)` → Stream  递归读取目录下所有文件

### 波形处理
- `detrend_stream(st, type='demean')` → Stream
- `taper_stream(st, max_percentage=0.05)` → Stream
- `filter_stream(st, filter_type, freqmin, freqmax, corners=4, zerophase=True)` → Stream
  filter_type: 'bandpass' | 'lowpass' | 'highpass' | 'bandstop'
- `resample_stream(st, sampling_rate)` → Stream
- `trim_stream(st, starttime, endtime)` → Stream
- `merge_stream(st)` → Stream
- `remove_response(st, inventory_path, output='VEL')` → Stream

### 可视化
- `plot_stream(st, title, outfile, picks, normalize=True)` → str(图像路径)
  picks = [{'time': UTCDateTime, 'phase': 'Pg', 'station': 'YN.YSW03'}]
- `plot_spectrogram(tr, title, outfile, wlen=1.0)` → str
- `plot_psd(tr, title, outfile)` → (freqs, psd, str)
- `plot_particle_motion(st, outfile)` → str
- `plot_travel_time_curve(dist_range, depth_km, model, phases)` → str

### 走时计算
- `taup_arrivals(dist_deg, depth_km, model='iasp91', phases)` → list of dict
- `p_travel_time(dist_km, depth_km, model)` → float(秒)
- `s_travel_time(dist_km, depth_km, model)` → float(秒)

### 频谱分析
- `compute_spectrum(tr, method='fft')` → (freqs, amplitudes)
- `compute_hvsr(st, freqmin, freqmax)` → (freqs, hvsr, str)

### 震源参数
- `estimate_magnitude_ml(tr, dist_km)` → float(ML)
- `estimate_corner_freq(tr, t_start, t_end, freqmin, freqmax)` → float(fc, Hz)
- `estimate_seismic_moment(tr, dist_km)` → float(M₀, N·m)
- `moment_to_mw(M0)` → float(Mw)
- `estimate_stress_drop(M0, fc, velocity)` → float(Pa)

### 工具
- `stream_info(st)` → str  打印台网/通道/采样率/时间范围
- `picks_to_dict(picks_file)` → list of dict  读取 SAGE 拾取文件

### 图像保存
- 所有 plot_* 函数会自动将图像保存到当前运行目录并打印路径
- 手动保存: `savefig('filename.png')`（已在环境中注入）
"""

_SYSTEM_PROMPT = """你是一位地震学数据处理专家和 Python 程序员。
用户会用自然语言描述地震学数据处理、分析和可视化需求，你的任务是生成可直接执行的 Python 代码。

## 规则
1. 只输出 Python 代码块（```python ... ```），不要输出任何解释
2. 代码必须能独立执行，不要假设有全局变量（除非之前对话中已明确定义）
3. 如果需要读取文件，使用用户提供的路径或对话历史中的路径
4. 优先使用内置工具包（无需 import），也可以使用 obspy / numpy / scipy / matplotlib
5. 生成的图像使用 plot_* 函数或 savefig() 保存，不要调用 plt.show()
6. 有错误时打印友好的中文提示，用 try/except 保护关键步骤
7. 数值结果用 print() 输出，格式清晰

## 可用库（已安装）
- obspy（地震数据读取、处理、仪器响应去除）
- numpy, scipy（数值计算、信号处理）
- matplotlib（绘图，Agg 后端，不用 show()）

{toolkit_summary}

## 关于外部工具（HypoDD、VELEST 等）
如果用户要求调用外部程序，生成相应的输入文件并用 subprocess.run() 调用，
或者生成格式说明文档，由用户手动运行。
""".format(toolkit_summary=_TOOLKIT_SUMMARY)


# ---------------------------------------------------------------------------
# LLM client (Ollama / OpenAI-compatible)
# ---------------------------------------------------------------------------

def _call_llm(messages: List[Dict], llm_config: Dict) -> str:
    """Call LLM and return the assistant message content."""
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")
    temperature = llm_config.get("temperature", 0.2)
    max_tokens = llm_config.get("max_tokens", 4096)

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    else:
        # OpenAI-compatible
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"LLM 服务连接失败（{url}）: {e}")

    # Extract content from response
    if provider == "ollama":
        return body.get("message", {}).get("content", "")
    else:
        choices = body.get("choices", [{}])
        return choices[0].get("message", {}).get("content", "")


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (```python ... ``` block)."""
    # Try fenced code block
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try any code block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Assume the whole response is code
    return text.strip()


# ---------------------------------------------------------------------------
# Code Engine
# ---------------------------------------------------------------------------

class CodeEngine:
    """
    LLM-driven seismological code generation and execution engine.

    Usage
    -----
    engine = CodeEngine(llm_config)
    result = engine.run("对 /data/wave.mseed 做 1-10Hz 带通滤波并绘图")
    print(result.response)
    for fig in result.figures:
        print(fig)
    """

    def __init__(self, llm_config: Optional[Dict] = None, project_root: Optional[str] = None):
        if llm_config is None:
            llm_config = self._load_llm_config()
        self.llm_config = llm_config
        self.project_root = project_root or str(Path(__file__).parent.parent)
        # Conversation history for multi-turn code generation
        self._history: List[Dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        self._last_exec_dir: Optional[str] = None

    @staticmethod
    def _load_llm_config() -> Dict:
        try:
            import sys
            from pathlib import Path as _P
            sys.path.insert(0, str(_P(__file__).parent.parent))
            from config_manager import LLMConfigManager
            return LLMConfigManager().get_llm_config()
        except Exception:
            return {"provider": "ollama", "model": "",
                    "api_base": "http://localhost:11434"}

    def is_llm_available(self) -> bool:
        """Quick check whether the configured LLM endpoint is reachable."""
        try:
            provider = self.llm_config.get("provider", "ollama")
            api_base = self.llm_config.get("api_base", "http://localhost:11434")
            url = api_base.rstrip("/") + ("/api/tags" if provider == "ollama" else "/models")
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    def run(
        self,
        user_request: str,
        data_hint: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 90,
    ) -> "CodeRunResult":
        """
        Generate code for *user_request* and execute it.

        Parameters
        ----------
        user_request : str
            Natural-language description of what to do.
        data_hint : str, optional
            File or directory path to include in the prompt context.
        max_retries : int
            How many times to retry on execution failure (with error feedback).
        timeout : int
            Execution timeout in seconds.

        Returns
        -------
        CodeRunResult
        """
        # Build user message
        msg_content = user_request
        if data_hint:
            msg_content += f"\n\n数据路径: {data_hint}"
        if self._last_exec_dir:
            msg_content += f"\n（上一次运行目录: {self._last_exec_dir}）"

        self._history.append({"role": "user", "content": msg_content})

        # Inject relevant skill docs into system prompt for this turn
        skill_ctx = _build_skill_context(user_request, max_chars=5000, top_k=2)
        if skill_ctx:
            # Prepend skill context as a system-role injection at the front
            messages_base = [
                {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + skill_ctx}
            ] + [m for m in self._history if m["role"] != "system"]
        else:
            messages_base = list(self._history)

        last_error = ""
        exec_result: Optional[ExecutionResult] = None
        code = ""

        for attempt in range(max_retries + 1):
            # Build messages for this attempt
            messages = list(messages_base)
            if attempt > 0 and last_error:
                messages.append({
                    "role": "user",
                    "content": (
                        f"上面生成的代码执行出错，错误信息如下，请修正代码：\n\n"
                        f"```\n{last_error}\n```\n\n"
                        "请只输出修正后的完整 Python 代码块，不要输出其他内容。"
                    )
                })

            # Call LLM
            try:
                raw_response = _call_llm(messages, self.llm_config)
            except ConnectionError as e:
                return CodeRunResult(
                    success=False,
                    response=str(e),
                    code="",
                    exec_result=None,
                )

            code = _extract_code(raw_response)

            # Execute
            exec_result = execute_code(
                code,
                project_root=self.project_root,
                timeout=timeout,
                keep_dir=True,  # Keep for figure collection
            )

            if exec_result.success:
                break

            last_error = f"{exec_result.stderr}\n{exec_result.error}".strip()

        # Add assistant turn to history (with the code)
        self._history.append({
            "role": "assistant",
            "content": f"```python\n{code}\n```"
        })

        if exec_result:
            self._last_exec_dir = exec_result.exec_dir

        return CodeRunResult(
            success=exec_result.success if exec_result else False,
            response=exec_result.short_summary() if exec_result else "执行失败",
            code=code,
            exec_result=exec_result,
        )

    def reset(self):
        """Reset conversation history."""
        self._history = [{"role": "system", "content": _SYSTEM_PROMPT}]
        self._last_exec_dir = None


@dataclass
class CodeRunResult:
    success: bool
    response: str         # Human-readable summary
    code: str             # Generated Python code
    exec_result: Optional[ExecutionResult]

    @property
    def figures(self) -> List[str]:
        return self.exec_result.figures if self.exec_result else []

    @property
    def output_files(self) -> List[str]:
        return self.exec_result.output_files if self.exec_result else []

    @property
    def stdout(self) -> str:
        return self.exec_result.stdout if self.exec_result else ""


# ---------------------------------------------------------------------------
# Singleton / factory
# ---------------------------------------------------------------------------

_engine_instance: Optional[CodeEngine] = None


def get_code_engine(llm_config: Optional[Dict] = None) -> CodeEngine:
    """Get or create the global CodeEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CodeEngine(llm_config)
    return _engine_instance


def reset_code_engine():
    """Reset conversation context of the global engine."""
    global _engine_instance
    if _engine_instance:
        _engine_instance.reset()
