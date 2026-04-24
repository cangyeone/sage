"""
agent_loop.py — 地震学自主 Agent 主循环

工作流程
--------
1. 加载文献（PDF / arXiv / DOI / 文本）
2. 用 LLM 理解文献，提取核心方法
3. 规划实现步骤（TaskPlanner）
4. 逐步执行（代码生成 + 沙箱执行）
5. 验证每步结果，失败时重试或重规划
6. 汇总所有结果、图像，输出摘要报告

Agent 模式
----------
- interactive  每步执行前征求用户确认（默认）
- autonomous   全自动，不打断用户
- single_step  只执行一步，返回结果
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .memory import AgentMemory, StepResult
from .paper_reader import PaperStore, load_paper
from .planner import PlanStep, TaskPlanner

# seismo_skill 技能文档检索（可选依赖）
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from seismo_skill import build_skill_context as _build_skill_context
except Exception:
    def _build_skill_context(query: str, **_kw) -> str:  # type: ignore
        return ""


# ---------------------------------------------------------------------------
# Code generation prompt for individual steps
# ---------------------------------------------------------------------------

_STEP_SYSTEM = """\
You are a professional seismology researcher and Python engineer.
You are incrementally completing a seismology research task.

Pre-injected seismology toolkit (call directly — no import needed):
  read_stream(path)  read_stream_from_dir(directory)
  detrend_stream(st)  taper_stream(st)  filter_stream(st, type, freqmin, freqmax)
  plot_stream(st, title, outfile)  plot_psd(tr, outfile)
  plot_spectrogram(tr, outfile)  plot_particle_motion(st, outfile)
  taup_arrivals(dist_deg, depth_km, model)  plot_travel_time_curve(...)
  compute_spectrum(tr)  compute_hvsr(st, ...)
  estimate_magnitude_ml(tr, dist_km)  estimate_corner_freq(tr, ...)
  estimate_seismic_moment(tr, dist_km)  moment_to_mw(M0)  estimate_stress_drop(M0, fc)
  stream_info(st)  picks_to_dict(picks_file)  run_gmt(script, outname, title)

Rules:
1. Output ONLY ```python ... ``` code blocks — no explanations
2. Print numerical results with print() using clear labels
3. Save images via plot_* functions or savefig('name.png') — NEVER use plt.show()
4. Wrap critical steps in try/except with informative error messages
5. Variables computed in prior steps are available (listed in context)
"""


def _generate_step_code(
    step: PlanStep,
    paper_context: str,
    memory_context: str,
    llm_config: Dict,
    goal: str,
) -> str:
    """Call LLM to generate code for a single step."""
    user_content = (
        f"Overall goal: {goal}\n\n"
        f"Current step [{step.index}]: {step.description}\n"
        f"Expected output: {step.expected_output}\n\n"
    )
    if paper_context:
        user_content += f"Paper method summary:\n{paper_context[:3000]}\n\n"
    if memory_context:
        user_content += f"Prior step results (variables/files available for reuse):\n{memory_context}\n\n"
    user_content += "Generate complete Python code for this step:"

    # Inject relevant skill documentation based on step description + goal
    skill_query = f"{goal} {step.description}"
    skill_ctx = _build_skill_context(skill_query, max_chars=4000, top_k=2)
    step_system = _STEP_SYSTEM
    if skill_ctx:
        step_system = _STEP_SYSTEM + "\n\n" + skill_ctx

    messages = [
        {"role": "system", "content": step_system},
        {"role": "user", "content": user_content},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.2, "num_predict": 3000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 3000}

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"})

    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode())

    import re
    if provider == "ollama":
        raw = body.get("message", {}).get("content", "")
    else:
        raw = body.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Extract code block
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _explain_paper_methods(paper_context: str, goal: str, llm_config: Dict) -> str:
    """Ask LLM to summarize the key method from the paper for the given goal."""
    if not paper_context.strip():
        return ""

    messages = [
        {"role": "system", "content": (
            "你是一位地震学专家。请从以下文献内容中，提取与用户目标最相关的"
            "核心方法、公式和算法步骤，用简洁的中文总结（500字以内）。"
        )},
        {"role": "user", "content": f"用户目标：{goal}\n\n文献内容：\n{paper_context[:5000]}"},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": 1000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1000}

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        if provider == "ollama":
            return body.get("message", {}).get("content", "")
        return body.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SeismoAgent
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    success: bool
    summary: str
    figures: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    steps_completed: int = 0
    steps_total: int = 0
    output_dir: str = ""


class SeismoAgent:
    """
    地震学自主 Agent

    Usage
    -----
    agent = SeismoAgent(llm_config)
    result = agent.run(
        goal="实现文献中的 HVSR 分析方法并绘图",
        paper_source="/path/to/paper.pdf",
        output_dir="results/agent_run/",
        progress_cb=print,
    )
    """

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        project_root: Optional[str] = None,
        mode: str = "autonomous",
    ):
        if llm_config is None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config_manager import LLMConfigManager
                llm_config = LLMConfigManager().get_llm_config()
            except Exception:
                llm_config = {"provider": "ollama", "model": "",
                              "api_base": "http://localhost:11434"}

        self.llm_config = llm_config
        self.project_root = project_root or str(Path(__file__).parent.parent)
        self.mode = mode
        self.memory = AgentMemory()
        self.planner = TaskPlanner(llm_config)

    def is_llm_available(self) -> bool:
        try:
            provider = self.llm_config.get("provider", "ollama")
            api_base = self.llm_config.get("api_base", "http://localhost:11434")
            url = api_base.rstrip("/") + ("/api/tags" if provider == "ollama" else "/models")
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    def load_paper(self, source: str, cb: Optional[Callable] = None) -> bool:
        """Load a paper from source (PDF path, arXiv ID, DOI, or text)."""
        if cb:
            cb(f"📖 加载文献：{source[:60]}")
        try:
            paper = load_paper(source)
            key = self.memory.add_paper(paper)
            if cb:
                cb(f"   ✓ 已加载：{paper.title[:60]} [{key}]")
            return True
        except Exception as e:
            if cb:
                cb(f"   ⚠  文献加载失败：{e}")
            return False

    def run(
        self,
        goal: str,
        paper_source: Optional[str] = None,
        output_dir: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        max_steps: int = 8,
        max_retries: int = 2,
    ) -> Dict:
        """
        Run the full agentic loop.

        Parameters
        ----------
        goal : str
            Research/programming goal in natural language.
        paper_source : str, optional
            Path/URL/ID of paper to read.
        output_dir : str, optional
            Directory for output figures and files.
        progress_cb : callable, optional
            Called with progress messages.
        max_steps : int
            Safety cap on number of steps.
        max_retries : int
            Retries per step on failure.

        Returns
        -------
        dict  with keys: success, summary, figures, output_files, ...
        """
        cb = progress_cb or (lambda x: None)
        self.memory.goal = goal

        # Output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.memory.output_dir = output_dir
        else:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix="sage_agent_")
            self.memory.output_dir = output_dir

        cb(f"\n🤖 SeismoAgent 启动")
        cb(f"   目标：{goal}")
        cb(f"   输出目录：{output_dir}\n")

        # Check LLM
        if not self.is_llm_available():
            return {
                "success": False,
                "summary": (
                    "⚠️  LLM 服务不可用。\n"
                    "请启动 Ollama（`ollama serve`）或配置 API（`python seismic_cli.py llm setup`）后重试。"
                ),
                "figures": [],
            }

        # Step 1: Load paper(s)
        if paper_source:
            self.load_paper(paper_source, cb)
        else:
            cb("   （无文献输入，根据目标直接规划）")

        # Step 2: Extract key methods from paper
        paper_context = self.memory.get_paper_context()
        method_summary = ""
        if paper_context:
            cb("\n🔍 理解文献方法...")
            try:
                method_summary = _explain_paper_methods(paper_context, goal, self.llm_config)
                if method_summary:
                    cb(f"   核心方法摘要：\n{method_summary[:400]}{'...' if len(method_summary) > 400 else ''}")
            except Exception as e:
                cb(f"   ⚠  方法提取失败（{e}），继续规划...")

        # Step 3: Plan
        cb("\n📋 规划执行步骤...")
        effective_context = method_summary or paper_context[:3000]
        steps = self.planner.plan(goal=goal, paper_context=effective_context)
        steps = steps[:max_steps]
        self.memory.plan = [s.description for s in steps]

        cb(f"   共 {len(steps)} 个步骤：")
        for s in steps:
            cb(f"   {s.index}. [{s.step_type}] {s.description}")

        # Step 4: Execute steps
        cb("\n⚙️  开始执行...\n")
        all_figures: List[str] = []
        all_output_files: List[str] = []
        completed_steps: List[PlanStep] = []

        for step in steps:
            cb(f"── 步骤 {step.index}/{len(steps)}: {step.description}")

            if step.step_type == "qa":
                # QA step — just log, no execution
                result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    stdout=f"（说明步骤，无需执行代码）",
                    success=True,
                )
                self.memory.record_step(result)
                completed_steps.append(step)
                cb(f"   ✓ 说明步骤，跳过执行\n")
                continue

            # Generate and execute code
            exec_result = None
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    cb(f"   ↩  重试 ({attempt}/{max_retries})...")

                try:
                    code = _generate_step_code(
                        step=step,
                        paper_context=effective_context,
                        memory_context=self.memory.accumulated_context(),
                        llm_config=self.llm_config,
                        goal=goal,
                    )
                except Exception as e:
                    cb(f"   ✗ 代码生成失败：{e}")
                    break

                # Execute
                from seismo_code.safe_executor import execute_code
                exec_result = execute_code(
                    code,
                    project_root=self.project_root,
                    timeout=120,
                    keep_dir=True,
                    extra_env={"SAGE_OUTDIR": output_dir},
                )

                if exec_result.success:
                    break

                # On failure, feed error back for retry
                cb(f"   ⚠  执行失败：{exec_result.error[:100]}")

            if exec_result is None:
                step_result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    success=False,
                    error="代码生成失败",
                )
            else:
                # Copy figures to output_dir
                step_figs = []
                for fig in exec_result.figures:
                    dst = os.path.join(output_dir, os.path.basename(fig))
                    try:
                        if fig != dst:
                            shutil.copy2(fig, dst)
                        step_figs.append(dst)
                    except Exception:
                        step_figs.append(fig)

                step_result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    code=code if exec_result.success else "",
                    stdout=exec_result.stdout,
                    figures=step_figs,
                    output_files=exec_result.output_files,
                    success=exec_result.success,
                    error=exec_result.error,
                )

            self.memory.record_step(step_result)
            all_figures.extend(step_result.figures)
            all_output_files.extend(step_result.output_files)

            # Status
            if step_result.success:
                out_preview = step_result.stdout.strip()[:120] if step_result.stdout.strip() else ""
                cb(f"   ✓ 完成" + (f"\n   输出: {out_preview}" if out_preview else ""))
                if step_result.figures:
                    cb(f"   图像: {[os.path.basename(f) for f in step_result.figures]}")
                completed_steps.append(step)
            else:
                cb(f"   ✗ 步骤失败: {step_result.error[:100]}")
            cb("")

        # Step 5: Summary
        n_ok = sum(1 for r in self.memory.step_results if r.success)
        n_total = len(self.memory.step_results)
        overall_success = n_ok == n_total

        summary_lines = [
            f"\n{'✅' if overall_success else '⚠️ '} Agent 执行完成",
            f"   步骤完成: {n_ok}/{n_total}",
        ]
        if all_figures:
            summary_lines.append(f"   生成图像: {len(all_figures)} 张")
            for f in all_figures:
                summary_lines.append(f"     • {f}")
        if all_output_files:
            summary_lines.append(f"   生成文件: {len(all_output_files)} 个")
        summary_lines.append(f"   输出目录: {output_dir}")

        # Detailed step log
        summary_lines.append("\n执行日志:")
        for r in self.memory.step_results:
            summary_lines.append(r.brief())

        summary = "\n".join(summary_lines)
        cb(summary)

        return {
            "success": overall_success,
            "summary": summary,
            "figures": all_figures,
            "output_files": all_output_files,
            "steps_completed": n_ok,
            "steps_total": n_total,
            "output_dir": output_dir,
            "method_summary": method_summary,
        }
