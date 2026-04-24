"""
planner.py — LLM 驱动的任务规划器

输入：用户目标 + 文献内容摘要
输出：结构化的执行步骤列表，每步包含：
  - 描述（做什么）
  - 类型（code / tool / qa / search）
  - 预期输出

规划原则
--------
1. 先充分理解文献中的方法，再规划实现步骤
2. 每步粒度适中（不超过 30 行代码）
3. 步骤之间依赖关系清晰（后步可复用前步输出）
4. 最后一步是结果验证/可视化
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    index: int
    description: str                    # 人类可读的步骤说明
    step_type: str = "code"             # 'code' | 'tool' | 'qa' | 'search'
    expected_output: str = ""           # 预期产出（如 "滤波后的波形图"）
    depends_on: List[int] = field(default_factory=list)  # 依赖的前序步骤索引
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[步骤 {self.index}] ({self.step_type}) {self.description}"


_PLANNER_SYSTEM = """\
You are an experienced seismology researcher and software engineer.
Given a research goal (and optionally paper content), break the task into clear execution steps.

Required output format (strict JSON array, nothing else):
[
  {
    "index": 1,
    "description": "Concise step description (≤ 15 words)",
    "step_type": "code",
    "expected_output": "Expected output (e.g., 'filtered waveform plot')",
    "depends_on": []
  },
  ...
]

step_type values:
- "code"   Generate and execute Python code
- "tool"   Call external tool (HypoDD, VELEST, etc.)
- "qa"     Text explanation only, no code execution
- "search" Search literature for specific information

Planning principles:
1. If paper is provided, start with 1-2 steps to extract key methods/formulas
2. Data preparation steps (read, preprocess) come first
3. Each step does one thing; keep granularity moderate
4. Final step is result visualization or verification
5. Aim for 3-8 total steps — avoid over-splitting
"""


def _call_planner_llm(messages: List[Dict], llm_config: Dict) -> str:
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": 2048}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 2048}

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())

    if provider == "ollama":
        return body.get("message", {}).get("content", "[]")
    return body.get("choices", [{}])[0].get("message", {}).get("content", "[]")


def _parse_plan(raw: str) -> List[PlanStep]:
    """Parse LLM response into PlanStep list."""
    # Strip code fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()

    # Find JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return _fallback_plan()

    try:
        steps_raw = json.loads(m.group())
    except json.JSONDecodeError:
        return _fallback_plan()

    steps = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict):
            continue
        steps.append(PlanStep(
            index=s.get("index", i + 1),
            description=s.get("description", f"步骤 {i+1}"),
            step_type=s.get("step_type", "code"),
            expected_output=s.get("expected_output", ""),
            depends_on=s.get("depends_on", []),
            metadata=s.get("metadata", {}),
        ))
    return steps if steps else _fallback_plan()


def _fallback_plan() -> List[PlanStep]:
    """Default 3-step plan when LLM parsing fails."""
    return [
        PlanStep(1, "Prepare data and verify format", "code", "Data loaded"),
        PlanStep(2, "Implement core processing", "code", "Processing result"),
        PlanStep(3, "Visualize and output results", "code", "Result figures"),
    ]


class TaskPlanner:
    """
    LLM 驱动的任务规划器。

    Usage
    -----
    planner = TaskPlanner(llm_config)
    steps = planner.plan(goal="实现论文中的走时校正方法",
                         paper_context="[Abstract] ...")
    """

    def __init__(self, llm_config: Optional[Dict] = None):
        if llm_config is None:
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config_manager import get_config_manager
                llm_config = get_config_manager().get_llm_config()
            except Exception:
                llm_config = {"provider": "ollama", "model": "qwen2.5:7b",
                              "api_base": "http://localhost:11434"}
        self.llm_config = llm_config

    def plan(
        self,
        goal: str,
        paper_context: str = "",
        prev_results_context: str = "",
    ) -> List[PlanStep]:
        """
        Generate a task plan.

        Parameters
        ----------
        goal : str
            User's research/programming goal.
        paper_context : str
            Extracted paper content (abstract + methods).
        prev_results_context : str
            Summary of previously completed steps (for replanning).

        Returns
        -------
        List[PlanStep]
        """
        user_content = f"Goal: {goal}\n"
        if paper_context:
            user_content += f"\nPaper content summary:\n{paper_context[:4000]}"
        if prev_results_context:
            user_content += f"\n\nAlready completed steps:\n{prev_results_context}"
        user_content += "\n\nGenerate execution steps (JSON array):"

        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = _call_planner_llm(messages, self.llm_config)
            return _parse_plan(raw)
        except Exception:
            return _fallback_plan()

    def replan(
        self,
        goal: str,
        completed_steps: List[PlanStep],
        failed_step: Optional[PlanStep],
        paper_context: str = "",
    ) -> List[PlanStep]:
        """
        Adjust plan after a step fails.
        Returns remaining steps (with adjustments).
        """
        completed_desc = "\n".join(
            f"  ✓ [{s.index}] {s.description}" for s in completed_steps
        )
        failed_desc = (
            f"  ✗ [{failed_step.index}] {failed_step.description}" if failed_step else ""
        )
        user_content = (
            f"Goal: {goal}\n\n"
            f"Completed steps:\n{completed_desc}\n"
            f"Failed step:\n{failed_desc}\n\n"
            f"{'Paper summary: ' + paper_context[:2000] if paper_context else ''}\n\n"
            "Re-plan remaining steps (JSON array, continue indexing from failed step):"
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        try:
            raw = _call_planner_llm(messages, self.llm_config)
            return _parse_plan(raw)
        except Exception:
            return [PlanStep(
                len(completed_steps) + 1,
                "Complete remaining task with alternative approach",
                "code",
                "Result"
            )]
