"""
memory.py — Agent 工作记忆

保存跨步骤积累的状态：
  - 已加载的文献（PaperStore）
  - 各步骤执行结果（变量、图像、输出文本）
  - 对话历史摘要
  - 当前任务计划
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

from .paper_reader import Paper, PaperStore


@dataclass
class StepResult:
    """单个执行步骤的结果记录。"""
    step_index: int
    description: str
    code: str = ""
    stdout: str = ""
    figures: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def brief(self) -> str:
        status = "✓" if self.success else "✗"
        lines = [f"{status} 步骤 {self.step_index}: {self.description}"]
        if self.stdout.strip():
            out = self.stdout.strip()[:200]
            lines.append(f"   输出: {out}")
        if self.figures:
            lines.append(f"   图像: {len(self.figures)} 张")
        if self.error:
            lines.append(f"   错误: {self.error[:100]}")
        return "\n".join(lines)


class AgentMemory:
    """Agent 的完整工作记忆。"""

    def __init__(self):
        self.papers = PaperStore()
        self.plan: List[str] = []           # 规划的步骤描述列表
        self.step_results: List[StepResult] = []
        self.variables: Dict[str, Any] = {} # 跨步骤共享变量
        self.figures: List[str] = []        # 所有生成的图像路径
        self.output_files: List[str] = []
        self.goal: str = ""
        self.output_dir: str = ""
        self.notes: List[str] = []          # Agent 的自我注释

    # ---- Papers ----
    def add_paper(self, paper: Paper, key: Optional[str] = None) -> str:
        return self.papers.add(paper, key)

    def get_paper_context(self, max_chars: int = 6000) -> str:
        if len(self.papers) == 0:
            return ""
        return self.papers.combined_context(max_chars_per_paper=max_chars // max(len(self.papers), 1))

    # ---- Steps ----
    def record_step(self, result: StepResult):
        self.step_results.append(result)
        self.figures.extend(result.figures)
        self.output_files.extend(result.output_files)

    def steps_summary(self) -> str:
        if not self.step_results:
            return "（尚无执行记录）"
        return "\n".join(r.brief() for r in self.step_results)

    def artifacts_summary(self, max_files: int = 20) -> str:
        """Summarize generated artifacts so later LLM steps can reuse them."""
        if not self.output_files and not self.figures:
            return ""
        lines = ["=== 已生成/可复用中间产物 ==="]
        seen = set()
        for path_str in list(self.output_files) + list(self.figures):
            if not path_str or path_str in seen:
                continue
            seen.add(path_str)
            path = Path(path_str)
            desc = f"- {path.name}: {path_str}"
            if path.suffix.lower() == ".csv" and path.exists():
                try:
                    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        header = next(reader, [])
                        row_count = sum(1 for _ in reader)
                    desc += f" | CSV rows={row_count}, columns={header[:16]}"
                except Exception as exc:
                    desc += f" | CSV metadata unavailable: {exc}"
            elif path.exists():
                try:
                    desc += f" | size={path.stat().st_size} bytes"
                except Exception:
                    pass
            lines.append(desc)
            if len(lines) > max_files:
                lines.append(f"- ... 还有 {len(seen) - max_files + 1} 个文件")
                break
        return "\n".join(lines)

    def accumulated_context(self, max_chars: int = 3000) -> str:
        """
        返回给 LLM 的积累上下文：各步骤输出摘要 + 变量列表。
        """
        parts = []
        artifacts = self.artifacts_summary()
        if artifacts:
            parts.append(artifacts)
        if self.step_results:
            parts.append("=== 已执行步骤 ===")
            for r in self.step_results[-5:]:  # 只传最近 5 步
                parts.append(r.brief())
        if self.variables:
            parts.append("=== 已计算变量 ===")
            for k, v in list(self.variables.items())[:10]:
                parts.append(f"  {k} = {repr(v)[:80]}")
        combined = "\n".join(parts)
        return combined[:max_chars]

    def add_note(self, note: str):
        self.notes.append(note)
