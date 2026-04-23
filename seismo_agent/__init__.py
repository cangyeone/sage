"""
seismo_agent — 地震学自主 Agent

子模块
------
paper_reader   文献加载（PDF / arXiv / DOI / 文本）→ Paper 对象
memory         Agent 工作记忆（文献、步骤结果、变量）
planner        LLM 任务规划（目标 + 文献 → 执行步骤列表）
agent_loop     主循环（规划 → 逐步代码生成 → 执行 → 汇总）

快速使用
--------
from seismo_agent import SeismoAgent

agent = SeismoAgent()
result = agent.run(
    goal="实现论文中的 HVSR 分析方法",
    paper_source="/path/to/paper.pdf",
    output_dir="results/agent_run/",
    progress_cb=print,
)
"""

from .paper_reader import load_paper, read_pdf, fetch_arxiv, fetch_doi, read_text, PaperStore, Paper
from .memory import AgentMemory, StepResult
from .planner import TaskPlanner, PlanStep
from .agent_loop import SeismoAgent, AgentRunResult

__all__ = [
    "load_paper", "read_pdf", "fetch_arxiv", "fetch_doi", "read_text",
    "PaperStore", "Paper",
    "AgentMemory", "StepResult",
    "TaskPlanner", "PlanStep",
    "SeismoAgent", "AgentRunResult",
]
