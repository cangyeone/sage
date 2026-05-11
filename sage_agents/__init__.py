"""
sage_agents — Higher-order scientific reasoning agents for SAGE.

Agents
------
LiteratureLoopAgent      : iterative geoscience literature interpretation agent.
"""

from .literature_loop_agent import (
    LiteratureLoopAgent,
    LoopController,
    AgentResult,
    Evidence,
    Hypothesis,
    ValidationCheck,
)

__all__ = [
    # Literature Loop Agent
    "LiteratureLoopAgent",
    "LoopController",
    "AgentResult",
    "Evidence",
    "Hypothesis",
    "ValidationCheck",
]
