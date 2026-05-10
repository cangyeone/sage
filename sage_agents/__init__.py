"""
sage_agents — Higher-order scientific reasoning agents for SAGE.

Agents
------
LiteratureLoopAgent      : iterative geoscience literature interpretation agent.
EvidenceDrivenGeoAgent   : autonomous, tool-using evidence-driven interpretation agent
                           with multimodal support (figures, tables) and optional
                           web search.
"""

from .literature_loop_agent import (
    LiteratureLoopAgent,
    LoopController,
    AgentResult,
    Evidence,
    Hypothesis,
    ValidationCheck,
)

from .evidence_driven_geo_agent import (
    EvidenceDrivenGeoAgent,
    AgentConfig,
    ToolRegistry,
    LocalFileSearchTool,
    LiteratureLibraryTool,
    RAGIndexTool,
    SeismoDataTool,
    GeoPlotTool,
    CodeExecutionTool,
    StateMemoryTool,
    ImageAnalysisTool,
    WebSearchTool,
    AgentLogger,
    GeoEvidence,
    GeoHypothesis,
    GeoAgentResult,
)

__all__ = [
    # Literature Loop Agent
    "LiteratureLoopAgent",
    "LoopController",
    "AgentResult",
    "Evidence",
    "Hypothesis",
    "ValidationCheck",
    # Evidence-Driven Geo Agent
    "EvidenceDrivenGeoAgent",
    "AgentConfig",
    "ToolRegistry",
    "LocalFileSearchTool",
    "LiteratureLibraryTool",
    "RAGIndexTool",
    "SeismoDataTool",
    "GeoPlotTool",
    "CodeExecutionTool",
    "StateMemoryTool",
    "ImageAnalysisTool",
    "WebSearchTool",
    "AgentLogger",
    "GeoEvidence",
    "GeoHypothesis",
    "GeoAgentResult",
]
