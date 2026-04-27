"""
seismo_skill — 地震学技能文档库 + 工作流系统

提供函数工具的使用说明和示例代码，供 Agent 和 CodeEngine 在代码生成时参考。

目录结构
--------
  seismo_skill/skills/      内置技能 .md 文件
  seismo_skill/workflows/   内置工作流 .md 文件
  seismo_skill/knowledge/   知识文档目录（供 RAG 索引）
  ~/.seismicx/skills/       用户自定义技能（运行时）
  ~/.seismicx/workflows/    用户自定义工作流（运行时）
"""

from . import skill_loader
from .skill_loader import (
    list_skills,
    load_skill,
    search_skills,
    build_skill_context,
    build_skill_context_with_rag,
    invalidate_cache,
    save_user_skill,
    delete_user_skill,
    get_user_skill_dir,
    get_skill_detail,
)

from . import workflow_runner
from .workflow_runner import (
    list_workflows,
    search_workflows,
    load_workflow,
    save_user_workflow,
    delete_user_workflow,
    build_workflow_context,
    get_user_workflow_dir,
    invalidate_cache as invalidate_workflow_cache,
)

__all__ = [
    # skills
    "skill_loader",
    "list_skills",
    "load_skill",
    "search_skills",
    "build_skill_context",
    "build_skill_context_with_rag",
    "invalidate_cache",
    "save_user_skill",
    "delete_user_skill",
    "get_user_skill_dir",
    "get_skill_detail",
    # workflows
    "workflow_runner",
    "list_workflows",
    "search_workflows",
    "load_workflow",
    "save_user_workflow",
    "delete_user_workflow",
    "build_workflow_context",
    "get_user_workflow_dir",
    "invalidate_workflow_cache",
]
