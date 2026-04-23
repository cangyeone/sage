"""
seismo_skill — 地震学技能文档库

提供函数工具的使用说明和示例代码，供 Agent 和 CodeEngine 在代码生成时参考。
支持内置技能（seismo_skill/）和用户自定义技能（~/.seismicx/skills/）。
"""

from . import skill_loader
from .skill_loader import (
    list_skills,
    load_skill,
    search_skills,
    build_skill_context,
    invalidate_cache,
    save_user_skill,
    delete_user_skill,
    get_user_skill_dir,
    get_skill_detail,
)

__all__ = [
    "skill_loader",
    "list_skills",
    "load_skill",
    "search_skills",
    "build_skill_context",
    "invalidate_cache",
    "save_user_skill",
    "delete_user_skill",
    "get_user_skill_dir",
    "get_skill_detail",
]
