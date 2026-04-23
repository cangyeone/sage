"""
seismo_tools — 外部地震学工具注册表与接口

子模块
------
tool_registry   内置工具知识库（HypoDD/VELEST/NonLinLoc/HYPOINVERSE/focmec）
                + 用户自定义工具注册、输入文件生成、工具调用
"""

from .tool_registry import (
    list_tools,
    get_tool,
    register_tool,
    generate_input_files,
    run_tool,
    BUILTIN_TOOLS,
)

__all__ = [
    "list_tools", "get_tool", "register_tool",
    "generate_input_files", "run_tool",
    "BUILTIN_TOOLS",
]
