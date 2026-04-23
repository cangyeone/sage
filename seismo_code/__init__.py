"""
seismo_code — LLM 驱动的地震学代码生成与执行引擎

子模块
------
safe_executor   在子进程中安全执行 Python 代码，捕获输出与图像
toolkit         内置地震学工具函数（滤波、走时、震源参数、可视化等）
code_engine     LLM 代码生成与多轮执行引擎
doc_parser      从文档文本/文件中解析外部工具接口，生成 ToolProfile
"""

from .safe_executor import execute_code, ExecutionResult
from .toolkit import *  # noqa: F401,F403  expose all toolkit functions
from .code_engine import CodeEngine, CodeRunResult, get_code_engine, reset_code_engine
from .doc_parser import DocParser, ToolProfile

__all__ = [
    "execute_code", "ExecutionResult",
    "CodeEngine", "CodeRunResult", "get_code_engine", "reset_code_engine",
    "DocParser", "ToolProfile",
]
