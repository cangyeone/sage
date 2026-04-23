"""
safe_executor.py — 安全地在子进程中执行 LLM 生成的 Python 代码。

工作原理
--------
1. 将代码写入临时目录
2. 在子进程中执行，有超时保护
3. 捕获 stdout / stderr
4. 收集生成的图像文件（PNG/PDF）
5. 返回 ExecutionResult 结构

安全说明
--------
• 代码在独立子进程中运行，主进程不受崩溃影响
• 限制执行超时（默认 60 秒）
• 禁止子进程继续 fork（可选）
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""           # Short error summary
    figures: List[str] = field(default_factory=list)   # Absolute paths of generated images
    output_files: List[str] = field(default_factory=list)  # Other generated files
    variables: Dict[str, Any] = field(default_factory=dict)  # Exported variables (via SAGE_EXPORT)
    exec_dir: str = ""        # The temp directory used (kept for inspection)

    def short_summary(self) -> str:
        lines = []
        if self.success:
            lines.append("✓ 代码执行成功")
        else:
            lines.append(f"✗ 执行失败: {self.error}")
        if self.stdout.strip():
            lines.append("输出:\n" + textwrap.indent(self.stdout.strip(), "  "))
        if self.figures:
            lines.append(f"生成图像 ({len(self.figures)} 个):")
            for f in self.figures:
                lines.append(f"  • {f}")
        if self.output_files:
            lines.append(f"生成文件 ({len(self.output_files)} 个):")
            for f in self.output_files:
                lines.append(f"  • {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

# Preamble injected at the top of every executed script
_PREAMBLE = """
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

# Add project root to path so seismo_* modules are importable
_proj = os.environ.get('SAGE_PROJECT_ROOT', '.')
if _proj not in sys.path:
    sys.path.insert(0, _proj)

# Import the built-in seismology toolkit
try:
    from seismo_code.toolkit import *   # noqa: F401,F403
except ImportError:
    pass

# matplotlib non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Directory for this run — all relative file saves land here
_OUTDIR = os.environ.get('SAGE_OUTDIR', '.')

def _savefig(name):
    \"\"\"Save current matplotlib figure to the output directory.\"\"\"
    import matplotlib.pyplot as _plt
    path = os.path.join(_OUTDIR, name)
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f'[FIGURE] {path}')
    return path

# Make savefig available as a helper
savefig = _savefig
"""


def execute_code(
    code: str,
    project_root: Optional[str] = None,
    timeout: int = 60,
    keep_dir: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> ExecutionResult:
    """
    Execute Python code in an isolated subprocess.

    Parameters
    ----------
    code : str
        Python source code to execute.
    project_root : str, optional
        Path to the SAGE project root (added to PYTHONPATH).
    timeout : int
        Maximum execution time in seconds. Default 60.
    keep_dir : bool
        If True, do not delete the temp directory after execution.
    extra_env : dict, optional
        Additional environment variables.

    Returns
    -------
    ExecutionResult
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent)

    # Create temp directory for this run
    tmp = tempfile.mkdtemp(prefix="sage_exec_")
    script_path = os.path.join(tmp, "run.py")

    # Assemble full script
    full_code = _PREAMBLE + "\n" + code
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    # Build environment
    env = os.environ.copy()
    env["SAGE_PROJECT_ROOT"] = project_root
    env["SAGE_OUTDIR"] = tmp
    env["MPLBACKEND"] = "Agg"
    # Add project root to PYTHONPATH
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else project_root
    if extra_env:
        env.update(extra_env)

    # Execute
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp,
            env=env,
        )
        success = proc.returncode == 0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        error = ""
        if not success:
            # Extract the last traceback line as short error
            lines = stderr.strip().splitlines()
            error = lines[-1] if lines else f"Exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = ""
        error = f"执行超时（>{timeout}s）"
    except Exception as e:
        success = False
        stdout = ""
        stderr = str(e)
        error = str(e)

    # Collect generated files
    figures = []
    output_files = []

    # Find files mentioned in stdout as [FIGURE] ...
    for line in stdout.splitlines():
        if line.startswith("[FIGURE] "):
            fig_path = line[len("[FIGURE] "):].strip()
            if os.path.isfile(fig_path):
                figures.append(fig_path)

    # Also scan the temp directory for any image/data files not already captured
    if os.path.isdir(tmp):
        for fname in sorted(os.listdir(tmp)):
            fpath = os.path.join(tmp, fname)
            if fname == "run.py":
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".png", ".pdf", ".svg") and fpath not in figures:
                figures.append(fpath)
            elif ext not in (".py",) and os.path.isfile(fpath) and fpath not in output_files:
                output_files.append(fpath)

    if not keep_dir and not figures and not output_files:
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    return ExecutionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        error=error,
        figures=figures,
        output_files=output_files,
        exec_dir=tmp,
    )
