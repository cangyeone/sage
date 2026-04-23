"""
doc_parser.py — 从文档（文本/文件）中解析外部地震学工具的接口信息。

功能
----
1. 接受用户粘贴的文档文本或指向本地文件的路径
2. 使用 LLM 提取工具名称、可执行程序、输入文件格式、参数和输出文件
3. 生成 ToolProfile（结构化的工具描述），保存到 seismo_tools/ 工具注册表
4. 后续可根据 ToolProfile 自动生成输入文件并调用工具
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ToolProfile — 结构化工具描述
# ---------------------------------------------------------------------------

@dataclass
class ToolProfile:
    """结构化的外部地震学工具接口描述。"""

    name: str                        # 工具名称 (e.g. "HypoDD")
    executable: str                  # 可执行程序名 (e.g. "hypoDD")
    description: str                 # 功能简介
    input_files: List[str] = field(default_factory=list)   # 需要的输入文件列表
    input_format: str = ""           # 主要输入文件格式的详细描述
    input_template: str = ""         # 输入文件模板（含占位符）
    parameters: Dict[str, str] = field(default_factory=dict)  # 关键参数及说明
    output_files: List[str] = field(default_factory=list)  # 生成的输出文件
    output_format: str = ""          # 输出文件格式说明
    run_command: str = ""            # 典型调用命令示例
    notes: str = ""                  # 额外说明
    source_doc: str = ""             # 来源文档路径或摘要

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, registry_dir: Optional[str] = None) -> str:
        """将 ToolProfile 保存为 JSON 到工具注册目录。"""
        if registry_dir is None:
            registry_dir = str(Path(__file__).parent.parent / "seismo_tools" / "registry")
        Path(registry_dir).mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", self.name.lower())
        out_path = os.path.join(registry_dir, f"{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return out_path

    @classmethod
    def load(cls, json_path: str) -> "ToolProfile":
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        lines = [
            f"工具: {self.name}",
            f"描述: {self.description}",
            f"程序: {self.executable}",
            f"输入文件: {', '.join(self.input_files) or '未知'}",
            f"输出文件: {', '.join(self.output_files) or '未知'}",
        ]
        if self.run_command:
            lines.append(f"调用示例: {self.run_command}")
        if self.parameters:
            lines.append("主要参数:")
            for k, v in list(self.parameters.items())[:8]:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_PARSE_SYSTEM_PROMPT = """你是一位专业的地震学软件工程师。
用户会提供一段地震学数据处理工具（如 HypoDD、VELEST、NonLinLoc 等）的文档或 README。
你的任务是从文档中提取工具的接口信息，并以严格的 JSON 格式输出。

输出 JSON 必须包含以下字段（所有字段均为字符串或字符串列表）:
{
  "name": "工具名称",
  "executable": "可执行程序名（如 hypoDD、velest 等）",
  "description": "功能简介（2-3句话）",
  "input_files": ["file1.inp", "file2.dat", ...],
  "input_format": "主要输入文件格式的详细说明",
  "input_template": "主要输入文件的内容模板（用 {{PLACEHOLDER}} 标注可替换部分）",
  "parameters": {"参数名": "说明", ...},
  "output_files": ["hypoDD.reloc", "hypoDD.log", ...],
  "output_format": "输出文件格式说明",
  "run_command": "典型调用命令",
  "notes": "其他注意事项"
}

若某字段在文档中找不到，填入空字符串或空列表。
只输出 JSON，不要有任何其他内容。"""


def _call_llm_for_parse(doc_text: str, llm_config: Dict) -> str:
    """Call LLM to parse documentation."""
    import urllib.request
    import urllib.error

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    # Truncate very long docs to ~6000 chars to fit context window
    doc_excerpt = doc_text[:6000] + ("\n...[truncated]" if len(doc_text) > 6000 else "")

    messages = [
        {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"请解析以下工具文档：\n\n{doc_excerpt}"},
    ]

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.1, "num_predict": 3000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 3000}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"}
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if provider == "ollama":
        return body.get("message", {}).get("content", "{}")
    else:
        return body.get("choices", [{}])[0].get("message", {}).get("content", "{}")


def _parse_json_from_response(text: str) -> Dict:
    """Extract JSON from LLM response, handling code fences."""
    # Strip code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


class DocParser:
    """Parse tool documentation and extract structured ToolProfile."""

    def __init__(self, llm_config: Optional[Dict] = None):
        if llm_config is None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config_manager import get_config_manager
                llm_config = get_config_manager().get_llm_config()
            except Exception:
                llm_config = {"provider": "ollama", "model": "qwen2.5:7b",
                              "api_base": "http://localhost:11434"}
        self.llm_config = llm_config

    def parse_text(self, doc_text: str, auto_save: bool = True) -> ToolProfile:
        """
        Parse a documentation string and return a ToolProfile.

        Parameters
        ----------
        doc_text : str
            Documentation text (README, man page, etc.)
        auto_save : bool
            Automatically save the profile to the registry. Default True.

        Returns
        -------
        ToolProfile
        """
        raw = _call_llm_for_parse(doc_text, self.llm_config)
        data = _parse_json_from_response(raw)

        profile = ToolProfile(
            name=data.get("name", "UnknownTool"),
            executable=data.get("executable", ""),
            description=data.get("description", ""),
            input_files=data.get("input_files", []),
            input_format=data.get("input_format", ""),
            input_template=data.get("input_template", ""),
            parameters=data.get("parameters", {}),
            output_files=data.get("output_files", []),
            output_format=data.get("output_format", ""),
            run_command=data.get("run_command", ""),
            notes=data.get("notes", ""),
            source_doc=doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
        )
        if auto_save:
            profile.save()
        return profile

    def parse_file(self, file_path: str, auto_save: bool = True) -> ToolProfile:
        """
        Read a local documentation file and parse it.

        Supported: .txt, .md, .rst, .pdf (requires PyMuPDF or pdfminer)
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Documentation file not found: {file_path}")

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(str(p))
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()

        profile = self.parse_text(text, auto_save=auto_save)
        profile.source_doc = str(p)
        if auto_save:
            profile.save()
        return profile

    @staticmethod
    def _read_pdf(path: str) -> str:
        """Extract text from PDF. Tries PyMuPDF then pdfminer."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            pages = [page.get_text() for page in doc]
            return "\n".join(pages)
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            raise ImportError(
                "无法读取 PDF：请安装 PyMuPDF (pip install pymupdf) 或 "
                "pdfminer.six (pip install pdfminer.six)"
            )
