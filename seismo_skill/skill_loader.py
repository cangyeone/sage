"""
skill_loader.py — seismo_skill 技能引擎 v2

支持三种技能格式：
  A. 单文件技能：skills/<name>.md
  B. 文件夹技能：skills/<name>/SKILL.md + agents/*.yaml + references/scripts/assets
  C. OpenAI/Codex skills repo：<repo>/skills/<name>/SKILL.md（可一次导入多个技能）

从三个目录加载技能：
  1. seismo_skill/skills/         内置技能（随项目发布）
  2. seismo_skill/user_skills/    项目内用户自定义技能（同名时覆盖内置）
  3. .seismicx/skills/            project-local imported skills

v2 新特性
---------
- 文件夹技能完整解析：SKILL.md + 目录内 references/scripts/assets + agents/
- OpenAI/Codex skill repo 兼容：识别 skills/<name>/SKILL.md、大小写 skill.md
- 辅助资料支持 Markdown/RST/TXT/YAML/JSON/Python/Shell 等文本资源
- folder references 按查询语义选择性注入（不超预算，不乱注全部）
- agents/*.yaml 配置解析（display_name / default_prompt / short_description）
- YAML 多行字段（>- 折叠块、| 原始块）正确解析（PyYAML 优先）
- description 字段在列表 / 搜索 / context 中完整使用
- install_skill_from_dir()：从本地目录安装文件夹技能
- list_skills() 返回 description 字段
- search_skills() 在 description 中也做检索

公开接口
--------
list_skills()                → list[dict]
search_skills(query, k)      → list[dict]
load_skill(name)             → str
save_user_skill(name, text)  → Path
delete_user_skill(name)      → bool
install_skill_from_dir(src)  → dict
build_skill_context(query)   → str
build_skill_context_with_rag(query) → (str, str)
get_skill_detail(name)       → dict | None
get_user_skill_dir()         → Path
invalidate_cache()
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from sage_paths import sage_home

# ── 目录定义 ─────────────────────────────────────────────────────────────────

_BUILTIN_SKILL_DIR = Path(__file__).parent / "skills"
_PROJECT_USER_SKILL_DIR = Path(__file__).parent / "user_skills"
_LEGACY_USER_SKILL_DIR = sage_home("skills")

_SKILL_MATCH_CACHE: Dict[Tuple[str, int, Tuple[str, ...]], List[str]] = {}


def _expand_query(query: str) -> str:
    """Compatibility hook: query expansion now comes from SKILL metadata."""
    return str(query or "")


def get_user_skill_dir() -> Path:
    """返回项目内用户自定义技能目录（seismo_skill/user_skills/）。"""
    d = _PROJECT_USER_SKILL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_legacy_user_skill_dir() -> Path:
    """返回旧版全局技能目录，仅用于兼容读取/删除旧数据。"""
    return _LEGACY_USER_SKILL_DIR


# ── YAML / Frontmatter 解析 ───────────────────────────────────────────────────

def _parse_yaml_frontmatter(raw: str) -> dict:
    """
    解析 YAML frontmatter 字符串。
    优先使用 PyYAML；不可用时退回手写解析器（支持 >- 折叠块）。
    """
    # ── 尝试 PyYAML ──────────────────────────────────────────────────────────
    try:
        import yaml  # type: ignore
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # ── 手写解析器（兼容 >- 和 | 块）────────────────────────────────────────
    meta: dict = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # 跳过空行和注释
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue

        if ":" not in line:
            i += 1
            continue

        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        i += 1

        # YAML 列表（下一行缩进以 - 开头）
        if rest == "" and i < len(lines) and lines[i].lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            meta[key] = items
            continue

        # 块标量 >- 或 >（折叠）/ | 或 |- （保留换行）
        if rest in (">-", ">", "|", "|-"):
            fold = rest in (">-", ">")
            collected: List[str] = []
            # 推算缩进基准：取下一个非空行的缩进长度
            indent = None
            while i < len(lines):
                ln = lines[i]
                stripped = ln.lstrip()
                if not stripped:          # 空行：折叠块保留一个换行
                    if not fold:
                        collected.append("")
                    i += 1
                    continue
                cur_indent = len(ln) - len(stripped)
                if indent is None:
                    indent = cur_indent
                if cur_indent < indent:   # 退出块
                    break
                collected.append(stripped)
                i += 1
            if fold:
                meta[key] = " ".join(collected)
            else:
                meta[key] = "\n".join(collected)
            continue

        # 普通单行值（去掉首尾引号）
        meta[key] = rest.strip('"').strip("'")

    return meta


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """
    解析 Markdown YAML 前置元数据（--- ... --- 块）。
    返回 (meta_dict, body_text)。
    """
    meta: dict = {}
    body = text

    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return meta, body

    frontmatter_raw, body = m.group(1), m.group(2)
    meta = _parse_yaml_frontmatter(frontmatter_raw)

    # ── 规范化列表字段 ─────────────────────────────────────────────────────
    for list_field in ("keywords", "rag_sources", "related_skills"):
        v = meta.get(list_field)
        if v is None:
            meta[list_field] = []
        elif isinstance(v, str):
            meta[list_field] = [s.strip() for s in v.split(",") if s.strip()]
        elif not isinstance(v, list):
            meta[list_field] = []

    # ── 规范化 description（折叠空白，便于单行展示）─────────────────────────
    if "description" in meta and isinstance(meta["description"], str):
        meta["description"] = " ".join(meta["description"].split())

    return meta, body


# ── agents/*.yaml 解析 ────────────────────────────────────────────────────────

def _load_agent_config(folder: Path) -> dict:
    """
    解析 agents/ 目录下第一个 .yaml / .yml 文件的接口配置。

    返回 dict（键：display_name, short_description, default_prompt, model_hints...）
    如果文件不存在或解析失败，返回空 dict。
    """
    agents_dir = folder / "agents"
    if not agents_dir.is_dir():
        return {}

    yaml_files = sorted(agents_dir.glob("*.yaml")) + sorted(agents_dir.glob("*.yml"))
    if not yaml_files:
        return {}

    try:
        raw = yaml_files[0].read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
            parsed = yaml.safe_load(raw)
        except Exception:
            parsed = _parse_yaml_frontmatter(raw)

        if not isinstance(parsed, dict):
            return {}

        # 扁平化 interface 子节点
        iface = parsed.get("interface", parsed)
        return {
            "display_name":    iface.get("display_name", ""),
            "short_description": iface.get("short_description", ""),
            "default_prompt":  iface.get("default_prompt", ""),
            "model_hints":     iface.get("model_hints", []),
            "agent_file":      str(yaml_files[0]),
        }
    except Exception:
        return {}


# ── 文件夹技能资料加载 ────────────────────────────────────────────────────────

_IGNORED_REF_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}

_SKILL_ENTRY_FILENAMES = ("SKILL.md", "skill.md", "Skill.md")

_TEXT_REFERENCE_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
}

_REFERENCE_MAX_BYTES = 256_000
_REFERENCE_MAX_CHARS = 80_000


def _is_hidden_or_ignored(path: Path, root: Path) -> bool:
    """判断 path 是否位于隐藏/缓存目录中。"""
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part.startswith(".") or part in _IGNORED_REF_DIRS for part in rel_parts)


def _reference_key(md_file: Path, folder: Path) -> str:
    """
    将 folder 内的 Markdown 文件转为稳定 key。

    examples:
      README.md -> README
      references/api.md -> references/api
      subskill/SKILL.md -> subskill/SKILL
    """
    rel = md_file.relative_to(folder).with_suffix("")
    return rel.as_posix()


def _find_skill_entry_file(folder: Path) -> Optional[Path]:
    """
    返回文件夹技能入口文件。

    OpenAI/Codex skill repo 约定使用 SKILL.md；这里同时兼容 skill.md/Skill.md，
    便于导入来自不同系统或人工整理的技能目录。
    """
    try:
        children = list(folder.iterdir())
    except Exception:
        return None
    for filename in _SKILL_ENTRY_FILENAMES:
        for candidate in children:
            if candidate.is_file() and candidate.name == filename:
                return candidate
    try:
        for candidate in folder.iterdir():
            if candidate.is_file() and candidate.name.lower() == "skill.md":
                return candidate
    except Exception:
        return None
    return None


def _as_list(value) -> List[str]:
    """把 YAML 字段规整为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [str(v).strip() for v in value.values() if str(v).strip()]
    return [s.strip() for s in re.split(r"[,，;\n]+", str(value)) if s.strip()]


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return ""


def _first_paragraph(text: str, max_chars: int = 260) -> str:
    """从正文推断短描述，避免没有 frontmatter 的技能完全不可检索。"""
    paragraphs: List[str] = []
    current: List[str] = []
    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith("#"):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith(("-", "*", "|", ">")):
            continue
        current.append(line)
        if len(" ".join(current)) >= max_chars:
            break
    if current:
        paragraphs.append(" ".join(current))
    if not paragraphs:
        return ""
    return " ".join(paragraphs[0].split())[:max_chars]


def _normalize_skill_meta(meta: dict, body: str, fallback_name: str, source: str) -> dict:
    """
    兼容 OpenAI/Codex skills repo 与 SAGE 旧格式的元数据。

    OpenAI skill 的触发主要依赖 name/description；SAGE 还会使用 category、
    keywords、related_skills、rag_sources 等字段。这里把 tags/aliases/triggers 等
    常见字段也并入关键词，减少技能“文不对题”的概率。
    """
    display_name = (
        str(meta.get("display_name") or meta.get("title") or _first_heading(body) or fallback_name).strip()
    )
    name = str(meta.get("name") or meta.get("id") or fallback_name).strip() or fallback_name
    description = str(meta.get("description") or meta.get("summary") or _first_paragraph(body)).strip()
    description = " ".join(description.split())
    category = str(meta.get("category") or meta.get("domain") or ("custom" if source == "user" else "")).strip()

    keywords: List[str] = []
    for key in (
        "keywords",
        "tags",
        "aliases",
        "triggers",
        "trigger_phrases",
        "use_cases",
        "when_to_use",
        "prefer_when",
    ):
        keywords.extend(_as_list(meta.get(key)))
    if display_name and display_name != name:
        keywords.append(display_name)

    return {
        "name": name,
        "display_name": display_name,
        "description": description,
        "category": category,
        "keywords": keywords,
        "rag_sources": _as_list(meta.get("rag_sources")),
        "related_skills": _as_list(meta.get("related_skills") or meta.get("related")),
        "prefer_when": _as_list(meta.get("prefer_when")),
        "avoid_when": _as_list(meta.get("avoid_when") or meta.get("do_not_use_when")),
        "workflow": meta.get("workflow", ""),
        "generated_from": meta.get("generated_from", ""),
        "source": meta.get("source", source),
    }


def _extract_markdown_headings(text: str, limit: int = 12) -> List[str]:
    """提取少量 Markdown/RST 标题，用于 manifest 和关键词推断。"""
    headings: List[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        m = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", line)
        heading = m.group(1).strip() if m else ""
        if not heading and idx + 1 < len(lines):
            marker = lines[idx + 1].strip()
            candidate = line.strip()
            if candidate and len(marker) >= max(3, min(len(candidate), 12)) and set(marker) <= set("=-~^\"'`"):
                heading = candidate
        if heading:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings

def _load_references(folder: Path, entry_file: Optional[Path] = None) -> Dict[str, str]:
    """
    递归加载文件夹技能目录内的辅助文本文件。

    约定：
      - skills/<name>/SKILL.md 是该文件夹技能的主入口，不作为 reference。
      - OpenAI/Codex skill 的 references/scripts/assets 中，文本型资料会进入
        references，便于检索和按需注入上下文。
      - 二进制资源不进全文 references，但会进入 resource_manifest。
      - 不把嵌套目录里的 SKILL.md 注册成独立顶层技能；它只是父技能的一份资料。

    返回 {ref_name: content_str}，文件读取失败则跳过。
    """
    refs: Dict[str, str] = {}
    entry_resolved = entry_file.resolve() if entry_file else None
    for ref_file in sorted(folder.rglob("*")):
        if not ref_file.is_file():
            continue
        if entry_resolved and ref_file.resolve() == entry_resolved:
            continue
        if entry_file and ref_file.parent == entry_file.parent and ref_file.name.lower() == "skill.md":
            continue
        if _is_hidden_or_ignored(ref_file, folder):
            continue
        if ref_file.suffix.lower() not in _TEXT_REFERENCE_EXTS:
            continue
        try:
            if ref_file.stat().st_size > _REFERENCE_MAX_BYTES:
                continue
        except Exception:
            continue
        try:
            content = ref_file.read_text(encoding="utf-8", errors="replace")
            if len(content) > _REFERENCE_MAX_CHARS:
                content = content[:_REFERENCE_MAX_CHARS] + "\n\n…（reference 已截断）\n"
            refs[_reference_key(ref_file, folder)] = content
        except Exception:
            continue
    return refs


def _build_reference_manifest(refs: Dict[str, str]) -> List[Dict[str, Union[str, List[str]]]]:
    """为列表展示和 LLM context 构建轻量 manifest，不放全文。"""
    manifest: List[Dict[str, Union[str, List[str]]]] = []
    for ref_name, content in refs.items():
        manifest.append({
            "name": ref_name,
            "headings": _extract_markdown_headings(content, limit=6),
        })
    return manifest


def _load_resource_manifest(folder: Path, entry_file: Optional[Path] = None) -> List[Dict[str, Union[str, int]]]:
    """
    列出文件夹技能携带的资源。

    这不会把二进制资源读进上下文，只让 UI/Agent 知道该技能有 scripts/assets/
    这类可用材料。OpenAI skills repo 经常依赖这些 bundled resources。
    """
    resources: List[Dict[str, Union[str, int]]] = []
    entry_resolved = entry_file.resolve() if entry_file else None
    for file in sorted(folder.rglob("*")):
        if not file.is_file():
            continue
        if entry_resolved and file.resolve() == entry_resolved:
            continue
        if entry_file and file.parent == entry_file.parent and file.name.lower() == "skill.md":
            continue
        if _is_hidden_or_ignored(file, folder):
            continue
        try:
            rel = file.relative_to(folder).as_posix()
            parts = file.relative_to(folder).parts
            kind = "reference"
            if parts and parts[0] in {"scripts", "bin"}:
                kind = "script"
            elif parts and parts[0] in {"assets", "templates", "examples"}:
                kind = "asset"
            resources.append({
                "path": rel,
                "kind": kind,
                "suffix": file.suffix.lower(),
                "size": file.stat().st_size,
            })
        except Exception:
            continue
    return resources


def _infer_keywords(
    name: str,
    meta_keywords: List[str],
    description: str,
    body: str,
    refs: Dict[str, str],
) -> List[str]:
    """
    用文件名、分类描述、标题和资料名补齐关键词。

    这解决了文件夹技能 frontmatter 比较薄时“搜不到/搜偏”的问题，但仍把
    显式 keywords 放在最前面，保持作者意图优先。
    """
    keywords: List[str] = []

    def add(value: str) -> None:
        value = " ".join(str(value).strip().split())
        if value and value.lower() not in {k.lower() for k in keywords}:
            keywords.append(value)

    for kw in meta_keywords:
        add(kw)
    add(name)
    add(name.replace("-", " "))

    sample = "\n".join([
        description,
        "\n".join(_extract_markdown_headings(body, limit=16)),
        "\n".join(refs.keys()),
    ])
    for tok in _tokenize(sample):
        if _tok_min_len(tok):
            add(tok)
        if len(keywords) >= 80:
            break
    return keywords


# ── 文件夹技能加载 ────────────────────────────────────────────────────────────

def _load_folder_skill(folder: Path, source: str) -> Optional[Dict]:
    """
    从文件夹技能目录加载一个完整的技能条目。
    文件夹必须包含 SKILL.md / skill.md；references/scripts/assets 和 agents/ 为可选。

    返回技能 dict，或 None（SKILL.md 不存在 / 读取失败）。
    """
    skill_md = _find_skill_entry_file(folder)
    if skill_md is None:
        return None

    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return None

    meta, body = _parse_frontmatter(text)
    norm = _normalize_skill_meta(meta, body, folder.name, source)
    effective_source = norm.get("source", source)
    refs = _load_references(folder, entry_file=skill_md)
    description = norm.get("description", "")
    keywords = _infer_keywords(
        name=norm["name"],
        meta_keywords=norm.get("keywords", []),
        description=description,
        body=body,
        refs=refs,
    )

    return {
        "name":            norm["name"],
        "display_name":    norm.get("display_name", norm["name"]),
        "description":     description,
        "category":        norm.get("category", ""),
        "keywords":        keywords,
        "path":            str(skill_md),
        "folder":          str(folder),
        "body":            body.strip(),
        "full_text":       text.strip(),
        "is_folder":       True,
        "references":      refs,
        "reference_manifest": _build_reference_manifest(refs),
        "resource_manifest": _load_resource_manifest(folder, entry_file=skill_md),
        "agent_config":    _load_agent_config(folder),
        "source":          effective_source,
        "filename":        skill_md.name,
        "format":          "openai_folder_skill",
        "rag_sources":     norm.get("rag_sources", []),
        "generated_from":  norm.get("generated_from", ""),
            "related_skills":  norm.get("related_skills", []),
            "prefer_when":     norm.get("prefer_when", []),
            "avoid_when":      norm.get("avoid_when", []),
            "workflow":        norm.get("workflow", ""),
        }


def _discover_skill_folders(directory: Path) -> List[Path]:
    """
    在技能根目录中发现文件夹型技能。

    支持：
      - <root>/<skill>/SKILL.md
      - <root>/skills/<skill>/SKILL.md
      - <root>/<repo>/skills/<skill>/SKILL.md

    最后一种用于兼容直接放入 OpenAI/Codex skills repo 的场景。不会把已有
    文件夹技能内部的嵌套 SKILL.md 注册成顶层技能，避免父子技能拆乱。
    """
    folders: List[Path] = []
    seen: Set[str] = set()

    def add(folder: Path) -> None:
        try:
            resolved = str(folder.resolve())
        except Exception:
            resolved = str(folder)
        if resolved in seen:
            return
        if _find_skill_entry_file(folder) is None:
            return
        seen.add(resolved)
        folders.append(folder)

    def add_children(container: Path) -> None:
        if not container.is_dir():
            return
        for child in sorted(container.iterdir()):
            if child.is_dir() and not _is_hidden_or_ignored(child, directory):
                add(child)

    add_children(directory)
    add_children(directory / "skills")

    try:
        for repo in sorted(directory.iterdir()):
            if not repo.is_dir() or _find_skill_entry_file(repo) is not None:
                continue
            if _is_hidden_or_ignored(repo, directory):
                continue
            add_children(repo / "skills")
    except Exception:
        pass

    return folders


# ── 目录扫描 ──────────────────────────────────────────────────────────────────

def _load_from_dir(directory: Path, source: str) -> List[Dict]:
    """
    扫描目录，同时处理：
      - 单文件技能：*.md
      - 文件夹技能：<subdir>/SKILL.md

    文件夹技能优先（同名时文件夹版本覆盖单文件版本）。
    """
    if not directory.exists():
        return []

    skills: Dict[str, Dict] = {}  # name -> entry（用 dict 去重）

    # ── 单文件技能 ────────────────────────────────────────────────────────────
    for md_file in sorted(directory.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        meta, body = _parse_frontmatter(text)
        norm = _normalize_skill_meta(meta, body, md_file.stem, source)
        effective_source = norm.get("source", source)
        name = norm["name"]
        entry: Dict = {
            "name":           name,
            "display_name":   norm.get("display_name", name),
            "description":    norm.get("description", ""),
            "category":       norm.get("category", ""),
            "keywords":       _infer_keywords(
                name=name,
                meta_keywords=norm.get("keywords", []),
                description=norm.get("description", ""),
                body=body,
                refs={},
            ),
            "path":           str(md_file),
            "folder":         "",
            "body":           body.strip(),
            "full_text":      text.strip(),
            "is_folder":      False,
            "references":     {},
            "reference_manifest": [],
            "resource_manifest": [],
            "agent_config":   {},
            "source":         effective_source,
            "filename":       md_file.name,
            "format":         "single_markdown_skill",
            "rag_sources":    norm.get("rag_sources", []),
            "generated_from": norm.get("generated_from", ""),
            "related_skills": norm.get("related_skills", []),
            "prefer_when":    norm.get("prefer_when", []),
            "avoid_when":     norm.get("avoid_when", []),
            "workflow":       norm.get("workflow", ""),
        }
        # 只在该名称还没被文件夹技能占用时才添加（文件夹优先）
        if name not in skills or not skills[name].get("is_folder"):
            skills[name] = entry

    # ── 文件夹技能 ────────────────────────────────────────────────────────────
    for skill_folder in _discover_skill_folders(directory):
        entry = _load_folder_skill(skill_folder, source)
        if entry is None:
            continue
        # 文件夹技能覆盖同名单文件技能
        skills[entry["name"]] = entry

    return list(skills.values())


# ── 全量加载 / 缓存 ───────────────────────────────────────────────────────────

def _load_all_skills() -> List[Dict]:
    """
    加载内置技能 + 项目内用户技能 + 旧版全局用户技能。
    同名技能优先级：项目内用户技能 > 旧版全局用户技能 > 内置技能。
    """
    builtin = _load_from_dir(_BUILTIN_SKILL_DIR, "builtin")
    legacy  = _load_from_dir(get_legacy_user_skill_dir(), "user") if get_legacy_user_skill_dir().exists() else []
    user    = _load_from_dir(get_user_skill_dir(), "user")

    user_names = {s["name"] for s in user}
    legacy_names = {s["name"] for s in legacy}
    merged = (
        [s for s in builtin if s["name"] not in legacy_names and s["name"] not in user_names]
        + [s for s in legacy if s["name"] not in user_names]
        + user
    )
    return merged


_SKILLS_CACHE: Optional[List[Dict]] = None


def _get_skills() -> List[Dict]:
    global _SKILLS_CACHE
    if _SKILLS_CACHE is None:
        _SKILLS_CACHE = _load_all_skills()
    return _SKILLS_CACHE


def invalidate_cache():
    """清除缓存（技能文件有更新时调用）。"""
    global _SKILLS_CACHE
    _SKILLS_CACHE = None
    _SKILL_MATCH_CACHE.clear()


# ── 公开 CRUD 接口 ────────────────────────────────────────────────────────────

def list_skills() -> List[Dict]:
    """
    列出所有可用技能的元信息（不含完整文档体和 references 内容）。

    返回 list[dict]，每条包含：
      name, description, category, keywords, source, is_folder,
      filename, rag_sources, related_skills, workflow,
      agent_config（仅含 display_name, short_description, default_prompt）
    """
    result = []
    for s in _get_skills():
        agent = s.get("agent_config", {})
        result.append({
            "name":             s["name"],
            "display_name":     s.get("display_name", s["name"]),
            "description":      s.get("description", ""),
            "category":         s["category"],
            "keywords":         s["keywords"],
            "source":           s["source"],
            "is_folder":        s.get("is_folder", False),
            "filename":         s["filename"],
            "format":           s.get("format", ""),
            "rag_sources":      s.get("rag_sources", []),
            "generated_from":   s.get("generated_from", ""),
            "related_skills":   s.get("related_skills", []),
            "prefer_when":      s.get("prefer_when", []),
            "avoid_when":       s.get("avoid_when", []),
            "workflow":         s.get("workflow", ""),
            "ref_names":        list(s.get("references", {}).keys()),
            "ref_count":        len(s.get("references", {})),
            "reference_manifest": s.get("reference_manifest", []),
            "resource_manifest": s.get("resource_manifest", []),
            "agent_config": {
                "display_name":      agent.get("display_name", ""),
                "short_description": agent.get("short_description", ""),
                "default_prompt":    agent.get("default_prompt", ""),
            } if agent else {},
        })
    return result


def load_skill(name: str) -> str:
    """按技能名称加载完整 SKILL.md 文本（不含 references）。"""
    for skill in _get_skills():
        if skill["name"] == name:
            return skill["full_text"]
    return ""


def get_skill_detail(name: str) -> Optional[Dict]:
    """
    返回技能完整条目（含 body / full_text / references / agent_config / source）。
    references 内容以 {ref_name: content} 形式返回。
    """
    for skill in _get_skills():
        if skill["name"] == name:
            return {**skill}
    return None


def save_user_skill(name: str, text: str) -> Path:
    """
    保存用户自定义单文件技能到 seismo_skill/user_skills/<name>.md。
    文件夹技能请用 install_skill_from_dir()。
    """
    skill_dir = get_user_skill_dir()
    safe_name = re.sub(r"[^\w\-]", "_", name)
    target = skill_dir / f"{safe_name}.md"
    target.write_text(text, encoding="utf-8")
    invalidate_cache()
    return target


def delete_user_skill(name: str) -> bool:
    """
    删除用户自定义技能（user / generated；内置技能不可删除）。

    文件夹技能和单文件技能均支持删除。
    """
    for skill in _get_skills():
        if skill["name"] == name and skill["source"] in ("user", "generated"):
            p = Path(skill["path"])
            if skill.get("is_folder") and skill.get("folder"):
                shutil.rmtree(Path(skill["folder"]), ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            invalidate_cache()
            return True
    return False


def _install_one_skill_folder(source_dir: Path, overwrite: bool = True) -> Dict:
    """安装一个已确认包含 SKILL.md/skill.md 的文件夹技能。"""
    skill_md = _find_skill_entry_file(source_dir)
    if skill_md is None:
        raise ValueError(f"目录中未找到 SKILL.md / skill.md：{source_dir}")

    text = skill_md.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    norm = _normalize_skill_meta(meta, body, source_dir.name, "user")
    name = norm["name"].strip() or source_dir.name
    safe_name = re.sub(r"[^\w\-]", "_", name)

    target_dir = get_user_skill_dir() / safe_name
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"技能目录已存在（overwrite=False）：{target_dir}")
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)
    entry = _load_folder_skill(target_dir, "user")
    if entry is None:
        raise RuntimeError(f"安装后无法重新加载技能：{target_dir}")
    return entry


def _discover_installable_skill_folders(source_dir: Path) -> List[Path]:
    """发现可安装的单个技能或 repo 内的多个 OpenAI/Codex 技能。"""
    if _find_skill_entry_file(source_dir) is not None:
        return [source_dir]
    folders = _discover_skill_folders(source_dir)
    if not folders:
        # 兼容直接传入 repo/skills 目录。
        folders = _discover_skill_folders(source_dir / "skills")
    return folders


def install_skills_from_dir(source_dir: Union[str, Path], overwrite: bool = True) -> List[Dict]:
    """
    从本地目录安装一个或多个文件夹技能。

    source_dir 可以是：
      - 单个技能目录：<skill>/SKILL.md
      - OpenAI/Codex skills repo：<repo>/skills/<skill>/SKILL.md
      - 技能根目录：其中包含多个 <skill>/SKILL.md
    """
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"目录不存在：{source_dir}")

    folders = _discover_installable_skill_folders(source_dir)
    if not folders:
        raise ValueError(f"目录中未找到可安装的 SKILL.md / skill.md：{source_dir}")

    entries: List[Dict] = []
    for folder in folders:
        entries.append(_install_one_skill_folder(folder, overwrite=overwrite))
    invalidate_cache()
    return entries


def install_skill_from_dir(source_dir: Union[str, Path], overwrite: bool = True) -> Dict:
    """
    从本地目录安装文件夹技能到 seismo_skill/user_skills/<name>/。

    source_dir 可以包含 SKILL.md / skill.md；若传入 OpenAI/Codex skills repo，
    建议使用 install_skills_from_dir() 批量安装。
    name 从 SKILL.md frontmatter 的 name 字段读取；不存在则用目录名。

    参数
    ----
    source_dir : str | Path  — 源目录路径
    overwrite  : bool        — True 表示同名技能直接覆盖（默认）

    返回
    ----
    dict — 安装后的技能条目（同 get_skill_detail 返回格式）

    异常
    ----
    FileNotFoundError  — source_dir 不存在
    ValueError         — source_dir 中不含 SKILL.md / skill.md
    FileExistsError    — overwrite=False 且目标目录已存在
    """
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"目录不存在：{source_dir}")

    if _find_skill_entry_file(source_dir) is None:
        folders = _discover_installable_skill_folders(source_dir)
        if len(folders) > 1:
            raise ValueError(
                f"目录包含 {len(folders)} 个技能；请使用 install_skills_from_dir() 批量安装：{source_dir}"
            )
        if not folders:
            raise ValueError(f"目录中未找到 SKILL.md / skill.md：{source_dir}")
        source_dir = folders[0]

    entry = _install_one_skill_folder(source_dir, overwrite=overwrite)
    invalidate_cache()
    return entry


# ── 分词 / 搜索 ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """混合分词：英文按空格/标点，中文按单字及 bigram。"""
    tokens: List[str] = []
    for tok in re.findall(r"[a-zA-Z0-9_\-\.]+", text):
        tokens.append(tok.lower())
    chinese_chars = re.findall(r"[一-鿿]", text)
    tokens.extend(chinese_chars)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


def _tok_min_len(tok: str) -> bool:
    """
    返回 token 是否达到最低匹配长度：
    - 中文 bigram（2个汉字）：len >= 2
    - 英文 / 数字：len >= 3
    """
    is_cjk = all("一" <= c <= "鿿" for c in tok) and len(tok) >= 1
    return len(tok) >= (2 if is_cjk else 3)


def _skill_profile_text(skill: Dict, body_chars: int = 2200) -> str:
    """Build the text used for generic skill retrieval."""
    parts: List[str] = [
        str(skill.get("name", "")),
        str(skill.get("display_name", "")),
        str(skill.get("description", "")),
        str(skill.get("category", "")),
        " ".join(str(x) for x in skill.get("keywords", []) or []),
        " ".join(str(x) for x in skill.get("related_skills", []) or []),
    ]
    agent_config = skill.get("agent_config") or {}
    if isinstance(agent_config, dict):
        parts.extend(str(agent_config.get(k, "")) for k in ("display_name", "short_description", "default_prompt"))
    for ref in skill.get("reference_manifest", [])[:12] or []:
        if isinstance(ref, dict):
            parts.append(str(ref.get("name", "")))
            parts.append(" ".join(str(h) for h in ref.get("headings", [])[:8]))
    for resource in skill.get("resource_manifest", [])[:20] or []:
        if isinstance(resource, dict):
            parts.append(str(resource.get("path", "")))
            parts.append(str(resource.get("kind", "")))
    body = str(skill.get("body", ""))
    if body:
        parts.append(body[:body_chars])
    return "\n".join(p for p in parts if p).strip()


def _metadata_score(query: str, skill: Dict) -> float:
    """Generic metadata overlap score. Domain intent lives in SKILL keywords, not code rules."""
    query_text = _expand_query(query)
    query_tokens = set(_tokenize(query_text))
    query_lower = query_text.lower()
    profile = _skill_profile_text(skill).lower()
    score = 0.0

    for kw in skill.get("keywords", []) or []:
        kw_lower = str(kw).lower()
        if kw_lower and kw_lower in query_lower:
            score += 6.0
        elif any(_tok_min_len(tok) and tok in kw_lower for tok in query_tokens):
            score += 3.0

    for phrase in skill.get("prefer_when", []) or []:
        phrase_lower = str(phrase).strip().lower()
        if phrase_lower and phrase_lower in query_lower:
            score += 8.0

    for phrase in skill.get("avoid_when", []) or []:
        phrase_lower = str(phrase).strip().lower()
        if phrase_lower and phrase_lower in query_lower:
            score -= 12.0

    name_lower = str(skill.get("name", "")).lower()
    if name_lower and name_lower in query_lower:
        score += 5.0
    if any(_tok_min_len(tok) and tok in name_lower for tok in query_tokens):
        score += 3.0

    matched = 0
    for tok in query_tokens:
        if _tok_min_len(tok) and tok in profile:
            matched += 1
    score += min(8.0, matched * 1.2)

    if score > 0 and skill.get("source") == "user":
        score += 1.0
    return score


def _embedding_rank_skills(query: str, skills: List[Dict], max_candidates: int) -> List[Tuple[float, Dict]]:
    """Use the local embedding backend when available to retrieve semantic skill candidates."""
    if not query.strip() or not skills:
        return []
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from rag_backends import EmbeddingModel  # type: ignore
        texts = [_skill_profile_text(skill, body_chars=3200) for skill in skills]
        vecs = EmbeddingModel.get().encode([query] + texts, batch_size=16)
        query_vec = vecs[0]
        skill_vecs = vecs[1:]
        scores = skill_vecs @ query_vec
        ranked = sorted(
            ((float(score), skill) for score, skill in zip(scores, skills)),
            key=lambda item: item[0],
            reverse=True,
        )
        return ranked[:max_candidates]
    except Exception:
        return []


def _candidate_skills(query: str, max_candidates: int = 12) -> List[Dict]:
    skills = _get_skills()
    if not skills:
        return []

    embedding_scores: Dict[str, float] = {}
    for score, skill in _embedding_rank_skills(query, skills, max_candidates=max_candidates * 2):
        embedding_scores[str(skill.get("name", ""))] = max(0.0, float(score))

    ranked: List[Tuple[float, Dict]] = []
    for skill in skills:
        meta_score = _metadata_score(query, skill)
        emb_score = embedding_scores.get(str(skill.get("name", "")), 0.0)
        combined = meta_score * 2.0 + emb_score * 6.0
        if combined > 0:
            ranked.append((combined, skill))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [skill for _, skill in ranked[:max_candidates]]


def _parse_llm_skill_matches(raw: str, candidates: List[Dict]) -> List[str]:
    allowed = {str(skill.get("name", "")): skill for skill in candidates}
    chosen: List[str] = []
    if re.search(r"(?im)^\s*(NO_SKILL|NONE|无匹配|不需要技能)\s*$", str(raw or "")):
        return ["__NO_SKILL__"]
    for line in str(raw or "").splitlines():
        text = line.strip().strip("-*0123456789. ")
        if not text:
            continue
        if ":" in text or "：" in text:
            text = re.split(r"[:：]", text, maxsplit=1)[1].strip()
        for name in allowed:
            if text == name or text.startswith(name) or f"`{name}`" in text:
                if name not in chosen:
                    chosen.append(name)
                break
    return chosen


def _llm_match_skills(query: str, candidates: List[Dict], top_k: int) -> List[str]:
    """Ask the active LLM to rerank candidates with a raw-text protocol."""
    if os.environ.get("SEISMICX_SKILL_LLM_MATCH", "1") == "0":
        return []
    if not query.strip() or not candidates:
        return []
    names = tuple(str(skill.get("name", "")) for skill in candidates)
    cache_key = (query.strip(), top_k, names)
    if cache_key in _SKILL_MATCH_CACHE:
        return list(_SKILL_MATCH_CACHE[cache_key])

    catalog_lines = []
    for idx, skill in enumerate(candidates, 1):
        keywords = ", ".join(str(x) for x in skill.get("keywords", [])[:12])
        prefer_when = ", ".join(str(x) for x in skill.get("prefer_when", [])[:8])
        avoid_when = ", ".join(str(x) for x in skill.get("avoid_when", [])[:8])
        desc = str(skill.get("description", "")).replace("\n", " ").strip()
        refs = ", ".join(
            str(r.get("name", "")) for r in skill.get("reference_manifest", [])[:6]
            if isinstance(r, dict)
        )
        catalog_lines.append(
            f"[{idx}] name={skill.get('name')}\n"
            f"description={desc}\n"
            f"keywords={keywords}\n"
            f"prefer_when={prefer_when}\n"
            f"avoid_when={avoid_when}\n"
            f"references={refs}"
        )
    prompt = f"""你是 SeismicX 的技能选择器。请根据用户请求，从候选 SKILL 中选择最相关的 {top_k} 个。

原则：
1. 只能选择候选列表里的 name，不能编造技能名。
2. 优先根据每个 SKILL 自己的 description、keywords、references 判断。
3. 用户只是问概念/论文解读/普通 QA 时，不要强行选择绘图或编程技能。
4. 输出 RAW TEXT，不要 JSON。
5. 如果没有明显相关技能，输出 NO_SKILL。

输出格式：
SKILL: skill_name
SKILL: another_skill
或：
NO_SKILL

用户请求：
{query}

候选 SKILL：
{chr(10).join(catalog_lines)}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        raw = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=500,
        )
        matched = _parse_llm_skill_matches(raw, candidates)[:top_k]
        _SKILL_MATCH_CACHE[cache_key] = matched
        return matched
    except Exception:
        return []


def search_skills(query: str, top_k: int = 3) -> List[Dict]:
    """
    按 SKILL 自带元数据检索最相关技能（支持中英文混合查询）。

    流程：
    1. 从 name / description / keywords / references / body 召回候选；
    2. 可用时用 BGE 向量召回补强多语言语义匹配；
    3. 可用时让 LLM 按 raw text 协议重排候选。
    """
    if not str(query or "").strip():
        return []
    candidates = _candidate_skills(query, max_candidates=max(12, top_k * 5))
    if not candidates:
        return []

    skill_map = {skill["name"]: skill for skill in candidates}
    chosen_names = _llm_match_skills(query, candidates, top_k=top_k)
    if chosen_names == ["__NO_SKILL__"]:
        return []
    hits: List[Dict] = []
    seen: set[str] = set()
    for name in chosen_names:
        skill = skill_map.get(name)
        if skill and name not in seen:
            hits.append(skill)
            seen.add(name)
    for skill in candidates:
        if skill["name"] not in seen:
            hits.append(skill)
            seen.add(skill["name"])
        if len(hits) >= top_k:
            break
    return hits[:top_k]


# ── related_skills 双向展开 ───────────────────────────────────────────────────

def _build_reverse_related_index() -> Dict[str, List[str]]:
    """构建反向关联索引：skill_name → [引用了它的其他技能名列表]。"""
    reverse: Dict[str, List[str]] = {}
    for skill in _get_skills():
        for ref in skill.get("related_skills", []):
            reverse.setdefault(ref, []).append(skill["name"])
    return reverse


def _expand_with_related(hits: List[Dict], max_extra: int = 3) -> List[Dict]:
    """
    将搜索结果按 related_skills 双向展开：
    - 正向：命中技能声明了 related_skills → 加入被引用技能
    - 反向：其他技能把命中技能列为 related_skills → 也加入
    """
    skill_map   = {s["name"]: s for s in _get_skills()}
    reverse_idx = _build_reverse_related_index()
    seen  = {s["name"] for s in hits}
    extra: List[Dict] = []

    for hit in hits:
        for ref in hit.get("related_skills", []):
            if ref not in seen and ref in skill_map:
                extra.append(skill_map[ref])
                seen.add(ref)
                if len(extra) >= max_extra:
                    return hits + extra
        for ref in reverse_idx.get(hit["name"], []):
            if ref not in seen and ref in skill_map:
                extra.append(skill_map[ref])
                seen.add(ref)
                if len(extra) >= max_extra:
                    return hits + extra

    return hits + extra


def _collect_rag_sources(skills: List[Dict]) -> List[str]:
    """Collect declared RAG source labels from selected skills."""
    sources: List[str] = []
    for skill in skills:
        for src in skill.get("rag_sources", []) or []:
            src = str(src).strip()
            if src and src not in sources:
                sources.append(src)
        generated_from = str(skill.get("generated_from", "")).strip()
        if generated_from.startswith("knowledge/"):
            src = generated_from[len("knowledge/"):].strip("/")
            if src and src not in sources:
                sources.append(src)
    return sources


# ── References 选择注入 ───────────────────────────────────────────────────────

def _score_reference(ref_name: str, ref_content: str, query_tokens: set, query_lower: str) -> int:
    """
    对单个 reference 文件计算与查询的相关性得分。

    ref_name    : 不含扩展名的文件名，如 "policy-principles"
    ref_content : 文件全文
    """
    score = 0
    name_parts = set(re.split(r"[-_\s/]+", ref_name.lower()))

    # 文件名与查询词匹配（高权重）
    for part in name_parts:
        if len(part) >= 3 and part in query_lower:
            score += 6

    # 标题行（# 开头）与查询词匹配（中权重）
    for line in ref_content.splitlines()[:10]:
        if line.startswith("#"):
            hd = line.lstrip("#").strip().lower()
            for tok in query_tokens:
                if _tok_min_len(tok) and tok in hd:
                    score += 3

    # 正文词频匹配（低权重，前 500 字）
    body_sample = ref_content[:500].lower()
    for tok in query_tokens:
        if _tok_min_len(tok) and tok in body_sample:
            score += 1

    return score


def _select_references(
    skill: Dict,
    query: str,
    max_chars: int = 3000,
    min_score: int = 1,
) -> str:
    """
    从文件夹技能的 references 中选出最相关的文件并格式化为字符串。

    策略：
    1. 对每个 reference 文件按查询相关性打分
    2. 按得分降序排列，依次累积直到超出 max_chars 预算
    3. 得分为 0 的文件不纳入（无关 reference 不污染 context）

    返回格式化的 reference 文本（空字符串表示无相关 reference）。
    """
    refs = skill.get("references", {})
    if not refs:
        return ""

    expanded_query = _expand_query(query)
    query_tokens = set(_tokenize(expanded_query))
    query_lower  = expanded_query.lower()

    scored = []
    for ref_name, ref_content in refs.items():
        s = _score_reference(ref_name, ref_content, query_tokens, query_lower)
        if s >= min_score:
            scored.append((s, ref_name, ref_content))

    if not scored:
        preferred = [
            name for name in refs
            if name.lower() in {"readme", "references/api", "references/common-patterns"}
            or name.lower().endswith("/readme")
        ]
        if not preferred:
            preferred = list(refs.keys())[:2]
        scored = [(0, name, refs[name]) for name in preferred[:2]]

    scored.sort(key=lambda x: x[0], reverse=True)

    parts: List[str] = ["#### 参考资料\n"]
    total = len(parts[0])

    for _, ref_name, ref_content in scored:
        header  = f"\n**{ref_name}**\n\n"
        section = header + ref_content.strip() + "\n"
        if total + len(section) > max_chars:
            remaining = max_chars - total - len(header) - 50
            if remaining > 300:
                parts.append(header + ref_content.strip()[:remaining] + "\n…（已截断）\n")
            break
        parts.append(section)
        total += len(section)

    return "".join(parts).strip()


def _format_reference_manifest(skill: Dict, max_chars: int = 1000) -> str:
    """将文件夹技能的内部资料结构压缩成可读 manifest。"""
    manifest = skill.get("reference_manifest") or []
    if not manifest:
        ref_names = list((skill.get("references") or {}).keys())
        if not ref_names:
            return ""
        manifest = [{"name": name, "headings": []} for name in ref_names]

    lines = ["#### 文件夹技能资料索引"]
    for item in manifest:
        name = str(item.get("name", ""))
        headings = item.get("headings") or []
        if headings:
            brief = "；".join(str(h) for h in headings[:3])
            line = f"- {name}: {brief}"
        else:
            line = f"- {name}"
        lines.append(line)
        if len("\n".join(lines)) >= max_chars:
            lines.append("…（资料索引已截断）")
            break
    resources = skill.get("resource_manifest") or []
    if resources and len("\n".join(lines)) < max_chars:
        shown = []
        for resource in resources[:8]:
            if isinstance(resource, dict):
                shown.append(f"{resource.get('kind', 'file')}:{resource.get('path', '')}")
        if shown:
            lines.append("- bundled resources: " + "；".join(shown))
    return "\n".join(lines)


# ── Context 构建 ──────────────────────────────────────────────────────────────

def _format_skill_section(skill: Dict, query: str, total_budget: int) -> str:
    """
    将单个技能格式化为 LLM 可注入的文本块。

    total_budget : 本 section 的总字符预算（body + refs 合计）。
    对文件夹技能，内部按 2:1 比例分配 body/refs 预算，确保 references 始终有空间。
    """
    if skill["source"] == "user":
        tag = "【自定义】"
    elif skill["source"] == "generated":
        tag = "【文档生成】"
    else:
        tag = ""

    desc    = skill.get("description", "")
    header  = f"### 技能：{tag}{skill['name']}\n"
    desc_ln = f"*{desc}*\n\n" if desc else ""
    overhead = len(header) + len(desc_ln) + 6   # padding "\n\n" + safety

    body_text = skill["body"]

    # ── 文件夹技能：预留 refs 空间，必要时截断 body ───────────────────────────
    if skill.get("is_folder") and skill.get("references"):
        # refs 最多占 1/3 预算（上限 3000 字符）
        ref_alloc  = min(3000, max(300, total_budget // 3))
        body_alloc = max(300, total_budget - ref_alloc - overhead)

        manifest_text = _format_reference_manifest(skill, max_chars=min(1000, ref_alloc // 2))
        ref_text = _select_references(skill, query, max_chars=ref_alloc)

        if ref_text:
            if len(body_text) > body_alloc:
                body_text = body_text[:body_alloc] + "\n\n…（正文已截断，完整内容见技能文件）\n"
            sections = [header + desc_ln + body_text]
            if manifest_text:
                sections.append(manifest_text)
            sections.append(ref_text)
            return "\n\n".join(sections) + "\n\n"

    # ── 单文件技能 / 无命中 references：body 占满预算 ────────────────────────
    body_alloc = max(300, total_budget - overhead)
    if len(body_text) > body_alloc:
        body_text = body_text[:body_alloc] + "\n\n…（已截断）\n"

    return header + desc_ln + body_text + "\n\n"


def build_skill_context(query: str, max_chars: int = 8000, top_k: int = 2) -> str:
    """
    为 LLM 提示构建技能上下文块。

    命中的技能会按 related_skills 双向展开；
    文件夹技能会自动将查询相关的 references 注入到上下文中。

    返回
    ----
    str — 可直接拼入系统提示的技能文档片段；无匹配时返回空字符串
    """
    hits = search_skills(query, top_k=top_k)
    if not hits:
        return ""

    hits = _expand_with_related(hits, max_extra=3)

    parts: List[str] = ["## 相关技能文档（请优先使用以下函数和示例）\n"]
    total = len(parts[0])

    for idx, skill in enumerate(hits):
        # 为本技能分配总预算（剩余空间，最少保证 500 字符）
        remaining_skills = max(1, len(hits) - idx)
        skill_budget = max(700, (max_chars - total) // remaining_skills)
        section = _format_skill_section(skill, query, skill_budget)

        if total + len(section) > max_chars:
            remaining = max_chars - total - 20
            if remaining > 200:
                parts.append(section[:remaining] + "\n…（已截断）\n")
            break
        parts.append(section)
        total += len(section)

    return "".join(parts).strip()


# ── RAG + Workflow 联合上下文构建 ─────────────────────────────────────────────

def build_skill_context_with_rag(
    query: str,
    max_skill_chars: int = 8000,
    max_rag_chars: int = 2000,
    top_k: int = 2,
) -> Tuple[str, str]:
    """
    为 LLM 构建技能 + Workflow + RAG 联合上下文。

    流程：
    1. 检索相关技能（含 related_skills 双向展开 + references 注入）
    2. 检索相关 Workflow 脚本；将其依赖技能强制加入检索池
    3. 对所有命中的 RAG-backed 技能查询知识库

    Returns
    -------
    (skill_context, rag_context)
    """
    expanded_query = _expand_query(query)
    hits = search_skills(expanded_query, top_k=top_k)
    hits = _expand_with_related(hits, max_extra=3)

    # ── Workflow 检索 ─────────────────────────────────────────────────────────
    workflow_ctx       = ""
    workflow_skill_names: List[str] = []
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _root = _Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from seismo_skill.workflow_runner import (
            build_workflow_context as _bwc,
            load_workflow as _load_wf,
        )
        workflow_ctx, workflow_skill_names = _bwc(expanded_query, max_chars=6000, top_k=2)
        for _hit in hits:
            _wf_name = _hit.get("workflow", "")
            if _wf_name:
                _wf = _load_wf(_wf_name)
                if _wf:
                    _wf_snames = _wf.get("skill_names") or [
                        s["name"] if isinstance(s, dict) else s
                        for s in _wf.get("skills", [])
                    ]
                    _wf_header = (
                        f"### 工作流：{_wf['name']} — {_wf['title']} "
                        f"(由技能 `{_hit['name']}` 声明)\n"
                        f"**依赖技能：** {', '.join(f'`{n}`' for n in _wf_snames) or '无'}\n\n"
                    )
                    _wf_body    = (_wf.get("guide") or _wf.get("description") or "")
                    _wf_section = _wf_header + _wf_body + "\n\n"
                    if _wf_section not in workflow_ctx:
                        workflow_ctx = (workflow_ctx + "\n\n" + _wf_section).strip()
                    for _sn in _wf_snames:
                        if _sn not in workflow_skill_names:
                            workflow_skill_names.append(_sn)
    except Exception:
        pass

    # Workflow 依赖技能强制加入 hits
    if workflow_skill_names:
        skill_map      = {s["name"]: s for s in _get_skills()}
        existing_names = {s["name"] for s in hits}
        for wf_skill_name in workflow_skill_names:
            if wf_skill_name not in existing_names and wf_skill_name in skill_map:
                hits.append(skill_map[wf_skill_name])
                existing_names.add(wf_skill_name)

    # ── 技能文档文本 ──────────────────────────────────────────────────────────
    skill_ctx = ""
    if hits:
        parts: List[str] = ["## 相关技能文档（请优先使用以下函数和示例）\n"]
        total = len(parts[0])
        for idx, skill in enumerate(hits):
            remaining_skills = max(1, len(hits) - idx)
            skill_budget = max(700, (max_skill_chars - total) // remaining_skills)
            section = _format_skill_section(skill, expanded_query, skill_budget)
            if total + len(section) > max_skill_chars:
                remaining = max_skill_chars - total - 20
                if remaining > 200:
                    parts.append(section[:remaining] + "\n…（已截断）\n")
                break
            parts.append(section)
            total += len(section)
        skill_ctx = "".join(parts).strip()

    if workflow_ctx:
        skill_ctx = (skill_ctx + "\n\n" + workflow_ctx).strip() if skill_ctx else workflow_ctx
    if max_skill_chars > 0 and len(skill_ctx) > max_skill_chars:
        skill_ctx = skill_ctx[:max_skill_chars] + "\n…（技能/工作流上下文已按预算截断）"

    # ── RAG 检索 ──────────────────────────────────────────────────────────────
    # Always try RAG. Older logic only queried KB for skills declaring rag_sources,
    # which caused weak recall for general coding/data tasks.
    rag_ctx = ""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _web = _Path(__file__).parent.parent / "web_app"
        if str(_web) not in _sys.path:
            _sys.path.insert(0, str(_web))
        from rag_engine import get_knowledge_base
        kb = get_knowledge_base()
        skill_terms = []
        for s in hits[:5]:
            skill_terms.append(s.get("name", ""))
            skill_terms.extend((s.get("keywords") or [])[:8])
        rag_sources = _collect_rag_sources(hits)
        rag_query = "\n".join([
            query,
            expanded_query,
            "Relevant skills: " + ", ".join(t for t in skill_terms if t),
            "RAG sources: " + ", ".join(rag_sources),
        ])
        if rag_sources:
            rag_ctx = kb.build_rag_context(
                rag_query,
                top_k=8,
                max_chars=max_rag_chars,
                score_threshold=0.0,
                sources=rag_sources,
            )
        if not rag_ctx:
            rag_ctx = kb.build_rag_context(
                rag_query, top_k=8, max_chars=max_rag_chars, score_threshold=0.15
            )
        if not rag_ctx:
            rag_ctx = kb.build_rag_context(
                expanded_query, top_k=8, max_chars=max_rag_chars, score_threshold=0.0
            )
    except Exception:
        pass

    return skill_ctx, rag_ctx


# ── 模板 ─────────────────────────────────────────────────────────────────────

SKILL_TEMPLATE = """\
---
name: {name}
description: >-
  {description}
category: custom
keywords: {keywords}
---

# {title}

## 描述

{description}

---

## 主要函数

### `function_name(param1, param2)`

**参数：**
- `param1` : type — 说明
- `param2` : type — 说明

**返回：** type — 说明

```python
# 示例代码
result = function_name(param1, param2)
print(result)
```

---

## 注意事项

- 注意事项 1
- 注意事项 2
"""

FOLDER_SKILL_TEMPLATE = {
    "SKILL.md": """\
---
name: {name}
description: >-
  {description}
category: custom
keywords: {keywords}
---

# {title}

## 说明

{description}

## 工作流

1. 步骤一
2. 步骤二

## 输出格式

描述期望输出格式。

## Related files

| 文件 | 使用时机 |
|---|---|
| [references/guide.md](references/guide.md) | 主要参考指南 |
""",
    "agents/openai.yaml": """\
interface:
  display_name: "{display_name}"
  short_description: "{description}"
  default_prompt: "请帮我使用 {name} 技能处理以下任务："
""",
    "references/guide.md": """\
# {title} — 参考指南

## 核心概念

在此描述该技能涉及的核心概念。

## 常用模式

在此列出常见用法模式。

## 注意事项

在此说明注意事项和边界情况。
""",
    "README.md": """\
# {title}

{description}

## 文件结构

```
{name}/
├── SKILL.md              # 主技能定义
├── agents/
│   └── openai.yaml       # LLM 接口配置
├── references/
│   └── guide.md          # 参考指南
└── README.md             # 本文件
```

## 使用方式

该技能由 SeismicX 技能引擎自动加载。将整个文件夹放入
`seismo_skill/skills/` 或 `seismo_skill/user_skills/` 即可使用。
""",
}
