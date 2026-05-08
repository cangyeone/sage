"""
skill_loader.py — seismo_skill 技能引擎 v2

支持两种技能格式：
  A. 单文件技能：skills/<name>.md
  B. 文件夹技能：skills/<name>/SKILL.md + agents/*.yaml + 目录内递归 *.md

从三个目录加载技能：
  1. seismo_skill/skills/         内置技能（随项目发布）
  2. seismo_skill/user_skills/    项目内用户自定义技能（同名时覆盖内置）
  3. ~/.seismicx/skills/          旧版兼容读取目录（不再作为新写入位置）

v2 新特性
---------
- 文件夹技能完整解析：SKILL.md + 目录内递归 Markdown 资料 + agents/
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
from typing import Dict, List, Optional, Tuple, Union

# ── 目录定义 ─────────────────────────────────────────────────────────────────

_BUILTIN_SKILL_DIR = Path(__file__).parent / "skills"
_PROJECT_USER_SKILL_DIR = Path(__file__).parent / "user_skills"
_LEGACY_USER_SKILL_DIR = Path.home() / ".seismicx" / "skills"

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


def _extract_markdown_headings(text: str, limit: int = 12) -> List[str]:
    """提取少量 Markdown 标题，用于 manifest 和关键词推断。"""
    headings: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", line)
        if not m:
            continue
        heading = m.group(1).strip()
        if heading:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings

def _load_references(folder: Path) -> Dict[str, str]:
    """
    递归加载文件夹技能目录内的辅助 Markdown 文件。

    约定：
      - skills/<name>/SKILL.md 是该文件夹技能的主入口，不作为 reference。
      - 目录内其他 *.md（README、references、docs、嵌套 SKILL.md 等）都作为
        该技能的内部资料加载。
      - 不把嵌套目录里的 SKILL.md 注册成独立顶层技能；它只是父技能的一份资料。

    返回 {ref_name: content_str}，文件读取失败则跳过。
    """
    refs: Dict[str, str] = {}
    for md_file in sorted(folder.rglob("*.md")):
        if md_file == folder / "SKILL.md":
            continue
        if _is_hidden_or_ignored(md_file, folder):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            refs[_reference_key(md_file, folder)] = content
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
    文件夹必须包含 SKILL.md；references/ 和 agents/ 为可选。

    返回技能 dict，或 None（SKILL.md 不存在 / 读取失败）。
    """
    skill_md = folder / "SKILL.md"
    if not skill_md.exists():
        return None

    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return None

    meta, body = _parse_frontmatter(text)
    effective_source = meta.get("source", source)
    refs = _load_references(folder)
    description = meta.get("description", "")
    keywords = _infer_keywords(
        name=meta.get("name", folder.name),
        meta_keywords=meta.get("keywords", []),
        description=description,
        body=body,
        refs=refs,
    )

    return {
        "name":            meta.get("name", folder.name),
        "description":     description,
        "category":        meta.get("category", "custom" if source == "user" else ""),
        "keywords":        keywords,
        "path":            str(skill_md),
        "folder":          str(folder),
        "body":            body.strip(),
        "full_text":       text.strip(),
        "is_folder":       True,
        "references":      refs,
        "reference_manifest": _build_reference_manifest(refs),
        "agent_config":    _load_agent_config(folder),
        "source":          effective_source,
        "filename":        "SKILL.md",
        "rag_sources":     meta.get("rag_sources", []),
        "generated_from":  meta.get("generated_from", ""),
        "related_skills":  meta.get("related_skills", []),
        "workflow":        meta.get("workflow", ""),
    }


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
        effective_source = meta.get("source", source)
        name = meta.get("name", md_file.stem)
        entry: Dict = {
            "name":           name,
            "description":    meta.get("description", ""),
            "category":       meta.get("category", "custom" if source == "user" else ""),
            "keywords":       meta.get("keywords", []),
            "path":           str(md_file),
            "folder":         "",
            "body":           body.strip(),
            "full_text":      text.strip(),
            "is_folder":      False,
            "references":     {},
            "reference_manifest": [],
            "agent_config":   {},
            "source":         effective_source,
            "filename":       md_file.name,
            "rag_sources":    meta.get("rag_sources", []),
            "generated_from": meta.get("generated_from", ""),
            "related_skills": meta.get("related_skills", []),
            "workflow":       meta.get("workflow", ""),
        }
        # 只在该名称还没被文件夹技能占用时才添加（文件夹优先）
        if name not in skills or not skills[name].get("is_folder"):
            skills[name] = entry

    # ── 文件夹技能 ────────────────────────────────────────────────────────────
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        entry = _load_folder_skill(subdir, source)
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
            "description":      s.get("description", ""),
            "category":         s["category"],
            "keywords":         s["keywords"],
            "source":           s["source"],
            "is_folder":        s.get("is_folder", False),
            "filename":         s["filename"],
            "rag_sources":      s.get("rag_sources", []),
            "generated_from":   s.get("generated_from", ""),
            "related_skills":   s.get("related_skills", []),
            "workflow":         s.get("workflow", ""),
            "ref_names":        list(s.get("references", {}).keys()),
            "ref_count":        len(s.get("references", {})),
            "reference_manifest": s.get("reference_manifest", []),
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


def install_skill_from_dir(source_dir: Union[str, Path], overwrite: bool = True) -> Dict:
    """
    从本地目录安装文件夹技能到 seismo_skill/user_skills/<name>/。

    source_dir 必须包含 SKILL.md。
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
    ValueError         — source_dir 中不含 SKILL.md
    FileExistsError    — overwrite=False 且目标目录已存在
    """
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"目录不存在：{source_dir}")

    skill_md = source_dir / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"目录中未找到 SKILL.md：{source_dir}")

    # 解析 name
    text = skill_md.read_text(encoding="utf-8")
    meta, _ = _parse_frontmatter(text)
    name = (meta.get("name") or source_dir.name).strip()
    safe_name = re.sub(r"[^\w\-]", "_", name)

    target_dir = get_user_skill_dir() / safe_name
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"技能目录已存在（overwrite=False）：{target_dir}")
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)
    invalidate_cache()

    entry = _load_folder_skill(target_dir, "user")
    if entry is None:
        raise RuntimeError(f"安装后无法重新加载技能：{target_dir}")
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

    candidates: List[Dict] = []
    seen: set[str] = set()

    for _, skill in _embedding_rank_skills(query, skills, max_candidates=max_candidates):
        if skill["name"] not in seen:
            candidates.append(skill)
            seen.add(skill["name"])

    lexical = sorted(
        ((_metadata_score(query, skill), skill) for skill in skills),
        key=lambda item: item[0],
        reverse=True,
    )
    for score, skill in lexical:
        if score <= 0 and candidates:
            continue
        if skill["name"] not in seen:
            candidates.append(skill)
            seen.add(skill["name"])
        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]


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
        desc = str(skill.get("description", "")).replace("\n", " ").strip()
        refs = ", ".join(
            str(r.get("name", "")) for r in skill.get("reference_manifest", [])[:6]
            if isinstance(r, dict)
        )
        catalog_lines.append(
            f"[{idx}] name={skill.get('name')}\n"
            f"description={desc}\n"
            f"keywords={keywords}\n"
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
