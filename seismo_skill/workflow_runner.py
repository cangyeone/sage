"""
workflow_runner.py — Markdown + YAML-frontmatter workflow discovery and context building.

Workflows are `.md` files with a YAML frontmatter block followed by a Markdown
guide body — the same format used by skills in seismo_skill/:

    ---
    name: gmt_terrain_map
    title: GMT 地形图绘制工作流
    version: "1.0"
    description: 使用 GMT 绘制带地形底图的完整流程
    keywords:
      - GMT
      - terrain
    skills:
      - name: gmt_plotting
        role: 地图底图绘制
      - name: _gen_gmt_docs_6_5
        role: 官方文档 RAG 参考
    steps:
      - id: prepare_cpt
        skill: gmt_plotting
        description: 生成地形色标
      - id: render_terrain
        skill: gmt_plotting
        description: 渲染地形底图
        depends_on: [prepare_cpt]
    ---

    ## GMT 地形图绘制步骤
    ...

Role responsibilities
---------------------
  workflow  — 作业流程书：what steps, which skills, in what order
  skill     — 专项操作手册：how to use a specific tool/method
  agent     — 调度员：matches workflow → loads skills → decomposes task
  code engine — 程序员：generates & fixes Python/GMT/Shell code
  tool      — 具体工具：Python, GMT, Shell executors

Directory layout
----------------
  seismo_skill/workflows/         builtin workflows (shipped with project)
  ~/.seismicx/workflows/          user-defined workflows (higher priority)

Public API
----------
  list_workflows()                → list[dict]  all workflow metadata
  search_workflows(query, top_k)  → list[dict]  ranked by relevance
  load_workflow(name)             → dict | None  full workflow entry
  save_user_workflow(name, text)  → Path
  delete_user_workflow(name)      → bool
  build_workflow_context(query)   → (str, list[str])
  get_user_workflow_dir()         → Path
  invalidate_cache()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── Directories ───────────────────────────────────────────────────────────────

_BUILTIN_WORKFLOW_DIR = Path(__file__).parent / "workflows"


def get_user_workflow_dir() -> Path:
    """Return ~/.seismicx/workflows/, creating it if needed."""
    d = Path.home() / ".seismicx" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Markdown + YAML frontmatter parsing ──────────────────────────────────────

def _split_frontmatter(text: str) -> Tuple[str, str]:
    """
    Split a markdown file into (frontmatter_yaml, body_markdown).
    Returns ('', text) if no frontmatter block is found.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return "", text
    # Find the closing ---
    rest = stripped[3:]
    end = rest.find("\n---")
    if end == -1:
        return "", text
    fm_yaml = rest[:end].strip()
    body    = rest[end + 4:].lstrip("\n")
    return fm_yaml, body


def _parse_md_workflow(text: str, path: Path, source_label: str) -> Optional[Dict]:
    """
    Parse a workflow .md file (YAML frontmatter + Markdown body) into an entry dict.

    Frontmatter fields
    ------------------
    name        str     workflow identifier (defaults to file stem)
    title       str
    version     str
    description str
    keywords    list[str]
    skills      list of {name, role}
    steps       list of {id, skill, description, depends_on}

    Body
    ----
    Markdown text used as the workflow guide injected into LLM context.
    """
    fm_yaml, body = _split_frontmatter(text)
    if not fm_yaml:
        return None  # no frontmatter → not a workflow file

    try:
        data = yaml.load(fm_yaml, Loader=yaml.SafeLoader)  # SafeLoader = pure Python; avoids yaml._yaml C ext in threads
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    name = str(data.get("name", path.stem))

    # keywords
    raw_kw = data.get("keywords", [])
    if isinstance(raw_kw, str):
        keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
    elif isinstance(raw_kw, list):
        keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
    else:
        keywords = []

    # skills: list of {name, role} or plain strings
    raw_skills = data.get("skills", [])
    skills: List[Dict] = []
    skill_names: List[str] = []
    for item in raw_skills:
        if isinstance(item, dict):
            sn = str(item.get("name", "")).strip()
            if sn:
                skills.append({"name": sn, "role": str(item.get("role", ""))})
                skill_names.append(sn)
        elif isinstance(item, str) and item.strip():
            skills.append({"name": item.strip(), "role": ""})
            skill_names.append(item.strip())

    # steps: list of step dicts
    raw_steps = data.get("steps", [])
    steps: List[Dict] = []
    for step in raw_steps:
        if isinstance(step, dict):
            sid = str(step.get("id", "")).strip()
            if sid:
                steps.append({
                    "id":          sid,
                    "skill":       str(step.get("skill", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "depends_on":  list(step.get("depends_on", [])),
                })

    return {
        "name":        name,
        "title":       str(data.get("title", name.replace("_", " ").title())),
        "version":     str(data.get("version", "1.0")),
        "category":    str(data.get("category", "workflow")),
        "keywords":    keywords,
        "description": str(data.get("description", "")).strip(),
        "skills":      skills,        # list of {name, role}
        "skill_names": skill_names,   # plain list[str] for fast lookup
        "steps":       steps,
        "guide":       body.strip(),  # Markdown body = the workflow guide
        "source_text": text,          # full .md source
        "path":        str(path),
        "filename":    path.name,
        "source":      str(data.get("source", source_label)),
    }


def _load_from_dir(directory: Path, source_label: str) -> List[Dict]:
    """Scan directory for .md workflow files and build entry dicts."""
    workflows: List[Dict] = []
    if not directory.exists():
        return workflows
    for md_file in sorted(directory.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        entry = _parse_md_workflow(text, md_file, source_label)
        if entry:
            workflows.append(entry)
    return workflows


# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE: Optional[List[Dict]] = None


def _get_workflows() -> List[Dict]:
    global _CACHE
    if _CACHE is None:
        builtin = _load_from_dir(_BUILTIN_WORKFLOW_DIR, "builtin")
        user    = _load_from_dir(get_user_workflow_dir(), "user")
        user_names = {w["name"] for w in user}
        _CACHE = [w for w in builtin if w["name"] not in user_names] + user
    return _CACHE


def invalidate_cache() -> None:
    """Invalidate workflow cache (call after adding/removing .md files)."""
    global _CACHE
    _CACHE = None


# ── Public CRUD ───────────────────────────────────────────────────────────────

def list_workflows() -> List[Dict]:
    """Return all workflows (metadata only — no source_text, no guide body)."""
    _SKIP = {"source_text", "guide"}
    return [{k: v for k, v in w.items() if k not in _SKIP} for w in _get_workflows()]


def load_workflow(name: str) -> Optional[Dict]:
    """Return the full workflow entry for `name`, or None if not found."""
    for w in _get_workflows():
        if w["name"] == name:
            return w
    return None


def save_user_workflow(name: str, text: str) -> Path:
    """Save a user workflow to ~/.seismicx/workflows/<name>.md."""
    safe_name = re.sub(r"[^\w\-]", "_", name)
    target = get_user_workflow_dir() / f"{safe_name}.md"
    target.write_text(text, encoding="utf-8")
    invalidate_cache()
    return target


def delete_user_workflow(name: str) -> bool:
    """Delete a user-defined workflow. Returns True if deleted."""
    for w in _get_workflows():
        if w["name"] == name and w["source"] == "user":
            Path(w["path"]).unlink(missing_ok=True)
            invalidate_cache()
            return True
    return False


# ── Search ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", text)]
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese)
    for i in range(len(chinese) - 1):
        tokens.append(chinese[i] + chinese[i + 1])
    return tokens


def search_workflows(query: str, top_k: int = 2) -> List[Dict]:
    """Search workflows by keyword relevance. Returns top_k results by score."""
    query_tokens = set(_tokenize(query))
    query_lower  = query.lower()
    scored: List[Tuple[int, Dict]] = []

    for wf in _get_workflows():
        score = 0
        for kw in wf["keywords"]:
            kw_lower = kw.lower()
            if kw_lower in query_lower:
                score += 5
                continue
            if any(len(t) >= 2 and t in kw_lower for t in query_tokens):
                score += 3
        name_lower = wf["name"].lower()
        if name_lower in query_lower or any(t in name_lower for t in query_tokens if len(t) >= 2):
            score += 2
        title_lower = wf["title"].lower()
        if any(t in title_lower for t in query_tokens if len(t) >= 3):
            score += 1
        desc_lower = wf["description"].lower()
        for t in query_tokens:
            if len(t) >= 3 and t in desc_lower:
                score += 1
        if score > 0:
            if wf["source"] == "user":
                score += 1
            scored.append((score, wf))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:top_k]]


# ── Context building ──────────────────────────────────────────────────────────

def build_workflow_context(
    query: str,
    max_chars: int = 8000,
    top_k: int = 2,
) -> Tuple[str, List[str]]:
    """
    Build a workflow context block for LLM injection.

    Finds relevant workflow .md files, formats their step DAG summary and
    Markdown guide body, and returns all referenced skill names so the caller
    can also inject skill docs.

    Returns
    -------
    (context_str, skill_names)
    """
    hits = search_workflows(query, top_k=top_k)
    if not hits:
        return "", []

    all_skill_refs: List[str] = []
    parts: List[str] = ["## 工作流参考（请按照以下流程组织代码结构）\n"]
    total = len(parts[0])

    for wf in hits:
        tag = "【用户自定义】" if wf["source"] == "user" else ""

        skill_parts = [
            f"`{s['name']}`" + (f"（{s['role']}）" if s.get("role") else "")
            for s in wf["skills"]
        ]
        skills_str = "、".join(skill_parts) or "无"

        steps_lines = ""
        if wf["steps"]:
            lines = []
            for st in wf["steps"]:
                deps = f" → 依赖: {', '.join(st['depends_on'])}" if st.get("depends_on") else ""
                lines.append(f"  - **{st['id']}** [{st.get('skill','')}]: {st['description']}{deps}")
            steps_lines = "\n**执行步骤：**\n" + "\n".join(lines) + "\n"

        header = (
            f"### 工作流：{tag}{wf['name']} — {wf['title']}\n"
            f"**依赖技能：** {skills_str}\n"
            f"{steps_lines}\n"
        )
        body   = wf["guide"] or wf["description"] or "(无详细步骤说明)"
        section = header + body + "\n\n"

        # Always collect skill refs regardless of truncation
        all_skill_refs.extend(wf["skill_names"])

        if total + len(section) > max_chars:
            remaining = max_chars - total - 20
            if remaining > 200:
                parts.append(section[:remaining] + "\n...(已截断)\n")
            break

        parts.append(section)
        total += len(section)

    ctx = "".join(parts).strip() if len(parts) > 1 else ""

    seen: set = set()
    deduped: List[str] = []
    for s in all_skill_refs:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return ctx, deduped
