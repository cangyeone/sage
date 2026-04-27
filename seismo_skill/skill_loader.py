"""
skill_loader.py — seismo_skill 技能文档加载与检索

从两个目录读取 Markdown 格式的技能说明文件：
  1. seismo_skill/skills/   内置技能（随项目发布）
  2. ~/.seismicx/skills/    用户自定义技能（优先级更高，可覆盖内置同名技能）

接口
----
list_skills()                → list[dict]   所有技能元信息（含来源标记）
search_skills(query, k)      → list[dict]   按关键词检索最相关技能
load_skill(name)             → str          加载完整技能文档文本
save_user_skill(name, text)  → Path         保存用户自定义技能文件
delete_user_skill(name)      → bool         删除用户自定义技能
build_skill_context(query)   → str          为 LLM 提示构建技能上下文块
get_user_skill_dir()         → Path         获取用户技能目录路径
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 目录定义 ────────────────────────────────────────────────────────────────

# 内置技能目录（seismo_skill/skills/ 子目录）
_BUILTIN_SKILL_DIR = Path(__file__).parent / "skills"

# 用户自定义技能目录
def get_user_skill_dir() -> Path:
    """返回用户自定义技能目录（~/.seismicx/skills/），不存在则自动创建。"""
    d = Path.home() / ".seismicx" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 解析工具 ────────────────────────────────────────────────────────────────

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
    for line in frontmatter_raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    # keywords 字段拆成列表
    if "keywords" in meta:
        meta["keywords"] = [kw.strip() for kw in meta["keywords"].split(",") if kw.strip()]

    # rag_sources 字段拆成列表（逗号分隔的文档名）
    if "rag_sources" in meta:
        meta["rag_sources"] = [s.strip() for s in meta["rag_sources"].split(",") if s.strip()]

    # related_skills 字段拆成列表（逗号分隔的技能名）
    if "related_skills" in meta:
        meta["related_skills"] = [s.strip() for s in meta["related_skills"].split(",") if s.strip()]

    # workflow 字段：单个工作流名称（对应 seismo_skill/workflows/<name>.md）
    # 不做列表处理，保留为字符串
    # （已在 key: val 解析中处理，此处无需额外操作）

    return meta, body


def _load_from_dir(directory: Path, source: str) -> List[Dict]:
    """扫描指定目录下所有 .md 文件并解析为技能条目。"""
    skills = []
    if not directory.exists():
        return skills
    for md_file in sorted(directory.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        meta, body = _parse_frontmatter(text)
        # frontmatter 中的 source 字段可覆盖目录级别的 source
        # （用于 generated 技能保存在 user 目录中，但标记为 generated）
        effective_source = meta.get("source", source)
        entry = {
            "name": meta.get("name", md_file.stem),
            "category": meta.get("category", "custom" if source == "user" else ""),
            "keywords": meta.get("keywords", []),
            "path": str(md_file),
            "body": body.strip(),
            "full_text": text.strip(),
            "source": effective_source,   # "builtin" | "user" | "generated"
            "filename": md_file.name,
            "rag_sources": meta.get("rag_sources", []),
            "generated_from": meta.get("generated_from", ""),
            "related_skills": meta.get("related_skills", []),
            "workflow": meta.get("workflow", ""),   # linked workflow script name
        }
        skills.append(entry)
    return skills


def _load_all_skills() -> List[Dict]:
    """
    加载内置技能 + 用户自定义技能。
    同名技能中用户版本覆盖内置版本（按 name 字段去重）。
    """
    builtin = _load_from_dir(_BUILTIN_SKILL_DIR, "builtin")
    user    = _load_from_dir(get_user_skill_dir(), "user")

    # 优先级：user > generated > builtin（同名时高优先级覆盖低优先级）
    user_names      = {s["name"] for s in user}
    generated_names = {s["name"] for s in user if s["source"] == "generated"}
    # generated 技能同样存储在 user 目录，已在 user 列表中；
    # 这里只需让真正的 user 覆盖 builtin，generated 也覆盖 builtin
    merged = [s for s in builtin if s["name"] not in user_names] + user
    return merged


# ── 缓存 ────────────────────────────────────────────────────────────────────

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


# ── 公开 CRUD 接口 ───────────────────────────────────────────────────────────

def list_skills() -> List[Dict]:
    """
    列出所有可用技能（不含完整文档体）。

    返回
    ----
    list of dict: [{name, category, keywords, source, filename}, ...]
      source: "builtin" | "user"
    """
    return [
        {
            "name": s["name"],
            "category": s["category"],
            "keywords": s["keywords"],
            "source": s["source"],
            "filename": s["filename"],
            "rag_sources": s.get("rag_sources", []),
            "generated_from": s.get("generated_from", ""),
            "related_skills": s.get("related_skills", []),
            "workflow": s.get("workflow", ""),
        }
        for s in _get_skills()
    ]


def _build_reverse_related_index() -> Dict[str, List[str]]:
    """
    构建反向关联索引：skill_name → [引用了它的其他技能名列表]。
    用于双向展开：命中 A 时自动拉取所有把 A 列为 related 的技能。
    """
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

    已在 hits 中的技能不重复添加。最多额外添加 max_extra 个。
    """
    skill_map = {s["name"]: s for s in _get_skills()}
    reverse_idx = _build_reverse_related_index()

    seen = {s["name"] for s in hits}
    extra: List[Dict] = []

    for hit in hits:
        # 正向
        for ref in hit.get("related_skills", []):
            if ref not in seen and ref in skill_map:
                extra.append(skill_map[ref])
                seen.add(ref)
                if len(extra) >= max_extra:
                    return hits + extra
        # 反向
        for ref in reverse_idx.get(hit["name"], []):
            if ref not in seen and ref in skill_map:
                extra.append(skill_map[ref])
                seen.add(ref)
                if len(extra) >= max_extra:
                    return hits + extra

    return hits + extra


def load_skill(name: str) -> str:
    """
    按技能名称加载完整 Markdown 文档。

    返回
    ----
    str — 技能文档全文；若不存在返回空字符串
    """
    for skill in _get_skills():
        if skill["name"] == name:
            return skill["full_text"]
    return ""


def save_user_skill(name: str, text: str) -> Path:
    """
    保存用户自定义技能文件到 ~/.seismicx/skills/<name>.md。

    参数
    ----
    name : str — 技能文件名（不含 .md，建议使用英文下划线命名）
    text : str — 完整 Markdown 文本（含 frontmatter）

    返回
    ----
    Path — 保存的文件路径
    """
    skill_dir = get_user_skill_dir()
    # 文件名只保留安全字符
    safe_name = re.sub(r"[^\w\-]", "_", name)
    target = skill_dir / f"{safe_name}.md"
    target.write_text(text, encoding="utf-8")
    invalidate_cache()
    return target


def delete_user_skill(name: str) -> bool:
    """
    删除用户自定义技能（user 或 generated source 均可删除；内置技能不可删除）。

    返回
    ----
    bool — True 表示删除成功，False 表示未找到该技能或它是内置技能
    """
    for skill in _get_skills():
        if skill["name"] == name and skill["source"] in ("user", "generated"):
            Path(skill["path"]).unlink(missing_ok=True)
            invalidate_cache()
            return True
    return False


def get_skill_detail(name: str) -> Optional[Dict]:
    """返回技能完整条目（含 body / full_text / source）。"""
    for skill in _get_skills():
        if skill["name"] == name:
            return skill
    return None


# ── 检索 ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    混合分词：英文按空格/标点分词，中文按单字及 bigram 分词。
    """
    tokens: List[str] = []
    for tok in re.findall(r"[a-zA-Z0-9_\-\.]+", text):
        tokens.append(tok.lower())
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese_chars)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


def search_skills(query: str, top_k: int = 3) -> List[Dict]:
    """
    按关键词检索最相关技能文档（支持中英文混合查询）。

    返回按相关性降序排列的技能列表。
    """
    query_tokens = set(_tokenize(query))
    query_lower = query.lower()
    scored: List[Tuple[int, Dict]] = []

    for skill in _get_skills():
        score = 0

        # 1. keywords 双向子串匹配（最高优先级）
        for kw in skill["keywords"]:
            kw_lower = kw.lower()
            if kw_lower in query_lower:
                score += 5
                continue
            for tok in query_tokens:
                if len(tok) >= 2 and tok in kw_lower:
                    score += 3
                    break

        # 2. 技能名匹配
        name_lower = skill["name"].lower()
        if name_lower in query_lower or any(tok in name_lower for tok in query_tokens if len(tok) >= 2):
            score += 2

        # 3. 文档体匹配（函数名、标题）
        body_lower = skill["body"].lower()
        for tok in query_tokens:
            if len(tok) >= 3 and tok in body_lower:
                score += 1

        # 用户自定义技能额外加分（优先推荐）
        if score > 0 and skill["source"] == "user":
            score += 1

        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]


def build_skill_context(query: str, max_chars: int = 6000, top_k: int = 2) -> str:
    """
    为 LLM 提示构建技能上下文块。

    命中的技能会按 related_skills 双向展开：被关联的技能自动附带注入，
    确保内置技能与生成技能能同时出现在上下文中。

    返回
    ----
    str — 可直接拼入系统提示的技能文档片段；无匹配时返回空字符串
    """
    hits = search_skills(query, top_k=top_k)
    if not hits:
        return ""

    # 双向展开关联技能（最多额外 3 个）
    hits = _expand_with_related(hits, max_extra=3)

    parts: List[str] = ["## 相关技能文档（请优先使用以下函数和示例）\n"]
    total = len(parts[0])

    for skill in hits:
        if skill["source"] == "user":
            tag = "【自定义】"
        elif skill["source"] == "generated":
            tag = "【文档生成】"
        else:
            tag = ""
        section = f"### 技能：{tag}{skill['name']}\n\n{skill['body']}\n\n"
        if total + len(section) > max_chars:
            remaining = max_chars - total - 20
            if remaining > 200:
                parts.append(section[:remaining] + "\n...(已截断)\n")
            break
        parts.append(section)
        total += len(section)

    return "".join(parts).strip()


# ── RAG + Workflow 联合上下文构建 ────────────────────────────────────────────

def build_skill_context_with_rag(
    query: str,
    max_skill_chars: int = 6000,
    max_rag_chars: int = 2000,
    top_k: int = 2,
) -> Tuple[str, str]:
    """
    为 LLM 构建技能 + Workflow + RAG 联合上下文。

    流程：
    1. 检索相关技能（含 related_skills 双向展开）
    2. 检索相关 Workflow 脚本；将其依赖技能强制加入检索池
    3. 对所有命中的 RAG-backed 技能查询知识库

    Returns
    -------
    (skill_context, rag_context)
        skill_context — 技能文档 + Workflow 引导文本（合并）
        rag_context   — RAG 检索到的文档片段
    """
    # ── 1. 技能检索（含关联展开）────────────────────────────────────────────
    hits = search_skills(query, top_k=top_k)
    hits = _expand_with_related(hits, max_extra=3)

    # ── 2. Workflow 检索（关键词搜索 + skill.workflow 字段声明）─────────────
    workflow_ctx = ""
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
        workflow_ctx, workflow_skill_names = _bwc(query, max_chars=6000, top_k=2)

        # 额外：被命中技能通过 workflow 字段声明的工作流也强制加入
        for _hit in hits:
            _wf_name = _hit.get("workflow", "")
            if _wf_name:
                _wf = _load_wf(_wf_name)
                if _wf:
                    # 追加工作流内容
                    # skills is list of {name, role} dicts; use skill_names for plain strings
                    _wf_snames = _wf.get("skill_names") or [
                        s["name"] if isinstance(s, dict) else s
                        for s in _wf.get("skills", [])
                    ]
                    _wf_header = (
                        f"### 工作流：{_wf['name']} — {_wf['title']} "
                        f"(由技能 `{_hit['name']}` 声明)\n"
                        f"**依赖技能：** {', '.join(f'`{n}`' for n in _wf_snames) or '无'}\n\n"
                    )
                    _wf_body = (_wf.get("guide") or _wf.get("description") or "")
                    _wf_section = _wf_header + _wf_body + "\n\n"
                    if _wf_section not in workflow_ctx:
                        workflow_ctx = (workflow_ctx + "\n\n" + _wf_section).strip()
                    # 把该 workflow 的依赖技能也加入展开池
                    for _sn in _wf_snames:
                        if _sn not in workflow_skill_names:
                            workflow_skill_names.append(_sn)
    except Exception:
        pass

    # 将 workflow 依赖技能强制加入 hits（未出现的才加，保留原顺序优先级）
    if workflow_skill_names:
        skill_map = {s["name"]: s for s in _get_skills()}
        existing_names = {s["name"] for s in hits}
        for wf_skill_name in workflow_skill_names:
            if wf_skill_name not in existing_names and wf_skill_name in skill_map:
                hits.append(skill_map[wf_skill_name])
                existing_names.add(wf_skill_name)

    # ── 3. 构建技能文档文本 ──────────────────────────────────────────────────
    skill_ctx = ""
    if hits:
        parts: List[str] = ["## 相关技能文档（请优先使用以下函数和示例）\n"]
        total = len(parts[0])
        for skill in hits:
            if skill["source"] == "user":
                tag = "【自定义】"
            elif skill["source"] == "generated":
                tag = "【文档生成】"
            else:
                tag = ""
            section = f"### 技能：{tag}{skill['name']}\n\n{skill['body']}\n\n"
            if total + len(section) > max_skill_chars:
                remaining = max_skill_chars - total - 20
                if remaining > 200:
                    parts.append(section[:remaining] + "\n...(已截断)\n")
                break
            parts.append(section)
            total += len(section)
        skill_ctx = "".join(parts).strip()

    # ── 4. 合并 workflow 引导文本 ────────────────────────────────────────────
    if workflow_ctx:
        skill_ctx = (skill_ctx + "\n\n" + workflow_ctx).strip() if skill_ctx else workflow_ctx

    # ── 5. RAG 检索 ──────────────────────────────────────────────────────────
    has_rag_backed = any(s.get("rag_sources") for s in hits)
    rag_ctx = ""

    if has_rag_backed:
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _web = _Path(__file__).parent.parent / "web_app"
            if str(_web) not in _sys.path:
                _sys.path.insert(0, str(_web))
            from rag_engine import get_knowledge_base
            kb = get_knowledge_base()
            rag_ctx = kb.build_rag_context(
                query,
                top_k=5,
                max_chars=max_rag_chars,
                score_threshold=0.5,
            )
        except Exception:
            pass

    return skill_ctx, rag_ctx


# ── 模板 ────────────────────────────────────────────────────────────────────

SKILL_TEMPLATE = """\
---
name: {name}
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
