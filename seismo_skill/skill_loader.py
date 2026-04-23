"""
skill_loader.py — seismo_skill 技能文档加载与检索

从两个目录读取 Markdown 格式的技能说明文件：
  1. seismo_skill/          内置技能（随项目发布）
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

# 内置技能目录（本文件所在目录）
_BUILTIN_SKILL_DIR = Path(__file__).parent

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
        entry = {
            "name": meta.get("name", md_file.stem),
            "category": meta.get("category", "custom" if source == "user" else ""),
            "keywords": meta.get("keywords", []),
            "path": str(md_file),
            "body": body.strip(),
            "full_text": text.strip(),
            "source": source,   # "builtin" | "user"
            "filename": md_file.name,
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

    # 用用户技能覆盖同名内置技能
    user_names = {s["name"] for s in user}
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
        }
        for s in _get_skills()
    ]


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
    删除用户自定义技能（只能删除 user source）。

    返回
    ----
    bool — True 表示删除成功，False 表示未找到该技能或它是内置技能
    """
    for skill in _get_skills():
        if skill["name"] == name and skill["source"] == "user":
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

    返回
    ----
    str — 可直接拼入系统提示的技能文档片段；无匹配时返回空字符串
    """
    hits = search_skills(query, top_k=top_k)
    if not hits:
        return ""

    parts: List[str] = ["## 相关技能文档（请优先使用以下函数和示例）\n"]
    total = len(parts[0])

    for skill in hits:
        tag = "【自定义】" if skill["source"] == "user" else ""
        section = f"### 技能：{tag}{skill['name']}\n\n{skill['body']}\n\n"
        if total + len(section) > max_chars:
            remaining = max_chars - total - 20
            if remaining > 200:
                parts.append(section[:remaining] + "\n...(已截断)\n")
            break
        parts.append(section)
        total += len(section)

    return "".join(parts).strip()


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
