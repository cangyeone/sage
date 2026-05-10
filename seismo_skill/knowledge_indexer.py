"""
knowledge_indexer.py — seismo_skill/knowledge/ 目录文档索引器

功能
----
1. 扫描 seismo_skill/knowledge/ 下的文档（多层级目录，支持 PDF/DOC/DOCX/TXT/MD/RST/HTML）
2. 维护 manifest（seismo_rag/dir_manifest.json）检测新增/修改/删除
3. 每个顶级子文件夹作为一个"项目"，统一索引为 RAG，并可生成一个文件夹型 Skill
4. knowledge/ 根目录下的单个文件也会作为独立文档项处理，并可生成一个文件夹型 Skill
5. 支持中断后继续（manifest 按文件粒度记录，skill 按项目粒度生成）

项目文件夹约定
--------------
knowledge/
├── GMT_docs-6.5/        ← 整个文件夹 = 一个项目 → 生成 1 个 Skill
│   ├── source/
│   │   ├── tutorial/
│   │   │   └── *.md
│   │   └── ...
│   └── ...
├── SeisPy_docs/         ← 另一个项目
└── some_paper.pdf       ← 根目录文件 → 生成 1 个 Skill（原有行为）
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── 常量 ───────────────────────────────────────────────────────────────────────

# 支持的文档格式
SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".txt", ".md", ".rst", ".html", ".htm"}

# Skill Builder Agent 可读取的源码/示例格式。RAG 索引仍只使用 SUPPORTED_EXTS；
# 这里额外纳入脚本，是为了让 LLM 能从文档示例中整理出可执行子技能。
SKILL_SOURCE_EXTS = SUPPORTED_EXTS | {
    ".sh", ".bash", ".zsh", ".csh", ".py", ".m", ".jl", ".r",
    ".gmt", ".cpt", ".dat", ".csv", ".tsv", ".sty", ".tex", ".bib",
}

CODE_EXAMPLE_EXTS = {
    ".sh", ".bash", ".zsh", ".csh", ".py", ".m", ".jl", ".r",
    ".gmt", ".cpt", ".dat", ".csv", ".tsv", ".sty", ".tex", ".bib",
}

# 跳过格式（二进制、脚本、图片等不适合 RAG 的文件）
SKIP_EXTS = {
    ".sh", ".py", ".js", ".css", ".yml", ".yaml", ".json", ".xml",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".eps", ".ps",
    ".sty", ".cls", ".tex", ".bib", ".po", ".mo",
    ".zip", ".tar", ".gz", ".DS_Store", ".gitkeep",
    ".dat", ".gmt", ".sgy", ".sac", ".bin",
}

# knowledge/ 目录（本文件所在 seismo_skill/ 目录下）
_DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent / "docs"  # 技能文档目录

# manifest 存储在 RAG 知识库目录（与 faiss_index.bin 同级）
_KB_DIR = Path(__file__).parent.parent / "seismo_rag"
_MANIFEST_FILE = _KB_DIR / "dir_manifest.json"

# 项目级 manifest 存储（folder → skill 映射）
_PROJ_MANIFEST_FILE = _KB_DIR / "proj_manifest.json"

# 项目内用户 Skill 存储目录。不要写入 ~/.seismicx/skills，便于项目清理和隔离。
_USER_SKILL_DIR = Path(__file__).parent / "user_skills"
_BUILTIN_SKILL_DIR = Path(__file__).parent / "skills"
_DOC_SKILL_GENERATOR = "seismo_skill_docs_builder"
_SKILL_BUILDER_CACHE_DIR = _KB_DIR / "skill_builder_cache"
_MARKDOWN_NORMALIZER_VERSION = "v2"

def _env_optional_int(name: str) -> Optional[int]:
    try:
        value = int(os.environ.get(name, "").strip())
        return value if value > 0 else None
    except Exception:
        return None


# 默认不限制文件数。GMT/ObsPy 这类手册通常有数百个页面，扫描阶段不应先砍掉文档；
# 如需临时限流，可设置 SEISMICX_MAX_RAG_FILES_PER_PROJECT / SEISMICX_MAX_SKILL_FILES_PER_PROJECT。
MAX_RAG_FILES_PER_PROJECT = _env_optional_int("SEISMICX_MAX_RAG_FILES_PER_PROJECT")
MAX_SKILL_FILES_PER_PROJECT = _env_optional_int("SEISMICX_MAX_SKILL_FILES_PER_PROJECT")
MAX_FILES_PER_PROJECT = MAX_RAG_FILES_PER_PROJECT


# ── 文件优先级与智能选取 ──────────────────────────────────────────────────────

def _file_priority(path: Path) -> Tuple[int, int]:
    """
    返回文件的索引优先级 (tier, depth)，越小越优先。
    tier:  0=README/index  1=section-level md/rst  2=txt  3=pdf  4=docx  5=html
    depth: 目录深度（同 tier 时越浅越优先）
    """
    name_lower = path.name.lower()
    ext = path.suffix.lower()
    depth = len(path.parts)

    if name_lower in ("readme.md", "readme.rst", "readme.txt",
                      "index.md", "index.rst", "readme"):
        return (0, depth)
    if ext in (".md", ".rst"):
        return (1, depth)
    if ext in (".txt", ".text"):
        return (2, depth)
    if ext == ".pdf":
        return (3, depth)
    if ext == ".docx":
        return (4, depth)
    if ext in (".html", ".htm"):
        return (5, depth)
    if ext in (".sh", ".bash", ".zsh", ".csh", ".py", ".m", ".jl", ".r"):
        return (6, depth)
    if ext in (".gmt", ".cpt", ".dat", ".csv", ".tsv"):
        return (7, depth)
    return (9, depth)


def _select_key_files(files: List[Path], max_count: Optional[int] = MAX_RAG_FILES_PER_PROJECT) -> List[Path]:
    """
    从文件列表中智能选取最有价值的文件，最多 max_count 个。

    策略：
    - README / index 文件全部保留
    - 按优先级 + 目录深度排序
    - 深度较浅（章节级）的文件优先于叶子页
    - 若文件数 <= max_count，全部保留
    """
    return _select_link_aware_files(files, max_count=max_count)


def _select_skill_builder_files(files: List[Path], max_count: Optional[int] = MAX_SKILL_FILES_PER_PROJECT) -> List[Path]:
    """Select a balanced directory digest for LLM skill synthesis."""
    return _select_link_aware_files(files, max_count=max_count)


def _select_link_aware_files(files: List[Path], max_count: Optional[int] = MAX_RAG_FILES_PER_PROJECT) -> List[Path]:
    """
    Select files for RAG/Skill Builder while preserving documentation links.

    Entry pages such as index.rst/readme.md are kept, and their linked documents
    are pulled in before the balanced per-directory sampling step. This prevents
    a docs tree from degenerating into only index pages.
    """
    sorted_files = sorted(files, key=_file_priority)
    if not sorted_files:
        return []
    effective_max = max_count if max_count and max_count > 0 else len(sorted_files)

    try:
        source_root = Path(os.path.commonpath([str(p.parent) for p in sorted_files])).resolve()
    except Exception:
        source_root = sorted_files[0].parent.resolve()

    selected: List[Path] = []
    seen: set[str] = set()
    file_set = {p.resolve() for p in sorted_files}
    entry_names = {"readme.md", "readme.rst", "readme.txt", "index.md", "index.rst", "readme"}
    entry_budget = max(8, effective_max // 4)
    entry_added = 0

    def add(path: Path):
        nonlocal entry_added
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.name.lower() in entry_names and entry_added >= entry_budget:
            return
        key = str(path)
        if key not in seen and len(selected) < effective_max:
            selected.append(path)
            seen.add(key)
            if path.name.lower() in entry_names:
                entry_added += 1

    entry_files = [
        path for path in sorted_files
        if path.name.lower() in entry_names
    ]
    queue: List[Path] = []
    visited_links: set[str] = set()

    def drain_queue(max_steps: int):
        steps = 0
        while queue and len(selected) < effective_max and steps < max_steps:
            steps += 1
            cur = queue.pop(0)
            cur_key = str(cur.resolve())
            if cur_key in visited_links:
                continue
            visited_links.add(cur_key)
            text = _read_source_text(cur, max_chars=50000)
            for ref in _extract_referenced_source_paths(cur, source_root, text):
                if ref.resolve() not in file_set:
                    continue
                before = len(selected)
                add(ref)
                if len(selected) > before and ref.suffix.lower() in {".md", ".rst", ".txt", ".html", ".htm"}:
                    queue.append(ref)
                if len(selected) >= effective_max:
                    break

    for entry in entry_files:
        add(entry)
        if len(selected) >= effective_max:
            break
        text = _read_source_text(entry, max_chars=50000)
        direct_refs = _extract_referenced_source_paths(entry, source_root, text)
        for ref in direct_refs:
            if ref.resolve() not in file_set:
                continue
            before = len(selected)
            add(ref)
            if len(selected) > before and ref.suffix.lower() in {".md", ".rst", ".txt", ".html", ".htm"}:
                queue.append(ref)
            if len(selected) >= effective_max:
                break
        drain_queue(max_steps=8)

    drain_queue(max_steps=effective_max)

    # Keep representative pages from every documentation sub-area instead of
    # letting one large directory consume the whole context budget.
    buckets: Dict[str, List[Path]] = {}
    for path in sorted_files:
        rel_parts = path.parts
        bucket = rel_parts[-2] if len(rel_parts) >= 2 else "."
        buckets.setdefault(bucket, []).append(path)

    for _round in range(8):
        for bucket in sorted(buckets):
            candidates = buckets[bucket]
            if _round < len(candidates):
                add(candidates[_round])

    for path in sorted_files:
        add(path)
    return selected


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    new: List[Path] = field(default_factory=list)
    modified: List[Path] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)   # rel_path strings
    unchanged: List[Path] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)    # previously failed

    # 项目级（文件夹）
    new_projects: List[str] = field(default_factory=list)       # folder names
    updated_projects: List[str] = field(default_factory=list)   # folder names (some files changed)

    @property
    def pending_count(self) -> int:
        return len(self.new) + len(self.modified) + len(self.failed)

    def summary(self) -> str:
        parts = []
        if self.new:
            parts.append(f"{len(self.new)} 个新文件")
        if self.modified:
            parts.append(f"{len(self.modified)} 个已修改")
        if self.failed:
            parts.append(f"{len(self.failed)} 个上次失败")
        if self.deleted:
            parts.append(f"{len(self.deleted)} 个已删除")
        return "，".join(parts) if parts else "无待处理文档"


@dataclass
class BuildResult:
    indexed: List[str] = field(default_factory=list)       # rel_paths
    skills_generated: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    interrupted: bool = False

    def summary(self) -> str:
        parts = [f"已索引 {len(self.indexed)} 个文件"]
        if self.skills_generated:
            parts.append(f"生成 {len(self.skills_generated)} 个 Skill")
        if self.skipped:
            parts.append(f"跳过 {len(self.skipped)} 个")
        if self.failed:
            parts.append(f"失败 {len(self.failed)} 个")
        if self.interrupted:
            parts.append("（已中断，进度已保存）")
        return "，".join(parts)


def _source_manifest_for_skill(files: List[Path], source_root: Path) -> Dict[str, dict]:
    """Return lightweight fingerprints for incremental docs-to-SKILL builds."""
    out: Dict[str, dict] = {}
    for path in files:
        try:
            stat = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            out[str(_safe_relpath(path, source_root))] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": h.hexdigest(),
            }
        except Exception:
            continue
    return out


# ── 主类 ──────────────────────────────────────────────────────────────────────

class KnowledgeIndexer:
    """
    seismo_skill/knowledge/ 目录文档扫描与 RAG 索引构建器。

    Parameters
    ----------
    knowledge_dir : Path | str | None
        要扫描的目录。None 时使用 seismo_skill/knowledge/（默认）。
    """

    def __init__(self, knowledge_dir: Optional[Path] = None, manifest_dir: Optional[Path] = None):
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else _DEFAULT_KNOWLEDGE_DIR
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        _USER_SKILL_DIR.mkdir(parents=True, exist_ok=True)
        # 支持自定义 manifest 存储目录（用于 ref_knowledge 等独立索引场景）
        _mdir = Path(manifest_dir) if manifest_dir else _KB_DIR
        _mdir.mkdir(parents=True, exist_ok=True)
        self._manifest_file = _mdir / "dir_manifest.json"
        self._proj_manifest_file = _mdir / "proj_manifest.json"
        self._manifest: Dict[str, dict] = self._load_manifest()
        self._proj_manifest: Dict[str, dict] = self._load_proj_manifest()

    # ── Manifest I/O ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> Dict[str, dict]:
        if self._manifest_file.exists():
            try:
                return json.loads(self._manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_manifest(self):
        self._manifest_file.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_proj_manifest(self) -> Dict[str, dict]:
        if self._proj_manifest_file.exists():
            try:
                return json.loads(self._proj_manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_proj_manifest(self):
        self._proj_manifest_file.write_text(
            json.dumps(self._proj_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 文件指纹 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(path: Path) -> tuple:
        """返回 (mtime, size, sha256_prefix)。"""
        stat = path.stat()
        data = path.read_bytes()[:16384]  # 只哈希前 16KB
        sha = hashlib.sha256(data).hexdigest()[:16]
        return stat.st_mtime, stat.st_size, sha

    def _is_changed(self, path: Path, entry: dict) -> bool:
        mtime, size, sha = self._fingerprint(path)
        return not (
            abs(mtime - entry.get("mtime", 0)) < 1.0
            and size == entry.get("size", -1)
            and sha == entry.get("sha256", "")
        )

    # ── 文件发现 ──────────────────────────────────────────────────────────────

    def _iter_supported_files(self, root: Path) -> List[Path]:
        """递归遍历 root，返回支持格式的文件列表（按优先级排序）。"""
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue
            if path.name.startswith(".") or path.name in (".gitkeep",):
                continue
            files.append(path)
        return sorted(files, key=_file_priority)

    def _iter_skill_source_files(self, root: Path) -> List[Path]:
        """递归遍历 root，返回 Skill Builder 可读取的文档和示例源码。"""
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith(".") or path.name in (".gitkeep",):
                continue
            if any(part.startswith(("_build", ".git", "__pycache__")) for part in path.parts):
                continue
            ext = path.suffix.lower()
            if ext not in SKILL_SOURCE_EXTS:
                continue
            files.append(path)
        return sorted(files, key=_file_priority)

    # ── 扫描 ──────────────────────────────────────────────────────────────────

    def scan(self) -> ScanResult:
        """
        扫描 knowledge_dir，返回变更情况。不修改任何文件。

        策略
        ----
        - knowledge/ 根目录直接文件 → 逐文件比较
        - knowledge/ 子文件夹（项目）→ 检查其下所有文件的变更

        注意：即使 manifest 标记为 "indexed"，若 KB 中实际不存在该 doc_id
        （如 KB 被清空或重启后 stale 清理），也会被视为 "new" 重新索引。
        """
        result = ScanResult()
        current_rel: set = set()

        # 一次性获取 KB 中现有 doc_id 集合，用于验证 manifest 与 KB 是否同步
        try:
            _live_doc_ids: Optional[set] = set(_get_kb()._docs.keys())
        except Exception:
            _live_doc_ids = None   # 无法访问 KB，跳过验证

        # 获取所有顶级项（文件 + 子目录）
        try:
            top_items = sorted(self.knowledge_dir.iterdir())
        except Exception:
            return result

        for item in top_items:
            if item.name.startswith(".") or item.name == ".gitkeep":
                continue

            if item.is_file():
                # 根目录直接文件 → 逐文件处理
                ext = item.suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                rel = str(item.relative_to(self.knowledge_dir))
                current_rel.add(rel)
                entry = self._manifest.get(rel)
                if entry is None:
                    result.new.append(item)
                elif entry.get("status") == "failed":
                    result.failed.append(rel)
                elif entry.get("status") != "indexed":
                    result.new.append(item)
                elif self._is_changed(item, entry):
                    result.modified.append(item)
                else:
                    # 验证 doc 是否真的存在于 KB（防止 KB 被清空后 manifest 仍标记 indexed）
                    doc_id = entry.get("doc_id", "")
                    if _live_doc_ids is not None and doc_id and doc_id not in _live_doc_ids:
                        result.new.append(item)
                    else:
                        result.unchanged.append(item)

            elif item.is_dir():
                # 子目录 → 项目模式（只处理智能选取后的关键文件）
                proj_name = item.name
                all_proj_files = self._iter_supported_files(item)
                proj_files = _select_key_files(all_proj_files)
                has_new = False

                for path in proj_files:
                    rel = str(path.relative_to(self.knowledge_dir))
                    current_rel.add(rel)
                    entry = self._manifest.get(rel)

                    if entry is None:
                        result.new.append(path)
                        has_new = True
                    elif entry.get("status") == "failed":
                        result.failed.append(rel)
                        has_new = True
                    elif entry.get("status") != "indexed":
                        result.new.append(path)
                        has_new = True
                    elif self._is_changed(path, entry):
                        result.modified.append(path)
                        has_new = True
                    else:
                        # 验证 doc 是否真的存在于 KB（防止 KB 被清空后 manifest 仍标记 indexed）
                        doc_id = entry.get("doc_id", "")
                        if _live_doc_ids is not None and doc_id and doc_id not in _live_doc_ids:
                            result.new.append(path)
                            has_new = True
                        else:
                            result.unchanged.append(path)

                # Track project-level changes
                proj_entry = self._proj_manifest.get(proj_name)
                if proj_entry is None:
                    result.new_projects.append(proj_name)
                elif has_new:
                    result.updated_projects.append(proj_name)

        # 已删除的文件
        for rel in self._manifest:
            if rel not in current_rel:
                result.deleted.append(rel)

        return result

    # ── 构建 ─────────────────────────────────────────────────────────────────

    def build(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
        stop_event=None,
        skip_skill_gen: bool = False,
    ) -> BuildResult:
        """
        对 scan() 返回的所有 pending 文件依次索引，并为每个项目文件夹生成一个 Skill。

        支持 Ctrl+C（KeyboardInterrupt）或 stop_event.set() 中断；
        中断后已完成文件的进度持久化，下次可跳过。
        """
        def _log(msg: str):
            if progress_cb:
                progress_cb(msg)

        scan = self.scan()
        result = BuildResult()

        # 清理已删除文件
        if scan.deleted:
            _log(f"🗑  清理 {len(scan.deleted)} 个已删除文档的索引…")
            self._cleanup_deleted(scan.deleted)

        pending: List[Path] = (
            scan.new
            + scan.modified
            + [self.knowledge_dir / r for r in scan.failed]
        )

        if not pending:
            _log("✅ 所有文档均已是最新，无需重新索引。")
            return result

        _log(f"📂 共 {len(pending)} 个文件待索引"
             f"（新增 {len(scan.new)}，修改 {len(scan.modified)}，重试 {len(scan.failed)}）")

        # 按项目分组，便于后续生成项目级 Skill
        proj_touched: Dict[str, List[str]] = {}  # folder_name → [rel_paths indexed]

        for i, path in enumerate(pending, 1):
            if stop_event and stop_event.is_set():
                _log("⚠  已中断（进度已保存）。")
                result.interrupted = True
                break

            rel = str(path.relative_to(self.knowledge_dir))
            _log(f"\n[{i}/{len(pending)}] {rel}")

            # 确定项目归属
            parts = Path(rel).parts
            proj_folder = parts[0] if len(parts) > 1 else None

            try:
                doc_id, first_chunks = self._index_file(
                    path, _log,
                    proj_folder=proj_folder or "",
                    source_type="skill_docs" if proj_folder else "skill_docs",
                )

                mtime, size, sha = self._fingerprint(path)
                self._manifest[rel] = {
                    "rel_path": rel,
                    "abs_path": str(path),
                    "mtime": mtime,
                    "size": size,
                    "sha256": sha,
                    "doc_id": doc_id,
                    "skill_name": "",   # Skill 在项目级生成
                    "indexed_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "indexed",
                    "error": "",
                    "proj_folder": proj_folder or "",
                }
                self._save_manifest()
                result.indexed.append(rel)

                if proj_folder:
                    proj_touched.setdefault(proj_folder, []).append(rel)

            except KeyboardInterrupt:
                _log("\n⚠  用户中断（Ctrl+C）。进度已保存。")
                result.interrupted = True
                break

            except Exception as exc:
                err = str(exc)
                _log(f"   ❌ 失败：{err}")
                self._manifest[rel] = {
                    "rel_path": rel, "abs_path": str(path),
                    "mtime": 0, "size": 0, "sha256": "",
                    "doc_id": "", "skill_name": "",
                    "indexed_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed", "error": err,
                    "proj_folder": proj_folder or "",
                }
                self._save_manifest()
                result.failed.append(rel)

        if result.interrupted:
            return result

        # ── 生成项目级 Skill ──────────────────────────────────────────────────
        if not skip_skill_gen:
            # 生成 Skill 的项目：本次触碰的 + 已有文件但没有 Skill 的
            needs_skill = set(proj_touched.keys())
            # 补充：已有 indexed 文件但从未生成 Skill 的项目
            for rel, entry in self._manifest.items():
                pf = entry.get("proj_folder")
                if pf and pf not in self._proj_manifest:
                    needs_skill.add(pf)
            # 根目录文件（proj_folder=""）逐文件生成 Skill
            for rel, entry in self._manifest.items():
                if not entry.get("proj_folder") and entry.get("status") == "indexed":
                    if not entry.get("skill_name"):
                        path = Path(entry["abs_path"])
                        kb = _get_kb()
                        first_chunks = [
                            c.text for c in kb._chunks.values()
                            if c.doc_id == entry["doc_id"]
                        ][:5]
                        skill_name = self._generate_file_skill(path, entry["doc_id"], first_chunks, _log)
                        entry["skill_name"] = skill_name
                        self._save_manifest()
                        if skill_name:
                            result.skills_generated.append(skill_name)

            for proj_name in needs_skill:
                proj_path = self.knowledge_dir / proj_name
                if not proj_path.is_dir():
                    continue
                _log(f"\n📝 为项目「{proj_name}」生成 Skill…")
                skill_name = self._generate_project_skill(proj_name, proj_path, _log)
                if skill_name:
                    result.skills_generated.append(skill_name)
                    self._proj_manifest[proj_name] = {
                        "proj_name": proj_name,
                        "skill_name": skill_name,
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    self._save_proj_manifest()

        _log(f"\n✅ 完成：{result.summary()}")
        return result

    # ── RAG 索引单个文件 ──────────────────────────────────────────────────────

    def _index_file(self, path: Path, log: Callable, proj_folder: str = "",
                    source_type: str = "skill_docs"):
        """将单个文件添加到 RAG 知识库，返回 (doc_id, first_chunks_text)。"""
        kb = _get_kb()
        meta = kb.add_document(str(path), progress_cb=log,
                               proj_folder=proj_folder, source_type=source_type)
        first_chunks = [
            c.text for c in list(kb._chunks.values())
            if c.doc_id == meta.doc_id
        ][:5]
        return meta.doc_id, first_chunks

    # ── 项目级 Skill 生成 ────────────────────────────────────────────────────

    def _generate_project_skill(
        self,
        proj_name: str,
        proj_path: Path,
        log: Callable,
    ) -> str:
        """
        为整个项目文件夹生成一个 RAG 增强型 Skill。
        使用 LLM 决定技能名称和简介；如果 LLM 不可用则回退到 slug 命名。
        """
        # 收集该项目下的已索引文件和它们的 chunk 样本
        indexed_files = [
            v for v in self._manifest.values()
            if v.get("proj_folder") == proj_name and v.get("status") == "indexed"
        ]
        if not indexed_files:
            return ""

        # 抽取代表性内容：从前 3 个文件各取第一个 chunk
        kb = _get_kb()
        sample_chunks: List[str] = []
        for entry in indexed_files[:3]:
            doc_id = entry.get("doc_id", "")
            chunks = [c.text for c in kb._chunks.values() if c.doc_id == doc_id][:2]
            sample_chunks.extend(chunks)

        # 尝试 LLM 命名
        skill_name, title, description = _llm_name_project(
            proj_name=proj_name,
            file_count=len(indexed_files),
            sample_chunks=sample_chunks,
        )

        keywords = _extract_keywords(proj_name, sample_chunks)
        doc_names = ", ".join(
            Path(v["rel_path"]).name for v in indexed_files[:10]
        )
        if len(indexed_files) > 10:
            doc_names += f"… 共 {len(indexed_files)} 个文件"

        preview = (sample_chunks[0][:500].strip() if sample_chunks else "（无预览）")

        # 自动检测关联内置技能，写入 related_skills 字段
        related = _find_related_builtin_skills(keywords, top_n=3)
        log(f"   🔗 关联内置技能：{related or '（无）'}")

        text = _PROJ_SKILL_TEMPLATE.format(
            name=skill_name,
            title=title,
            keywords=", ".join(keywords),
            proj_name=proj_name,
            file_count=len(indexed_files),
            doc_names=doc_names,
            description=description,
            preview=preview,
            related_skills=", ".join(related),
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )

        skill_path = _USER_SKILL_DIR / f"{skill_name}.md"
        skill_path.write_text(text, encoding="utf-8")
        _invalidate_skill_cache()
        log(f"   ✅ 项目 Skill 已生成：{skill_name}")
        return skill_name

    # ── 根目录文件级 Skill 生成 ──────────────────────────────────────────────

    def _generate_file_skill(
        self,
        path: Path,
        doc_id: str,
        first_chunks: List[str],
        log: Callable,
    ) -> str:
        """为根目录下的单个文件生成 Skill（原有行为）。"""
        stem = re.sub(r"[^\w]", "_", path.stem.lower()).strip("_")
        skill_name = f"_gen_{stem}"
        keywords = _extract_keywords(path.stem, first_chunks)
        rel_path = str(path.relative_to(self.knowledge_dir.parent)
                       if path.is_relative_to(self.knowledge_dir.parent)
                       else path)
        preview = (first_chunks[0][:400].strip() if first_chunks else "（无预览）")

        text = _FILE_SKILL_TEMPLATE.format(
            name=skill_name,
            title=path.stem.replace("_", " ").replace("-", " "),
            keywords=", ".join(keywords),
            doc_name=path.name,
            rel_path=rel_path,
            preview=preview,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )
        skill_path = _USER_SKILL_DIR / f"{skill_name}.md"
        skill_path.write_text(text, encoding="utf-8")
        _invalidate_skill_cache()
        log(f"   📝 已生成 Skill：{skill_name}")
        return skill_name

    # ── 文件夹型 Skill 生成 ──────────────────────────────────────────────────

    def build_folder_skills(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
        stop_event=None,
        use_llm: bool = True,
        overwrite: bool = True,
        rag_assist: bool = True,
        rag_cluster_target: int = 0,
    ) -> BuildResult:
        """
        将 seismo_skill/docs/ 下的每个顶级文档项转换为一个文件夹型 Skill。

        顶级文档项包括：
        - 子文件夹：一个文件夹 = 一个 Skill，内部可包含多层文档；
        - 根目录单文件：一个 PDF/MD/DOCX/TXT/HTML 文件 = 一个 Skill。

        目标格式：
            seismo_skill/user_skills/<skill_name>/SKILL.md
            seismo_skill/user_skills/<skill_name>/references/*.md
            seismo_skill/user_skills/<skill_name>/agents/openai.yaml

        这个路径把文档固化为可被其它 Skill 系统复用的通用技能包。
        RAG/embedding 只作为辅助：用于向量化判断相似片段是否应合并到
        同一个子技能，而不是作为最终产物类型。若同名技能已经存在且不是
        本构建器生成的，会自动追加后缀，避免覆盖手写用户技能。
        """
        def _log(msg: str):
            if progress_cb:
                progress_cb(msg)

        result = BuildResult()
        _USER_SKILL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            projects = [
                p for p in sorted(self.knowledge_dir.iterdir())
                if (
                    not p.name.startswith(".")
                    and p.name != ".gitkeep"
                    and (
                        p.is_dir()
                        or (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
                    )
                )
            ]
        except Exception as exc:
            result.failed.append(str(exc))
            _log(f"❌ 无法扫描文档目录：{exc}")
            return result

        if not projects:
            _log("ℹ️  seismo_skill/docs 下暂无支持的文档文件或文件夹，未生成文件夹型 Skill。")
            return result

        assist_label = "开启" if rag_assist else "关闭"
        _log(f"📦 将 {len(projects)} 个文档项转换为 OpenAI-style 文件夹 Skill…")
        _log(f"   🔎 RAG/向量辅助：{assist_label}")
        if rag_assist:
            cluster_hint = rag_cluster_target if rag_cluster_target > 0 else "自动建议"
            _log(f"   🧭 目标主题簇数：{cluster_hint}")
        for idx, proj_path in enumerate(projects, 1):
            if stop_event and stop_event.is_set():
                result.interrupted = True
                _log("⚠  已中断（文件夹型 Skill 构建停止）。")
                break

            is_single_file = proj_path.is_file()
            source_root = proj_path.parent if is_single_file else proj_path
            proj_name = proj_path.stem if is_single_file else proj_path.name
            _log(f"\n[{idx}/{len(projects)}] {proj_name}")
            all_source_files = [proj_path] if is_single_file else self._iter_skill_source_files(proj_path)
            files = [proj_path] if is_single_file else _select_skill_builder_files(all_source_files)
            if not files:
                result.skipped.append(proj_name)
                _log("   ⚠ 未发现支持的文档文件，跳过。")
                continue

            source_manifest = _source_manifest_for_skill(files, source_root)
            proj_entry = self._proj_manifest.get(proj_name, {})
            existing_skill = str(proj_entry.get("skill_name") or "").strip()
            existing_path = Path(str(proj_entry.get("skill_path") or "")) if proj_entry.get("skill_path") else None
            if (
                existing_skill
                and existing_path
                and (existing_path / "SKILL.md").exists()
                and proj_entry.get("generated_by") == _DOC_SKILL_GENERATOR
                and proj_entry.get("source_manifest") == source_manifest
            ):
                result.skipped.append(proj_name)
                _log(f"   ✅ SKILL 已是最新，跳过增量重建：{existing_skill}")
                continue

            docs = []
            max_docs = len(files)
            if is_single_file:
                _log("   📚 收集目录证据：1 个文件")
            else:
                _log(f"   📚 收集目录证据：选择 {max_docs}/{len(all_source_files)} 个候选文件")
            for doc_i, path in enumerate(files[:max_docs], 1):
                if stop_event and stop_event.is_set():
                    result.interrupted = True
                    _log("⚠  已中断（文件夹型 Skill 构建停止）。")
                    break
                if doc_i == 1 or doc_i % 20 == 0 or doc_i == max_docs:
                    _log(f"   · 读取证据 {doc_i}/{max_docs}: {_safe_relpath(path, source_root)}")
                # Skill generation should digest the source documents, not just
                # keep a pointer to the original PDFs. Single-file skills get a
                # larger text budget so a whole paper/manual can be converted
                # into hierarchical Markdown references.
                text_budget = 140000 if is_single_file else 18000
                text = _read_source_text(path, max_chars=text_budget)
                if text.strip() and not is_single_file:
                    text = _append_referenced_code(path, source_root, text, max_chars=26000)
                quality_issue = _source_text_quality_issue(path, text)
                if quality_issue:
                    _log(f"   ⚠ 跳过 {path.name}：{quality_issue}")
                    if path.suffix.lower() == ".pdf":
                        _log("      建议提供可复制文本版 PDF、Markdown/HTML 原文，或先进行中文 OCR 后再构建 SKILL。")
                    continue
                if not text.strip():
                    continue
                if use_llm:
                    rel_for_log = _safe_relpath(path, source_root)
                    cache_hit = _markdown_cache_file(path, text).exists()
                    _log(
                        f"   🤖 Markdown 标准化 {doc_i}/{max_docs}: {rel_for_log}"
                        + ("（缓存）" if cache_hit else "")
                    )
                    text = _llm_convert_to_markdown(path, text, max_chars=text_budget) or text
                docs.append({
                    "path": str(_safe_relpath(path, source_root)),
                    "abs_path": str(path),
                    "text": text,
                    "headings": _extract_headings(text),
                })

            if result.interrupted:
                break

            if not docs:
                result.skipped.append(proj_name)
                _log("   ⚠ 文档无法抽取文本，跳过。")
                continue

            try:
                spec = _llm_folder_skill_spec(proj_name, docs) if use_llm else {}
                if not spec:
                    spec = _fallback_folder_skill_spec(proj_name, docs)
                skill_name = _generated_skill_slug(spec.get("name") or proj_name)
                target = _resolve_generated_skill_target(skill_name, overwrite=overwrite)
                final_name = target.name
                spec["name"] = final_name
                self._write_folder_skill(
                    target,
                    proj_name,
                    source_root,
                    files,
                    docs,
                    spec,
                    _log,
                    use_llm=use_llm,
                    rag_assist=rag_assist,
                    rag_cluster_target=rag_cluster_target,
                )
                self._proj_manifest[proj_name] = {
                    "proj_name": proj_name,
                    "skill_name": final_name,
                    "skill_path": str(target),
                    "skill_kind": "single_file" if is_single_file else "folder",
                    "skill_location": "user",
                    "generated_by": _DOC_SKILL_GENERATOR,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "source_path": str(proj_path),
                    "source_manifest": source_manifest,
                }
                self._save_proj_manifest()
                result.skills_generated.append(final_name)
                _log(f"   ✅ OpenAI-style 文件夹 Skill 已生成：{final_name}")
            except Exception as exc:
                result.failed.append(proj_name)
                _log(f"   ❌ 生成失败：{exc}")

        _invalidate_skill_cache()
        _log(f"\n✅ 完成：{result.summary()}")
        return result

    def _write_folder_skill(
        self,
        target: Path,
        proj_name: str,
        proj_path: Path,
        files: List[Path],
        docs: List[dict],
        spec: dict,
        log: Callable[[str], None],
        use_llm: bool = True,
        rag_assist: bool = True,
        rag_cluster_target: int = 0,
    ) -> None:
        work_target = target.with_name(f".{target.name}.building")
        if work_target.exists():
            shutil.rmtree(work_target, ignore_errors=True)
        (work_target / "references").mkdir(parents=True, exist_ok=True)
        (work_target / "subskills").mkdir(parents=True, exist_ok=True)
        (work_target / "agents").mkdir(parents=True, exist_ok=True)

        ref_lines = [
            f"# 转换清单：{proj_name}",
            "",
            f"- Generated by: `{_DOC_SKILL_GENERATOR}`",
            f"- Source root: `{proj_path}`",
            f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Selected files: {len(files)}",
            "",
            "## 已转换文件",
        ]
        for path in files:
            rel = _safe_relpath(path, proj_path)
            ref_lines.append(f"- `{rel}`")
            ref_lines.append(f"  - Absolute path: `{path}`")
        try:
            (work_target / "references" / "manifest.md").write_text("\n".join(ref_lines) + "\n", encoding="utf-8")

            skill_plan = {}
            if use_llm:
                log("   🧠 Skill Builder Agent：读取文档并规划层级 SKILL 结构…")
                skill_plan = _llm_hierarchical_skill_plan(
                    proj_name,
                    docs,
                    rag_assist=rag_assist,
                    rag_cluster_target=rag_cluster_target,
                    log=log,
                ) if docs else {}
                if skill_plan.get("_error"):
                    log(f"   ❌ Skill Builder Agent 失败：{skill_plan.get('_error')}")
                    if skill_plan.get("_raw_preview"):
                        log(f"   🧾 LLM 原始输出预览：{skill_plan.get('_raw_preview')}")
            outline_lines = [f"# {proj_name} 层级技能资料", ""]
            ref_count = 0

            raw_units = skill_plan.get("subskills") or skill_plan.get("references") or []
            planned_units = _filter_usable_subskills(_merge_similar_subskills(raw_units, log=log, use_embedding=rag_assist))
            if use_llm and not planned_units:
                detail = skill_plan.get("_error") if isinstance(skill_plan, dict) else ""
                raise RuntimeError(
                    "Skill Builder Agent 未能生成有效功能型子技能；已停止，避免生成不可用的机械切分 SKILL。"
                    + (f"原因：{detail}。" if detail else "")
                    + "请检查 LLM 配置或换用更强模型后重试。"
                )
            if planned_units:
                outline_lines.append("## Skill Builder Agent 规划的子技能")
                for unit_idx, unit in enumerate(planned_units[:80], 1):
                    ref_count += 1
                    title = str(unit.get("title_zh") or unit.get("title") or f"章节 {unit_idx}").strip()
                    slug = _ascii_slug(str(unit.get("slug") or ""), fallback="")
                    if not slug:
                        slug = _fallback_english_slug(str(unit.get("title_en") or title), fallback=f"subskill_{unit_idx}")
                    ref_name = f"{unit_idx:02d}_{slug}.md"
                    outline_lines.append(f"- `subskills/{ref_name}`：{title}")
                    body = _render_llm_subskill(unit)
                    (work_target / "subskills" / ref_name).write_text(body, encoding="utf-8")
                outline_lines.append("")
            else:
                log("   ⚠ LLM 规划失败，回退为功能聚类拆分。")
                fallback_units = []
                for doc_idx, doc in enumerate(docs[:60], 1):
                    units = _split_document_into_skill_units(doc["path"], doc["text"])
                    for unit_idx, unit in enumerate(units, 1):
                        unit.setdefault("source_evidence", [f"{doc['path']}：{unit.get('title_zh') or unit.get('title') or unit_idx}"])
                        fallback_units.append(unit)
                fallback_units = _filter_usable_subskills(_merge_similar_subskills(fallback_units, log=log, use_embedding=rag_assist))
                outline_lines.append("## 功能聚类生成的子技能")
                if not fallback_units:
                    outline_lines.append("- 未能抽取有效内容")
                for unit_idx, unit in enumerate(fallback_units[:80], 1):
                    ref_count += 1
                    title = str(unit.get("title_zh") or unit.get("title") or f"子技能 {unit_idx}").strip()
                    slug = _ascii_slug(str(unit.get("slug") or ""), fallback="")
                    if not slug:
                        slug = _fallback_english_slug(str(unit.get("title_en") or title), fallback=f"subskill_{unit_idx}")
                    ref_name = f"{unit_idx:02d}_{slug}.md"
                    outline_lines.append(f"- `subskills/{ref_name}`：{title}")
                    body = _render_converted_reference("merged_sources", unit)
                    (work_target / "subskills" / ref_name).write_text(body, encoding="utf-8")
                outline_lines.append("")
            (work_target / "references" / "outline.md").write_text("\n".join(outline_lines).strip() + "\n", encoding="utf-8")
            log(f"   🧩 已生成 {ref_count} 个子技能 Markdown 文档")

            if skill_plan:
                spec = _merge_hierarchical_plan_into_spec(spec, skill_plan)
                audit = _skill_builder_audit(skill_plan, ref_count)
                (work_target / "references" / "builder_audit.md").write_text(audit, encoding="utf-8")
            skill_md = _render_folder_skill_md(proj_name, spec, docs)
            (work_target / "SKILL.md").write_text(skill_md, encoding="utf-8")
            (work_target / "agents" / "openai.yaml").write_text(_render_agent_yaml(spec), encoding="utf-8")

            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(work_target), str(target))
            log(f"   📁 写入：{target}")
        except Exception:
            shutil.rmtree(work_target, ignore_errors=True)
            raise

    # ── 清理已删除文件 ────────────────────────────────────────────────────────

    def _cleanup_deleted(self, deleted_rels: List[str]):
        kb = _get_kb()
        deleted_proj_folders: set = set()

        for rel in deleted_rels:
            entry = self._manifest.pop(rel, None)
            if entry and entry.get("doc_id"):
                try:
                    kb.delete_doc(entry["doc_id"])
                except Exception:
                    pass
            if entry and entry.get("proj_folder"):
                deleted_proj_folders.add(entry["proj_folder"])
            # 根目录文件的独立 skill
            if entry and entry.get("skill_name") and not entry.get("proj_folder"):
                sp = _USER_SKILL_DIR / f"{entry['skill_name']}.md"
                sp.unlink(missing_ok=True)

        # 如果某个项目的所有文件都被删除，删除项目 Skill
        for proj_name in deleted_proj_folders:
            still_has_files = any(
                v.get("proj_folder") == proj_name
                for v in self._manifest.values()
            )
            if not still_has_files:
                proj_entry = self._proj_manifest.pop(proj_name, None)
                if proj_entry and proj_entry.get("skill_name"):
                    sp = _USER_SKILL_DIR / f"{proj_entry['skill_name']}.md"
                    sp.unlink(missing_ok=True)
                    delete_generated_builtin_skill(proj_entry["skill_name"])
                self._save_proj_manifest()

        self._save_manifest()
        _invalidate_skill_cache()

    # ── 状态查询 ─────────────────────────────────────────────────────────────

    def manifest_summary(self) -> dict:
        """
        返回项目粒度的状态摘要，供 API 和 UI 使用。

        projects 列表每项对应一个子文件夹（或根目录文件），格式：
        {
            name:         文件夹名（或文件名）
            is_folder:    True/False
            total_files:  该项目下支持格式文件总数
            selected_files: 实际选取用于索引的文件数
            indexed_files:  已索引文件数
            failed_files:   失败文件数
            status:       "new" | "partial" | "indexed" | "modified"
            skill_name:   已生成的 Skill 名（空字符串表示未生成）
            skill_generated_at: 生成时间
        }
        """
        scan = self.scan()

        # 把 scan 结果按项目分桶
        modified_rels = {str(p.relative_to(self.knowledge_dir)) for p in scan.modified}
        new_rels      = {str(p.relative_to(self.knowledge_dir)) for p in scan.new}

        projects: List[dict] = []

        try:
            top_items = sorted(self.knowledge_dir.iterdir())
        except Exception:
            top_items = []

        for item in top_items:
            if item.name.startswith(".") or item.name == ".gitkeep":
                continue

            if item.is_file():
                ext = item.suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                rel = str(item.relative_to(self.knowledge_dir))
                entry = self._manifest.get(rel, {})
                proj_entry = self._proj_manifest.get(item.stem, {})
                source_manifest = _source_manifest_for_skill([item], item.parent)
                skill_name = entry.get("skill_name", "") or proj_entry.get("skill_name", "")
                skill_current = bool(
                    skill_name
                    and proj_entry.get("generated_by") == _DOC_SKILL_GENERATOR
                    and proj_entry.get("source_manifest") == source_manifest
                )
                if skill_current and entry.get("status") != "indexed":
                    status = "skill"
                elif rel in new_rels:
                    status = "new"
                elif rel in modified_rels:
                    status = "modified"
                elif entry.get("status") == "indexed":
                    status = "indexed"
                elif entry.get("status") == "failed":
                    status = "failed"
                else:
                    status = "new"
                projects.append({
                    "name": item.name,
                    "is_folder": False,
                    "total_files": 1,
                    "selected_files": 1,
                    "indexed_files": 1 if entry.get("status") == "indexed" else 0,
                    "failed_files": 1 if entry.get("status") == "failed" else 0,
                    "status": status,
                    "skill_name": skill_name,
                    "skill_generated_at": proj_entry.get("generated_at", ""),
                    "files": [{
                        "name": item.name,
                        "rel_path": rel,
                        "status": entry.get("status", "new"),
                        "doc_id": entry.get("doc_id", ""),
                    }],
                })

            elif item.is_dir():
                all_files = self._iter_supported_files(item)
                selected  = _select_key_files(all_files)
                sel_rels  = {str(p.relative_to(self.knowledge_dir)) for p in selected}

                indexed = sum(1 for r in sel_rels
                              if self._manifest.get(r, {}).get("status") == "indexed")
                failed  = sum(1 for r in sel_rels
                              if self._manifest.get(r, {}).get("status") == "failed")
                has_new = any(r in new_rels or r in modified_rels for r in sel_rels)
                proj_entry = self._proj_manifest.get(item.name, {})
                source_manifest = _source_manifest_for_skill(selected, item)
                skill_current = bool(
                    proj_entry.get("skill_name")
                    and proj_entry.get("generated_by") == _DOC_SKILL_GENERATOR
                    and proj_entry.get("source_manifest") == source_manifest
                )

                if skill_current and indexed == 0 and failed == 0:
                    status = "skill"
                elif failed > 0 and indexed == 0 and not has_new:
                    status = "failed"
                elif indexed == 0:
                    status = "new"
                elif has_new:
                    status = "modified" if indexed > 0 else "new"
                elif indexed < len(selected):
                    status = "partial"
                else:
                    status = "indexed"

                # 收集该文件夹下各文件的状态，供前端展开显示
                folder_files = []
                for p in selected:
                    r = str(p.relative_to(self.knowledge_dir))
                    e = self._manifest.get(r, {})
                    folder_files.append({
                        "name": p.name,
                        "rel_path": r,
                        "status": e.get("status", "new"),
                        "doc_id": e.get("doc_id", ""),
                    })
                projects.append({
                    "name": item.name,
                    "is_folder": True,
                    "total_files": len(all_files),
                    "selected_files": len(selected),
                    "indexed_files": indexed,
                    "failed_files": failed,
                    "status": status,
                    "skill_name": proj_entry.get("skill_name", ""),
                    "skill_generated_at": proj_entry.get("generated_at", ""),
                    "files": folder_files,
                })

        total_indexed = sum(1 for v in self._manifest.values() if v.get("status") == "indexed")
        total_failed  = sum(1 for v in self._manifest.values() if v.get("status") == "failed")
        pending_projects = sum(1 for p in projects if p["status"] in ("new", "partial", "modified", "failed"))

        return {
            "knowledge_dir": str(self.knowledge_dir),
            "total_indexed_files": total_indexed,
            "total_failed_files": total_failed,
            "pending_new": len(scan.new),
            "pending_modified": len(scan.modified),
            "pending_total": scan.pending_count,
            "pending_projects": pending_projects,
            "projects": projects,
            # Legacy fields for homepage stats
            "indexed": total_indexed,
            "failed": total_failed,
        }


# ── LLM 项目命名 ──────────────────────────────────────────────────────────────

def _llm_name_project(
    proj_name: str,
    file_count: int,
    sample_chunks: List[str],
) -> Tuple[str, str, str]:
    """
    使用 LLM 为项目文件夹生成 Skill 名称、标题和一句话描述。
    返回 (skill_name, title, description)。
    如果 LLM 不可用，回退到基于文件夹名的 slug。
    """
    _slug = re.sub(r"[^\w]", "_", proj_name.lower()).strip("_")
    fallback_name  = f"_gen_{_slug}"
    fallback_title = proj_name.replace("-", " ").replace("_", " ")
    fallback_desc  = f"{proj_name} 文档，共 {file_count} 个文件，支持 RAG 增强检索。"

    preview = "\n".join(sample_chunks[:2])[:600] if sample_chunks else ""
    prompt = (
        f"我有一个文档项目文件夹，名称为「{proj_name}」，"
        f"包含 {file_count} 个技术文档文件。\n"
        f"文档内容预览（前几段）：\n{preview}\n\n"
        "请用英文生成：\n"
        "1. skill_name：Python 变量风格的小写下划线名称（以 _gen_ 开头），最多 40 字符，"
        "要能准确体现这是什么工具/库的文档，如 _gen_gmt_map_drawing\n"
        "2. title：简洁的中文标题，5~15 字，如「GMT 地图绘制工具文档」\n"
        "3. description：一句话中文描述，不超过 50 字\n\n"
        "严格按以下格式回复，不要额外内容：\n"
        "skill_name: <value>\n"
        "title: <value>\n"
        "description: <value>"
    )

    try:
        # 动态导入 LLM 客户端
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from llm_client import get_llm_client
        client = get_llm_client()
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        text = response.strip()
        parsed = {}
        for line in text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                parsed[k.strip()] = v.strip()

        skill_name  = parsed.get("skill_name", fallback_name)
        title       = parsed.get("title", fallback_title)
        description = parsed.get("description", fallback_desc)

        # 安全校验 skill_name
        skill_name = re.sub(r"[^\w]", "_", skill_name.lower()).strip("_")
        if not skill_name.startswith("_gen_"):
            skill_name = f"_gen_{skill_name}"
        if len(skill_name) > 60:
            skill_name = skill_name[:60].rstrip("_")

        return skill_name, title, description

    except Exception:
        return fallback_name, fallback_title, fallback_desc


# ── 文件夹型 Skill 辅助函数 ───────────────────────────────────────────────────

def _safe_skill_slug(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (name or "").strip().lower()).strip("_-")
    if not slug:
        slug = "document_skill"
    if slug[0].isdigit():
        slug = f"skill_{slug}"
    return slug[:64].rstrip("_-") or "document_skill"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", (name or "reference").strip().lower()).strip("_-")[:80] or "reference"


def _ascii_slug(text: str, fallback: str = "subskill") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (slug[:72].rstrip("_") or fallback)


_CN_SLUG_TERMS = {
    "线条": "line",
    "颜色": "color",
    "宽度": "width",
    "样式": "style",
    "背景": "background",
    "填充": "fill",
    "字体": "font",
    "标注": "annotation",
    "投影": "projection",
    "区域": "region",
    "图层": "layer",
    "顺序": "order",
    "色标": "colorbar",
    "符号": "symbol",
    "震相": "phase",
    "拾取": "picking",
    "波形": "waveform",
    "滤波": "filtering",
    "地形": "terrain",
    "地图": "map",
    "绘图": "plotting",
    "控制": "control",
}


def _fallback_english_slug(text: str, fallback: str = "subskill") -> str:
    raw = str(text or "")
    parts = []
    for cn, en in _CN_SLUG_TERMS.items():
        if cn in raw and en not in parts:
            parts.append(en)
    ascii_bits = re.findall(r"[A-Za-z0-9]+", raw)
    parts.extend(bit.lower() for bit in ascii_bits if bit.lower() not in parts)
    return _ascii_slug("_".join(parts), fallback=fallback)


def _generated_skill_slug(name: str, fallback: str = "document_skill") -> str:
    slug = _ascii_slug(name, fallback=fallback)
    if slug.startswith("gen_"):
        slug = f"_{slug}"
    elif not slug.startswith("_gen_"):
        slug = f"_gen_{slug}"
    return slug[:72].rstrip("_") or "_gen_document_skill"


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        try:
            return path.relative_to(root)
        except Exception:
            return Path(path.name)


def _read_source_text(path: Path, max_chars: int = 8000) -> str:
    """Best-effort text extraction for docs-to-skill generation."""
    ext = path.suffix.lower()
    try:
        if ext in {".md", ".txt", ".rst", ".sty", ".tex", ".bib"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        if ext in {".sh", ".bash", ".zsh", ".csh", ".py", ".m", ".jl", ".r", ".gmt", ".cpt", ".dat", ".csv", ".tsv"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        if ext in {".html", ".htm"}:
            return _strip_html(path.read_text(encoding="utf-8", errors="ignore"))[:max_chars]
        if ext == ".pdf":
            return _read_pdf_text_best_effort(path, max_chars=max_chars)
        if ext == ".doc":
            try:
                proc = subprocess.run(
                    ["textutil", "-convert", "txt", "-stdout", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=25,
                )
                if proc.stdout.strip():
                    return proc.stdout[:max_chars]
            except Exception:
                return ""
        if ext == ".docx":
            try:
                from docx import Document  # type: ignore
                document = Document(str(path))
                return "\n".join(p.text for p in document.paragraphs)[:max_chars]
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


def _append_referenced_code(path: Path, source_root: Path, text: str, max_chars: int = 26000) -> str:
    refs = [p for p in _extract_referenced_source_paths(path, source_root, text) if p.suffix.lower() in CODE_EXAMPLE_EXTS]
    if not refs:
        return text[:max_chars]
    parts = [text]
    used = len(text)
    for ref in refs[:8]:
        if used >= max_chars:
            break
        try:
            code = ref.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not code:
            continue
        remaining = max_chars - used
        excerpt = code[: max(0, remaining - 120)]
        if not excerpt:
            break
        rel = _safe_relpath(ref, source_root)
        block = f"\n\n[Referenced example code: {rel}]\n```{_code_fence_lang(ref)}\n{excerpt}\n```"
        parts.append(block)
        used += len(block)
    return "\n".join(parts)[:max_chars]


def _extract_referenced_source_paths(path: Path, source_root: Path, text: str) -> List[Path]:
    refs: List[Path] = []
    patterns = [
        r"\.\.\s+(?:literalinclude|include|gmtplot)::\s+([^\s]+)",
        r"\.\.\s+toctree::\s*\n(?P<items>(?:\s+[^:\n][^\n]*\n?)+)",
        r":doc:`[^`<]*<?([^`<>]+)>?`",
        r":download:`[^`<]*<?([^`<>]+)>?`",
        r":file:`([^`]+)`",
        r"\(([^)]+\.(?:pdf|docx|html|htm|txt|md|rst|sh|py|gmt|cpt|dat|csv|tsv))\)",
        r"`([^`]+?\.(?:pdf|docx|html|htm|txt|md|rst|sh|py|gmt|cpt|dat|csv|tsv))`",
    ]
    for pat in patterns:
        if "?P<items>" in pat:
            matches = [m.group("items") for m in re.finditer(pat, text, flags=re.MULTILINE)]
        else:
            matches = re.findall(pat, text)
        for raw in matches:
            names = _reference_names_from_raw(raw)
            for name in names:
                candidates = _reference_candidates(path, source_root, name)
                for cand in candidates:
                    try:
                        cand.relative_to(source_root.resolve())
                    except Exception:
                        continue
                    if cand.exists() and cand.is_file() and cand.suffix.lower() in SKILL_SOURCE_EXTS and cand not in refs:
                        refs.append(cand)
    return refs


def _reference_names_from_raw(raw) -> List[str]:
    names: List[str] = []
    raw_text = str(raw or "")
    for line in raw_text.splitlines() or [raw_text]:
        name = line.strip().strip("<>").strip()
        if not name or name.startswith(("http://", "https://", "@", ":")):
            continue
        if name.startswith(".. ") or name.startswith("#"):
            continue
        if " <" in name and name.endswith(">"):
            name = name.rsplit("<", 1)[-1].rstrip(">").strip()
        if name:
            names.append(name)
    return names


def _reference_candidates(path: Path, source_root: Path, name: str) -> List[Path]:
    raw = str(name).strip()
    if not raw:
        return []
    raw = raw.split("#", 1)[0].split("?", 1)[0].strip()
    raw = raw.lstrip("/")
    bases = [(path.parent / raw), (source_root / raw)]
    candidates: List[Path] = []
    for base in bases:
        candidates.append(base.resolve())
        if base.suffix:
            continue
        for ext in (".rst", ".md", ".txt", ".html", ".htm", ".pdf", ".docx"):
            candidates.append(base.with_suffix(ext).resolve())
        for index_name in ("index.rst", "index.md", "README.md", "readme.md"):
            candidates.append((base / index_name).resolve())
    deduped: List[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            deduped.append(cand)
            seen.add(key)
    return deduped


def _code_fence_lang(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".csh": "bash",
        ".m": "matlab",
        ".jl": "julia",
        ".r": "r",
        ".csv": "csv",
        ".tsv": "tsv",
    }.get(ext, "text")


def _read_pdf_text_best_effort(path: Path, max_chars: int = 8000) -> str:
    candidates: List[str] = []

    try:
        from pdfminer.high_level import extract_text  # type: ignore
        candidates.append(extract_text(str(path), maxpages=24)[:max_chars])
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "24", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.stdout:
            candidates.append(proc.stdout[:max_chars])
    except Exception:
        pass

    try:
        import fitz  # type: ignore
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc[: min(24, len(doc))]]
        doc.close()
        candidates.append("\n".join(pages)[:max_chars])
    except Exception:
        pass

    candidates = [c for c in candidates if c and c.strip()]
    if not candidates:
        return ""

    clean = [c for c in candidates if not _source_text_quality_issue(path, c)]
    if clean:
        return max(clean, key=_source_text_score)
    return max(candidates, key=_source_text_score)


def _source_text_score(text: str) -> float:
    total = max(len(text), 1)
    alnum = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    cid = len(re.findall(r"\(cid:\d+\)", text))
    weird = len(re.findall(r"[\u0700-\u0fff\u1200-\u137f]", text))
    replacement = text.count("\ufffd")
    dot_ratio = text.count(".") / total
    score = (alnum / total) + min(cjk / 800, 0.25)
    score -= min(cid / 80, 0.8)
    score -= min(weird / 200, 0.8)
    score -= min(replacement / 20, 0.5)
    score -= max(0.0, dot_ratio - 0.18)
    return score


def _source_text_quality_issue(path: Path, text: str) -> str:
    if not text or len(text.strip()) < 300:
        # Short RST/Markdown index pages and small scripts are still useful to
        # the directory-level Skill Builder Agent because they expose toctrees,
        # command names, examples, and cross references. Treat short text as a
        # hard failure only for single-document binary formats.
        if path.suffix.lower() in {".pdf", ".docx"}:
            return "未能抽取到足够正文文本"
        return ""

    total = max(len(text), 1)
    cid = len(re.findall(r"\(cid:\d+\)", text))
    weird = len(re.findall(r"[\u0700-\u0fff\u1200-\u137f]", text))
    replacement = text.count("\ufffd")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    alnum = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text))
    dot_ratio = text.count(".") / total
    name_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", path.name))

    if cid >= 20 or cid / total > 0.002:
        return "PDF 文本抽取出现大量 CID 占位符，疑似字体编码缺失"
    if weird >= 40 or weird / total > 0.008:
        return "PDF 文本抽取出现大量异常 Unicode 字符，疑似中文字体错码"
    if replacement >= 10:
        return "文本抽取包含大量无法解码字符"
    if name_has_cjk and total > 2000 and cjk < 180:
        return "文件名显示为中文文档，但抽取正文中的中文字符过少"
    if dot_ratio > 0.28 and (alnum / total) < 0.32:
        return "抽取结果主要像目录点线或残缺文本，正文密度不足"
    return ""


def _extract_headings(text: str, limit: int = 12) -> List[str]:
    headings: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            h = s.lstrip("#").strip()
        elif len(s) < 90 and re.match(r"^[A-Z][A-Za-z0-9 .:_/-]{4,}$", s):
            h = s
        else:
            continue
        if h and h not in headings:
            headings.append(h)
        if len(headings) >= limit:
            break
    return headings


def _json_from_text(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _llm_folder_skill_spec(proj_name: str, docs: List[dict]) -> dict:
    snippets = []
    for doc in docs[:10]:
        headings = ", ".join(doc.get("headings") or [])
        snippets.append(
            f"FILE: {doc['path']}\nHEADINGS: {headings}\nTEXT:\n{doc['text'][:1600]}"
        )
    prompt = f"""You convert a documentation folder into a portable OpenAI/Codex-style Skill.

Folder name: {proj_name}

Use this RAW TEXT format. Do not output JSON.

SKILL_NAME: lowercase_machine_safe_name
DISPLAY_NAME: human readable bilingual name
DESCRIPTION: trigger description. Say when this skill should be used.
KEYWORDS: keyword1, keyword2, keyword3
WHEN_TO_USE:
- concrete situation
WORKFLOW:
- concise step
VALIDATION:
- check or test the agent should run
EXAMPLE_PROMPTS:
- realistic user prompt

Make the skill generic and reusable by other systems. Do not mention RAG as a requirement.

Documentation excerpts:
{chr(10).join(snippets)[:16000]}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        raw = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=1200,
        )
        parsed = _raw_folder_skill_spec_from_text(raw)
        if not parsed:
            parsed = _json_from_text(raw)
        return _normalize_folder_skill_spec(parsed, proj_name, docs)
    except Exception:
        return {}


def _raw_folder_skill_spec_from_text(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    out: dict = {}
    field_map = {
        "SKILL_NAME": "name",
        "DISPLAY_NAME": "display_name",
        "DESCRIPTION": "description",
    }
    for raw_key, key in field_map.items():
        m = re.search(rf"(?im)^\s*{raw_key}\s*[:：]\s*(.+)$", text)
        if m:
            out[key] = m.group(1).strip()
    for raw_key, key in {
        "KEYWORDS": "keywords",
        "WHEN_TO_USE": "when_to_use",
        "WORKFLOW": "workflow",
        "VALIDATION": "validation",
        "EXAMPLE_PROMPTS": "example_prompts",
    }.items():
        m = re.search(rf"(?ims)^\s*{raw_key}\s*[:：]\s*(.*?)(?=^\s*[A-Z_]+\s*[:：]|\Z)", text)
        if m:
            out[key] = _parse_raw_list(m.group(1))
    return out


def _fallback_folder_skill_spec(proj_name: str, docs: List[dict]) -> dict:
    keywords = _extract_keywords(proj_name, [d["text"][:1200] for d in docs[:5]])
    title = proj_name.replace("_", " ").replace("-", " ").strip() or proj_name
    return {
        "name": _generated_skill_slug(_fallback_english_slug(proj_name, fallback="document_skill")),
        "display_name": title,
        "description": (
            f"Use this skill when working with {title} documentation, APIs, workflows, "
            "examples, command patterns, or domain methods from the bundled references."
        ),
        "keywords": keywords,
        "when_to_use": [
            "The user asks how to use this documented tool, method, workflow, or domain package.",
            "The task needs examples, parameters, file formats, or implementation guidance from the references.",
            "The answer should be grounded in local documentation instead of generic memory.",
        ],
        "workflow": [
            "Read SKILL.md first to understand the scope.",
            "Load only the relevant files under references/ for the user request.",
            "Extract concrete commands, APIs, parameters, assumptions, and constraints from the references.",
            "If code is needed, produce a minimal runnable implementation and include a small self-check.",
            "Cite the reference file names used in the final answer when possible.",
        ],
        "validation": [
            "Check referenced commands or code snippets for syntax before presenting them.",
            "Do not invent unavailable APIs, parameters, or results.",
            "State missing documentation or uncertainty explicitly.",
        ],
        "example_prompts": [
            f"How do I use {title} for a practical task?",
            f"Summarize the key workflow from the {title} docs.",
        ],
    }


def _normalize_folder_skill_spec(spec: dict, proj_name: str, docs: List[dict]) -> dict:
    fallback = _fallback_folder_skill_spec(proj_name, docs)
    if not isinstance(spec, dict):
        return fallback
    out = {**fallback}
    for key in ("name", "display_name", "description"):
        if isinstance(spec.get(key), str) and spec[key].strip():
            out[key] = spec[key].strip()
    for key in ("keywords", "when_to_use", "workflow", "validation", "example_prompts"):
        val = spec.get(key)
        if isinstance(val, list):
            cleaned = [str(v).strip() for v in val if str(v).strip()]
            if cleaned:
                out[key] = cleaned[:14]
    ascii_name = _ascii_slug(str(out.get("name") or ""), fallback="")
    if not ascii_name:
        ascii_name = _fallback_english_slug(
            str(out.get("display_name_en") or out.get("display_name") or proj_name),
            fallback="document_skill",
        )
    out["name"] = _generated_skill_slug(ascii_name)
    return out


def _doc_digest_for_llm(docs: List[dict], max_chars: int = 42000) -> str:
    parts = []
    used = 0
    for doc in _rank_docs_for_skill_digest(docs)[:18]:
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        commands = _extract_command_like_items(text, limit=18)
        params = _extract_parameter_like_items(text, limit=18)
        headings = doc.get("headings") or []
        preface = [
            f"文件：{doc.get('path')}",
            f"标题线索：{', '.join(headings[:8]) if headings else '（无）'}",
        ]
        if commands:
            preface.append("命令/代码线索：" + "; ".join(commands[:10]))
        if params:
            preface.append("参数线索：" + "; ".join(params[:10]))
        preface_text = "\n".join(preface) + "\n\n"
        remaining = max_chars - used - len(preface_text)
        if remaining <= 0:
            break
        excerpt = _condense_rst_for_skill_builder(text[:remaining])
        used += len(excerpt)
        parts.append(preface_text + excerpt)
    return "\n\n---\n\n".join(parts)


def _rank_docs_for_skill_digest(docs: List[dict]) -> List[dict]:
    def score(doc: dict) -> Tuple[int, int]:
        path = str(doc.get("path") or "").lower()
        text = str(doc.get("text") or "").lower()
        s = 0
        if "index" in Path(path).name:
            s += 8
        if any(key in path for key in ("tutorial", "examples", "dataset", "cpt", "module", "proj")):
            s += 6
        if any(key in path for key in ("coast", "grdview", "grdimage", "makecpt", "colorbar", "basemap", "plot", "text", "legend")):
            s += 8
        s += min(len(_extract_command_like_items(text, limit=20)), 8)
        return (-s, len(path))
    return sorted(docs, key=score)


def _embedding_merge_hints(docs: List[dict], max_docs: int = 36, top_pairs: int = 18) -> str:
    ranked = _rank_docs_for_skill_digest(docs)[:max_docs]
    if len(ranked) < 2:
        return ""
    texts = []
    labels = []
    for doc in ranked:
        text = str(doc.get("text") or "")
        summary = "\n".join([
            str(doc.get("path") or ""),
            " ".join(doc.get("headings") or []),
            " ".join(_extract_command_like_items(text, limit=12)),
            " ".join(_extract_parameter_like_items(text, limit=12)),
            text[:900],
        ])
        texts.append(summary)
        labels.append(str(doc.get("path") or ""))

    pairs: List[Tuple[float, str, str]] = []
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from rag_backends import EmbeddingModel  # type: ignore
        vecs = EmbeddingModel.get().encode(texts, batch_size=16)
        sims = vecs @ vecs.T
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                score = float(sims[i, j])
                if score >= 0.58:
                    pairs.append((score, labels[i], labels[j]))
    except Exception:
        pairs = _lexical_merge_hints(texts, labels)

    if not pairs:
        return ""
    lines = []
    for score, a, b in sorted(pairs, reverse=True)[:top_pairs]:
        lines.append(f"- {score:.2f}: `{a}` ↔ `{b}`")
    return "\n".join(lines)


def _suggest_skill_cluster_target(n_docs: int) -> int:
    """Suggest a practical topic-cluster count for large documentation sets."""
    if n_docs <= 0:
        return 0
    if n_docs <= 24:
        return max(3, min(8, n_docs))
    # sqrt-like growth keeps LLM calls bounded while preserving major topics.
    try:
        import math
        suggested = int(round(math.sqrt(n_docs) * 1.25))
    except Exception:
        suggested = 20
    return max(8, min(36, suggested))


def _skill_doc_semantic_text(doc: dict) -> str:
    text = str(doc.get("text") or "")
    parts = [
        str(doc.get("path") or ""),
        " ".join(doc.get("headings") or []),
        " ".join(_extract_command_like_items(text, limit=18)),
        " ".join(_extract_parameter_like_items(text, limit=18)),
        text[:2200],
    ]
    return "\n".join(p for p in parts if p).strip()[:5000]


def _semantic_doc_batches_for_skill_builder(
    docs: List[dict],
    rag_assist: bool,
    fallback_batch_size: int,
    cluster_target: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Group docs into LLM batches; use embedding + DBSCAN for large sets."""
    docs = [d for d in docs if isinstance(d, dict)]
    if not docs:
        return []
    fallback_batch_size = max(4, fallback_batch_size)
    if not rag_assist or len(docs) <= fallback_batch_size:
        return [
            {"label": f"batch_{i // fallback_batch_size + 1}", "docs": docs[i:i + fallback_batch_size]}
            for i in range(0, len(docs), fallback_batch_size)
        ]

    target = cluster_target if cluster_target and cluster_target > 0 else _suggest_skill_cluster_target(len(docs))
    try:
        clusters = _dbscan_cluster_docs(docs, target_clusters=target)
    except Exception:
        clusters = []

    if not clusters:
        if log:
            log("   ⚠ DBSCAN 文档聚类不可用，回退为顺序分批。")
        return [
            {"label": f"batch_{i // fallback_batch_size + 1}", "docs": docs[i:i + fallback_batch_size]}
            for i in range(0, len(docs), fallback_batch_size)
        ]

    batches: List[dict] = []
    try:
        max_docs_per_cluster = int(os.environ.get("SEISMICX_SKILL_MAX_DOCS_PER_CLUSTER", "36"))
    except Exception:
        max_docs_per_cluster = 36
    max_docs_per_cluster = max(fallback_batch_size, max_docs_per_cluster)
    for idx, cluster_docs in enumerate(clusters, 1):
        if len(cluster_docs) <= max_docs_per_cluster:
            label = _cluster_label_from_docs(cluster_docs, idx)
            batches.append({"label": label, "docs": cluster_docs})
            continue
        for part_no, start in enumerate(range(0, len(cluster_docs), max_docs_per_cluster), 1):
            chunk = cluster_docs[start:start + max_docs_per_cluster]
            label = f"{_cluster_label_from_docs(chunk, idx)} part {part_no}"
            batches.append({"label": label, "docs": chunk})

    if log:
        log(f"   🔎 DBSCAN 文档聚类：{len(docs)} 个文档 → {len(clusters)} 个主题簇，LLM 批次 {len(batches)} 个（目标 {target}）")
    return batches


def _dbscan_cluster_docs(docs: List[dict], target_clusters: int = 20) -> List[List[dict]]:
    if len(docs) <= 1:
        return [docs]
    texts = [_skill_doc_semantic_text(doc) for doc in docs]
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from rag_backends import EmbeddingModel  # type: ignore
        import numpy as np  # type: ignore

        vecs = EmbeddingModel.get().encode(texts, batch_size=16)
        vecs = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        sims = vecs @ vecs.T
    except Exception:
        return []

    target_clusters = max(2, min(max(2, len(docs)), int(target_clusters or _suggest_skill_cluster_target(len(docs)))))
    best_labels = None
    best_score = 10**9
    # eps is cosine distance; larger eps merges more documents.
    for eps in [0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.56, 0.62]:
        labels = _dbscan_labels_from_similarity(sims, eps=eps, min_samples=2)
        n_clusters = len({label for label in labels if label >= 0})
        n_noise = sum(1 for label in labels if label < 0)
        effective = n_clusters + max(1, (n_noise + 7) // 8 if n_noise else 0)
        score = abs(effective - target_clusters) + 0.05 * n_noise
        if score < best_score:
            best_labels = labels
            best_score = score

    if best_labels is None:
        return []

    grouped: Dict[int, List[dict]] = {}
    noise: List[dict] = []
    order: List[int] = []
    for doc, label in zip(docs, best_labels):
        if label < 0:
            noise.append(doc)
            continue
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        grouped[label].append(doc)

    clusters = [grouped[label] for label in order if grouped.get(label)]
    # Preserve noise instead of discarding rare but useful topics.
    if noise:
        chunk_size = max(4, min(10, (len(noise) + max(1, target_clusters // 2) - 1) // max(1, target_clusters // 2)))
        clusters.extend(noise[i:i + chunk_size] for i in range(0, len(noise), chunk_size))

    return [cluster for cluster in clusters if cluster]


def _dbscan_labels_from_similarity(sims, eps: float, min_samples: int = 2) -> List[int]:
    threshold = 1.0 - float(eps)
    n = int(getattr(sims, "shape", [0])[0])
    labels = [-99] * n
    cluster_id = 0

    def neighbors(i: int) -> List[int]:
        return [j for j in range(n) if float(sims[i, j]) >= threshold]

    for i in range(n):
        if labels[i] != -99:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = [j for j in nbrs if j != i]
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                continue
            labels[j] = cluster_id
            nbrs_j = neighbors(j)
            if len(nbrs_j) >= min_samples:
                for k in nbrs_j:
                    if labels[k] in {-99, -1} and k not in seeds:
                        seeds.append(k)
        cluster_id += 1
    return labels


def _cluster_label_from_docs(docs: List[dict], idx: int) -> str:
    tokens: Dict[str, int] = {}
    for doc in docs[:12]:
        text = _skill_doc_semantic_text(doc).lower()
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", text):
            if tok in {"the", "and", "for", "with", "文件", "标题线索"}:
                continue
            tokens[tok] = tokens.get(tok, 0) + 1
    top = [k for k, _ in sorted(tokens.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
    return f"cluster_{idx}: " + (", ".join(top) if top else "misc")


def _lexical_merge_hints(texts: List[str], labels: List[str]) -> List[Tuple[float, str, str]]:
    def toks(s: str) -> set:
        return set(re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", s.lower()))
    token_sets = [toks(t) for t in texts]
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            score = len(a & b) / max(1, len(a | b))
            if score >= 0.12:
                pairs.append((score, labels[i], labels[j]))
    return pairs


def _condense_rst_for_skill_builder(text: str) -> str:
    """Keep references and examples useful, but mark Sphinx syntax as structure."""
    lines = []
    in_toctree = False
    toctree_items = []
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```{toctree}") or stripped.startswith(".. toctree::"):
            in_toctree = True
            toctree_items = []
            continue
        if in_toctree:
            if not stripped:
                continue
            if stripped.startswith(":"):
                continue
            if stripped.startswith("```"):
                if toctree_items:
                    lines.append("文档目录线索: " + ", ".join(toctree_items[:40]))
                in_toctree = False
                continue
            if not stripped.startswith(".. "):
                toctree_items.append(stripped)
            continue
        line = re.sub(r":doc:`([^`<]+)<([^`>]+)>`", r"\1 (\2)", line)
        line = re.sub(r":doc:`([^`]+)`", r"\1", line)
        line = re.sub(r":file:`([^`]+)`", r"`\1`", line)
        line = re.sub(r":download:`([^`<]+)<([^`>]+)>`", r"\1 (\2)", line)
        line = re.sub(r":download:`([^`]+)`", r"\1", line)
        line = re.sub(r":gmt-docs:`([^`<]+)<([^`>]+)>`", r"\1 (\2)", line)
        if stripped.startswith(".. hlist::") or stripped.startswith(".. only::"):
            continue
        lines.append(line)
    if in_toctree and toctree_items:
        lines.append("文档目录线索: " + ", ".join(toctree_items[:40]))
    return "\n".join(lines).strip()


def _llm_convert_to_markdown(path: Path, text: str, max_chars: int = 18000) -> str:
    """Use LLM to normalize mixed RST/HTML/extracted text into clean Markdown."""
    source = str(text or "").strip()
    if not source:
        return ""
    cache_file = _markdown_cache_file(path, source)
    if cache_file.exists():
        try:
            cached = cache_file.read_text(encoding="utf-8")
            if cached.strip():
                return cached[:max_chars]
        except Exception:
            pass
    # Already-clean Markdown/code files can skip the extra call unless they
    # contain Sphinx directives that confuse downstream skill synthesis.
    if path.suffix.lower() in {".md", ".txt"} and not re.search(r"```\\{|\.\.\s+\w+::|:\w+:`", source):
        return source[:max_chars]

    excerpt = source[: min(len(source), 12000)]
    prompt = f"""请把下面的文档片段转换为干净、可读、可被二次处理的 Markdown。

只做格式标准化，不要总结，不要扩写，不要添加原文没有的信息。

要求：
1. 保留标题、说明、参数、命令、代码块、示例、输入输出。
2. 把 RST/Sphinx 语法转换成普通 Markdown：toctree 变成“相关主题列表”，:doc:/:file: 变成普通文本或代码样式。
3. 把 .. gmtplot::、.. literalinclude::、```{{eval-rst}} 中的真实命令提取为 fenced code block。
4. 删除纯布局指令，例如 :width:、:align:、.. only::、.. hlist::。
5. 如果文本明显是乱码或正文不足，输出空字符串。
6. 只输出 Markdown 正文，不要解释。

文件名：{path.name}

原文：
{excerpt}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        md = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=5000,
        ).strip()
        md = re.sub(r"^```(?:markdown|md)?\s*", "", md.strip(), flags=re.I)
        md = re.sub(r"\s*```$", "", md.strip())
        if len(md) < 40 and len(source) > 300:
            return source[:max_chars]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(md[:max_chars], encoding="utf-8")
        return md[:max_chars]
    except Exception:
        return source[:max_chars]


def _markdown_cache_file(path: Path, text: str) -> Path:
    h = hashlib.sha256()
    h.update(_MARKDOWN_NORMALIZER_VERSION.encode())
    h.update(str(path).encode(errors="ignore"))
    h.update(str(len(text)).encode())
    h.update(text[:2000].encode(errors="ignore"))
    return _SKILL_BUILDER_CACHE_DIR / "markdown" / f"{h.hexdigest()[:24]}.md"


def _llm_hierarchical_skill_plan(
    proj_name: str,
    docs: List[dict],
    rag_assist: bool = True,
    rag_cluster_target: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Skill Builder Agent: plan, design, generate, and self-check a hierarchical skill package."""
    mode = os.environ.get("SEISMICX_SKILL_BUILDER_MODE", "map_reduce").strip().lower()
    try:
        batch_size_hint = int(os.environ.get("SEISMICX_SKILL_BUILDER_BATCH_SIZE", "8"))
    except Exception:
        batch_size_hint = 8
    # 默认采用小模型友好的 map-reduce：文档项稍多时先分批抽取功能，再用 RAG/embedding 合并。
    if mode != "single" and len(docs) > max(4, batch_size_hint):
        mapped = _llm_map_reduce_skill_plan(
            proj_name,
            docs,
            rag_assist=rag_assist,
            rag_cluster_target=rag_cluster_target,
            log=log,
        )
        if _normalize_llm_units(_extract_subskill_units(mapped)):
            return mapped

    digest = _doc_digest_for_llm(docs)
    if not digest:
        return {}
    merge_hints = _embedding_merge_hints(docs) if rag_assist else ""

    prompt = f"""你是 Skill Builder Agent，负责把一个混乱的文档目录二次整理成可复用的 seismo_skill 文件夹型 SKILL。

你必须按四步工作：
1. 读目录：把 index/toctree、:doc: 引用、literalinclude/gmtplot 示例脚本、代码块和说明文字都当成证据线索。
2. 重组归档：文档原始结构通常混乱，必须由你按“功能能力/任务场景”重新归类，不要照抄目录、章节或文件名。
3. 生成内容：每个子技能都要是一页可独立使用的中文 Markdown，包含可执行步骤、命令/API、关键参数、输入输出、代码/脚本示例和验证方法。
4. 自检覆盖：指出覆盖了哪些能力、哪些内容文档不足、哪些地方需要人工复核。

任务：把下面的 PDF/文档内容转换成层级化 SKILL 包。
要求：
1. 用中文组织，不要只是引用 PDF/文件，也不要只给摘要。
2. 子技能标题必须是具体功能能力，不要使用“第 1 章”“第 2 页”这种标题。
3. 每个 subskill 都要是一页可独立阅读的 Markdown 技能资料，内容必须来自文档，不确定处写“文档未说明”。
4. 不要编造文档里没有的命令、参数、结果或案例。
5. SKILL 内部必须中英文双语：中文优先，英文用于给其它系统复用。
6. 每个 subskill 必须给出英文 slug，只能包含小写英文字母、数字和下划线，用作文件名。
7. 严禁在输出正文中保留 Sphinx/RST/Markdown 构建语法，例如 ```{{toctree}}、.. toctree::、.. hlist::、:doc:`...`、:file:`...`、.. literalinclude::、.. gmtplot::。
8. 遇到 toctree/index 页面时，要沿着其中的条目归纳“可做什么”，例如安装方式、地图底图、线条样式、色标 CPT、三维地形、矢量场、地震台站图等；不要把 toctree 当作用户可执行步骤。
9. 遇到代码块或引用脚本时，要提取其中真实命令和参数，整理成“示例代码”和“如何验证”，不要只说“参考某脚本”。
10. 必须合并同类功能：如果多个文件/片段都是同一个命令、同一个参数族或同一个任务（例如 `gmt coast -EAU` 和 `gmt coast -E=OC` 都是 `coast -E` 边界绘制），只能生成一个子技能，在其中列出多个场景和示例。
11. 一个子技能应该聚合“概念说明 + 参数解释 + 示例代码 + 常见错误 + 验证方法”，不要把两个相邻示例拆成两个技能。
12. 参考“相似片段候选”决定哪些内容应合并，但最终合并/拆分由你根据功能语义判断。
13. SOURCE_EVIDENCE 必须列出相对文件路径、章节、脚本或代码块线索；如果证据来自多个文件，要全部列出，方便后续打开源文件复核。
14. 优先输出下面的 RAW TEXT 协议；不要强行输出 JSON。小模型也可以稳定遵循这个协议。

RAW TEXT 协议：
SKILL_NAME: 中文技能包名称
SKILL_NAME_EN: English skill name
DESCRIPTION: 一句话说明这个技能包何时使用
KEYWORDS: GMT, 地图, CPT, 投影, ...

=== SUBSKILL ===
SLUG: line_color_control
TITLE_ZH: 线条颜色、宽度与样式控制
TITLE_EN: Line color width and style control
PURPOSE: 这个子技能解决什么问题
WHEN_TO_USE:
- 何时使用
CONTENT:
这里写整理后的中文正文，必须是可直接使用的步骤、参数、代码示例和说明。
COMMANDS:
- gmt plot
- -W
PARAMETERS:
- -W<pen>: 控制线条宽度、颜色和样式
WORKFLOW:
- 第一步
- 第二步
VALIDATION:
- 如何检查结果正确
SOURCE_EVIDENCE:
- source/path/file.md：章节或代码块线索
=== END_SUBSKILL ===

可以重复多个 SUBSKILL。不要输出 index、section、纯数字标题这类子技能。

文档项目名：{proj_name}

相似片段候选（RAG/embedding 召回，供你判断合并/拆分）：
{merge_hints or "无；请直接根据文档内容判断。"}

文档内容：
{digest}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        raw = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=9000,
        )
        plan = _plan_from_llm_text(raw)
        if not isinstance(plan, dict):
            return {"_error": "LLM 输出无法解析为 SKILL 计划", "_raw_preview": str(raw)[:800]}
        units = _extract_subskill_units(plan)
        if not isinstance(units, list) or not units:
            sub_plan = _llm_subskills_only_plan(proj_name, docs, plan)
            if sub_plan.get("_error"):
                sub_plan.setdefault("_raw_preview", str(raw)[:800])
                return sub_plan
            plan.update({k: v for k, v in sub_plan.items() if not k.startswith("_")})
            units = _extract_subskill_units(plan)
        cleaned_refs = _normalize_llm_units(units)
        if not cleaned_refs:
            return {"_error": "LLM 返回的 subskills 缺少 title_zh/title 或 content", "_raw_preview": str(raw)[:800]}
        plan["subskills"] = cleaned_refs
        return plan
    except Exception as exc:
        return {"_error": str(exc)}


def _llm_map_reduce_skill_plan(
    proj_name: str,
    docs: List[dict],
    rag_assist: bool = True,
    rag_cluster_target: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Build a skill plan with small-model-friendly map-reduce calls.

    The LLM planner only sees compact semantic evidence batches and merges
    capability-level subskills. When rag_assist is enabled, embeddings + DBSCAN
    group scattered files into topic clusters before LLM extraction.
    """
    try:
        evidence_limit = int(os.environ.get("SEISMICX_SKILL_BUILDER_EVIDENCE_DOCS", "96"))
    except Exception:
        evidence_limit = 96
    try:
        batch_size = int(os.environ.get("SEISMICX_SKILL_BUILDER_BATCH_SIZE", "8"))
    except Exception:
        batch_size = 8
    evidence_limit = max(16, evidence_limit)
    batch_size = max(4, min(16, batch_size))

    ranked = _rank_docs_for_skill_digest(docs)[:evidence_limit]
    batches = _semantic_doc_batches_for_skill_builder(
        ranked,
        rag_assist=rag_assist,
        fallback_batch_size=batch_size,
        cluster_target=rag_cluster_target,
        log=log,
    )
    all_units: List[dict] = []
    failed = 0
    for batch_idx, batch_info in enumerate(batches, 1):
        batch = batch_info["docs"]
        batch_plan = _llm_subskills_from_doc_batch(
            proj_name=proj_name,
            docs=batch,
            batch_no=batch_idx,
            batch_total=len(batches),
            batch_label=batch_info.get("label", ""),
        )
        units = _normalize_llm_units(_extract_subskill_units(batch_plan))
        if units:
            all_units.extend(units)
        else:
            failed += 1

    merged_units = _filter_usable_subskills(_merge_similar_subskills(all_units, use_embedding=rag_assist))
    if not merged_units:
        return {
            "_error": f"map-reduce 未生成有效子技能；失败批次 {failed}",
            "subskills": [],
        }

    keywords = _extract_keywords(proj_name, [str(d.get("text", ""))[:1000] for d in ranked[:8]])
    return {
        "display_name": f"{proj_name} 技能包",
        "display_name_en": f"{proj_name} skill package",
        "description": "基于完整文档目录索引和分批证据抽取整理的可复用技能包。",
        "description_en": "Reusable skill package built from indexed documentation with batched evidence extraction.",
        "keywords": keywords,
        "workflow": [
            "先读取 SKILL.md 判断任务范围",
            "打开 references/outline.md 定位相关子技能",
            "根据 subskills/*.md 中的命令、参数、示例和验证步骤完成任务",
            "回答时区分文档已说明和文档未说明，避免编造",
        ],
        "validation": [
            "检查使用的命令、参数或 API 是否能在子技能来源依据中找到",
            "对代码任务运行最小示例或语法检查",
            "若证据不足，明确说明缺失信息",
        ],
        "coverage_audit": {
            "covered_capabilities": [str(u.get("title_zh") or u.get("title") or "") for u in merged_units[:24]],
            "missing_or_weak": [
                f"完整目录共有 {len(docs)} 个文档项；本轮 LLM 编排使用排序后的 {len(ranked)} 个代表证据文档。",
                f"本轮按 {'RAG/embedding + DBSCAN 主题簇' if rag_assist else '顺序分批'} 组织为 {len(batches)} 个证据批次。",
                "其余文档仍保留在索引/清单中，可通过 RAG 或后续增量构建补充。",
            ],
            "suggested_tests": [
                "用 3-5 个常见任务查询检查能否命中对应子技能",
                "抽查命令型子技能中的示例是否能运行或至少通过语法检查",
            ],
        },
        "subskills": merged_units,
    }


def _llm_subskills_from_doc_batch(
    proj_name: str,
    docs: List[dict],
    batch_no: int,
    batch_total: int,
    batch_label: str = "",
) -> dict:
    digest = _doc_digest_for_llm(docs, max_chars=16000)
    if not digest:
        return {}
    prompt = f"""你是 Skill Builder Agent 的分批证据抽取器。请从当前文档批次中抽取可复用 SKILL 子技能。

要求：
1. 按功能能力合并，不要按文件名、章节名、index 页机械拆分。
2. 只使用本批次证据，不能编造命令、参数、结果或案例。
3. 输出 RAW TEXT，不要 JSON。
4. 每个子技能必须有可执行步骤、命令/API、关键参数、验证方法和来源依据。
5. 如果多个片段属于同一任务，合并成一个子技能。

RAW TEXT 格式：
=== SUBSKILL ===
SLUG: english_slug
TITLE_ZH: 中文功能标题
TITLE_EN: English title
PURPOSE: 这个子技能解决什么问题
CONTENT:
整理后的中文技能正文，包含步骤、示例和说明。
COMMANDS:
- 命令或 API
PARAMETERS:
- 参数说明
WORKFLOW:
- 步骤
VALIDATION:
- 检查方法
SOURCE_EVIDENCE:
- 文件名或章节线索
=== END_SUBSKILL ===

项目：{proj_name}
批次：{batch_no}/{batch_total}
批次主题：{batch_label or "未命名主题簇"}

文档证据：
{digest}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        raw = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=4500,
        )
        return _plan_from_llm_text(raw)
    except Exception as exc:
        return {"_error": str(exc)}


def _extract_subskill_units(plan: dict) -> List[dict]:
    if not isinstance(plan, dict):
        return []
    for key in ("subskills", "sub_skills", "skills", "skill_items", "references", "children", "技能", "子技能"):
        value = plan.get(key)
        if isinstance(value, list) and value:
            return [v for v in value if isinstance(v, dict)]
    # Some models nest the useful list under a skill_tree/capabilities object.
    for key in ("skill_tree", "capabilities", "能力树", "功能树"):
        value = plan.get(key)
        if isinstance(value, dict):
            nested = _extract_subskill_units(value)
            if nested:
                return nested
        if isinstance(value, list) and value:
            return [v for v in value if isinstance(v, dict)]
    return []


def _plan_from_llm_text(raw: str) -> dict:
    """Parse either JSON or the relaxed RAW TEXT protocol from small models."""
    plan = _json_from_text(raw)
    if isinstance(plan, dict) and plan:
        return plan
    return _raw_skill_plan_from_text(raw)


def _raw_skill_plan_from_text(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    plan: dict = {}
    header = text.split("=== SUBSKILL ===", 1)[0]
    header_map = {
        "SKILL_NAME": "display_name",
        "SKILL_NAME_EN": "display_name_en",
        "DESCRIPTION": "description",
        "DESCRIPTION_EN": "description_en",
    }
    for key, out_key in header_map.items():
        m = re.search(rf"(?im)^\s*{key}\s*[:：]\s*(.+)$", header)
        if m:
            plan[out_key] = m.group(1).strip()
    m = re.search(r"(?im)^\s*KEYWORDS\s*[:：]\s*(.+)$", header)
    if m:
        plan["keywords"] = [x.strip() for x in re.split(r"[,，;；]", m.group(1)) if x.strip()]

    blocks = re.split(r"(?im)^===\s*SUBSKILL\s*===\s*$", text)
    subskills = []
    for block in blocks[1:]:
        block = re.split(r"(?im)^===\s*END_SUBSKILL\s*===\s*$", block)[0].strip()
        unit = _parse_raw_subskill_block(block)
        if unit:
            subskills.append(unit)
    if subskills:
        plan["subskills"] = subskills
    return plan


def _parse_raw_subskill_block(block: str) -> dict:
    fields = {
        "SLUG": "slug",
        "TITLE_ZH": "title_zh",
        "TITLE_EN": "title_en",
        "PURPOSE": "purpose",
        "PURPOSE_EN": "purpose_en",
        "CONTENT": "content",
        "CONTENT_EN": "content_en",
        "WHEN_TO_USE": "when_to_use",
        "WHEN_TO_USE_EN": "when_to_use_en",
        "KEY_POINTS": "key_points",
        "KEY_POINTS_EN": "key_points_en",
        "COMMANDS": "commands_or_api",
        "COMMANDS_OR_API": "commands_or_api",
        "PARAMETERS": "parameters",
        "WORKFLOW": "mini_workflow",
        "MINI_WORKFLOW": "mini_workflow",
        "PITFALLS": "pitfalls",
        "VALIDATION": "validation",
        "SOURCE_EVIDENCE": "source_evidence",
    }
    matches = list(re.finditer(r"(?im)^\s*([A-Z_]+)\s*[:：]\s*(.*)$", block))
    if not matches:
        return {}
    unit: dict = {}
    for i, match in enumerate(matches):
        raw_key = match.group(1).strip().upper()
        out_key = fields.get(raw_key)
        if not out_key:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        inline = match.group(2).strip()
        tail = block[start:end].strip()
        value = (inline + ("\n" + tail if tail else "")).strip()
        if out_key in {
            "when_to_use", "when_to_use_en", "key_points", "key_points_en",
            "commands_or_api", "parameters", "mini_workflow", "pitfalls",
            "validation", "source_evidence",
        }:
            unit[out_key] = _parse_raw_list(value)
        else:
            unit[out_key] = value
    if not unit.get("title_zh") and unit.get("title_en"):
        unit["title_zh"] = unit["title_en"]
    if not unit.get("slug"):
        unit["slug"] = _fallback_english_slug(unit.get("title_en") or unit.get("title_zh") or "", fallback="subskill")
    return unit if (unit.get("title_zh") and unit.get("content")) else {}


def _parse_raw_list(value: str) -> List[str]:
    lines = []
    for line in str(value or "").splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^[-*•]\s*", "", s).strip()
        if s:
            lines.append(s)
    if len(lines) <= 1 and ("," in str(value) or "，" in str(value)):
        lines = [x.strip() for x in re.split(r"[,，;；]", str(value)) if x.strip()]
    return lines


def _normalize_llm_units(units: List[dict]) -> List[dict]:
    cleaned_refs = []
    for ref in units[:80]:
        if not isinstance(ref, dict):
            continue
        ref = dict(ref)
        title = str(
            ref.get("title_zh")
            or ref.get("title")
            or ref.get("title_en")
            or ref.get("name")
            or ref.get("功能")
            or ""
        ).strip()
        content = str(
            ref.get("content")
            or ref.get("skill_content")
            or ref.get("body")
            or ref.get("description")
            or ref.get("说明")
            or ""
        ).strip()
        if title and content:
            ref.setdefault("title_zh", title)
            ref.setdefault("content", content)
            cleaned_refs.append(ref)
    return cleaned_refs


def _llm_subskills_only_plan(proj_name: str, docs: List[dict], base_plan: dict) -> dict:
    digest = _doc_digest_for_llm(docs, max_chars=36000)
    capability_hint = ", ".join(_as_list(base_plan.get("keywords"), 20))
    prompt = f"""你已经给出了 {proj_name} 的总体技能包说明，但缺少子技能列表。

现在请按 RAW TEXT 协议输出，不要 JSON，不要解释。可以被小模型稳定生成。

目标：按功能能力合并整理 8-16 个可直接使用的子技能。不要按文件、章节、index、数字拆分。
每个子技能必须包含：
- SLUG: 英文小写下划线文件名
- TITLE_ZH / TITLE_EN
- PURPOSE
- CONTENT: 中文正文，包含可执行步骤、关键参数、命令或代码示例。严禁保留 toctree、gmtplot、literalinclude、:doc: 等文档构建语法。
- COMMANDS
- PARAMETERS
- WORKFLOW
- VALIDATION
- SOURCE_EVIDENCE

格式：
=== SUBSKILL ===
SLUG: gmt_coast_boundary_plotting
TITLE_ZH: GMT coast 国家与区域边界绘制
TITLE_EN: GMT coast country and region boundary plotting
PURPOSE: 使用 GMT coast 的 -E 选项绘制国家、地区或大洲边界。
CONTENT:
这里写可直接使用的中文技能内容。
COMMANDS:
- gmt coast
PARAMETERS:
- -E<code>+p<pen>: 绘制并设置边界线
WORKFLOW:
- ...
VALIDATION:
- ...
SOURCE_EVIDENCE:
- ...
=== END_SUBSKILL ===

如果多个片段属于同一功能，必须合并。例如 coast -EAU 和 coast -E=OC 都属于“GMT coast 国家与区域边界绘制”。

总体关键词：{capability_hint or "GMT 绘图、地图、CPT、投影、数据集、示例脚本"}

文档证据：
{digest}
"""
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from helpers import get_llm_config, llm_call  # type: ignore
        raw = llm_call(
            [{"role": "user", "content": prompt}],
            get_llm_config(),
            max_tokens=9000,
        )
        plan = _plan_from_llm_text(raw)
        if not isinstance(plan, dict):
            return {"_error": "第二轮 subskills 输出无法解析", "_raw_preview": str(raw)[:800]}
        units = _normalize_llm_units(_extract_subskill_units(plan))
        if not units:
            return {"_error": "第二轮仍未返回有效 subskills", "_raw_preview": str(raw)[:800]}
        return {"subskills": units}
    except Exception as exc:
        return {"_error": f"第二轮 subskills 生成失败：{exc}"}


def _as_list(value, limit: int = 12) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _merge_hierarchical_plan_into_spec(spec: dict, plan: dict) -> dict:
    merged = dict(spec or {})
    for key in ("display_name", "description"):
        if isinstance(plan.get(key), str) and plan[key].strip():
            merged[key] = plan[key].strip()
    for key in ("display_name_en", "description_en"):
        if isinstance(plan.get(key), str) and plan[key].strip():
            merged[key] = plan[key].strip()
    for key in (
        "keywords", "when_to_use", "when_to_use_en", "workflow", "workflow_en",
        "validation", "validation_en", "example_prompts",
    ):
        vals = _as_list(plan.get(key), 14)
        if vals:
            merged[key] = vals
    return merged


def _merge_similar_subskills(
    units,
    log: Optional[Callable[[str], None]] = None,
    use_embedding: bool = True,
) -> List[dict]:
    """
    Merge candidate subskills by semantic/RAG similarity.

    LLMs, especially small models, often split adjacent examples into separate
    skills. When RAG/embedding assistance is enabled, we embed each candidate's
    title, purpose, commands, parameters, and source evidence, then cluster
    similar candidates. Keyword matching remains as a fallback.
    """
    if not isinstance(units, list):
        return []
    normalized = [dict(u) for u in units if isinstance(u, dict)]
    if not normalized:
        return []

    clusters = _embedding_cluster_subskills(normalized) if use_embedding else []
    method = "RAG/embedding"
    if not clusters:
        clusters = _lexical_cluster_subskills(normalized)
        method = "keyword fallback" if use_embedding else "keyword"
    if log and len(normalized) != len(clusters):
        log(f"   🔎 使用 {method} 相似性合并子技能：{len(normalized)} → {len(clusters)}")
    return clusters


def _subskill_semantic_text(unit: dict) -> str:
    fields = [
        "slug", "title_zh", "title", "title_en", "purpose", "purpose_en",
        "content", "commands_or_api", "parameters", "mini_workflow",
        "validation", "source_evidence",
    ]
    parts: List[str] = []
    for key in fields:
        value = unit.get(key)
        if isinstance(value, list):
            parts.append(" ; ".join(str(v) for v in value[:40]))
        elif value:
            parts.append(str(value))
    return "\n".join(parts)[:5000]


def _embedding_cluster_subskills(units: List[dict]) -> List[dict]:
    if len(units) <= 1:
        return units
    texts = [_subskill_semantic_text(unit) for unit in units]
    try:
        _web = Path(__file__).parent.parent / "web_app"
        if str(_web) not in sys.path:
            sys.path.insert(0, str(_web))
        from rag_backends import EmbeddingModel  # type: ignore
        import numpy as np  # type: ignore

        vecs = EmbeddingModel.get().encode(texts, batch_size=16)
        vecs = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        sims = vecs @ vecs.T
    except Exception:
        return []

    try:
        threshold = float(os.environ.get("SEISMICX_SKILL_MERGE_SIM_THRESHOLD", "0.66"))
    except Exception:
        threshold = 0.66
    parent = list(range(len(units)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if float(sims[i, j]) >= threshold or _strong_command_overlap(units[i], units[j]):
                union(i, j)

    grouped: Dict[int, List[dict]] = {}
    order: List[int] = []
    for idx, unit in enumerate(units):
        root = find(idx)
        if root not in grouped:
            grouped[root] = []
            order.append(root)
        grouped[root].append(unit)

    merged: List[dict] = []
    for root in order:
        cluster = grouped[root]
        base = cluster[0]
        for unit in cluster[1:]:
            base = _merge_two_subskill_units(base, unit)
        merged.append(base)
    return merged


def _strong_command_overlap(a: dict, b: dict) -> bool:
    a_cmds = {str(x).lower().strip() for x in _as_list(a.get("commands_or_api"), 80)}
    b_cmds = {str(x).lower().strip() for x in _as_list(b.get("commands_or_api"), 80)}
    if a_cmds and b_cmds and a_cmds & b_cmds:
        return True
    a_params = {str(x).lower().strip().split(":", 1)[0] for x in _as_list(a.get("parameters"), 80)}
    b_params = {str(x).lower().strip().split(":", 1)[0] for x in _as_list(b.get("parameters"), 80)}
    return bool(a_params and b_params and len(a_params & b_params) >= 2)


def _lexical_cluster_subskills(units: List[dict]) -> List[dict]:
    """Embedding unavailable fallback: merge by token Jaccard and command overlap."""
    if len(units) <= 1:
        return units

    def toks(unit: dict) -> set:
        text = _subskill_semantic_text(unit).lower()
        return set(re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]{2,}", text))

    groups: List[dict] = []
    group_tokens: List[set] = []
    for unit in units:
        unit_tokens = toks(unit)
        best = -1
        best_score = 0.0
        for idx, existing_tokens in enumerate(group_tokens):
            score = len(unit_tokens & existing_tokens) / max(1, len(unit_tokens | existing_tokens))
            if score > best_score:
                best = idx
                best_score = score
        if best >= 0 and (best_score >= 0.18 or _strong_command_overlap(groups[best], unit)):
            groups[best] = _merge_two_subskill_units(groups[best], unit)
            group_tokens[best] |= unit_tokens
        else:
            groups.append(unit)
            group_tokens.append(set(unit_tokens))
    return groups


def _filter_usable_subskills(units: List[dict]) -> List[dict]:
    usable: List[dict] = []
    seen_titles: set[str] = set()
    for unit in units:
        title = str(unit.get("title_zh") or unit.get("title") or unit.get("title_en") or "").strip()
        content = str(unit.get("content") or "").strip()
        commands = _as_list(unit.get("commands_or_api"), 60)
        params = _as_list(unit.get("parameters"), 60)
        if not _is_meaningful_subskill(title, content, commands, params):
            continue
        key = _ascii_slug(title, fallback=title)[:80]
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unit["content"] = _clean_generated_skill_markdown(content)
        usable.append(unit)
    return usable[:40]


def _is_meaningful_subskill(title: str, content: str, commands: List[str], params: List[str]) -> bool:
    t = str(title or "").strip()
    c = str(content or "").strip()
    if not t:
        return False
    low = t.lower().strip()
    if low in {"index", "section", "api", "文档维护", "许可协议", "贡献者"}:
        return False
    if re.fullmatch(r"(?:section[_ ]*)?\d+(?:[._ -]\d+)*", low):
        return False
    if re.fullmatch(r"[-+0-9.,\\s]+(?:[a-z]{1,4})?", low):
        return False
    if len(c) < 80 and not commands and not params:
        return False
    bad_markers = ["```{toctree}", ".. toctree::", ".. hlist::"]
    if any(marker in c for marker in bad_markers) and not commands:
        return False
    # Reject table-coordinate fragments such as "1.75 11p Helvetica BL OTHER".
    alpha_or_cjk = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", t + c[:200]))
    digits = len(re.findall(r"\d", t + c[:200]))
    if digits > alpha_or_cjk * 2 and not commands:
        return False
    return True


def _merge_two_subskill_units(a: dict, b: dict) -> dict:
    merged = dict(a)
    for key in ("content", "content_en", "purpose", "purpose_en"):
        av = str(merged.get(key) or "").strip()
        bv = str(b.get(key) or "").strip()
        if bv and bv not in av:
            merged[key] = (av + "\n\n" + bv).strip() if av else bv
    for key in (
        "when_to_use", "when_to_use_en", "key_points", "key_points_en",
        "commands_or_api", "parameters", "mini_workflow", "mini_workflow_en",
        "pitfalls", "pitfalls_en", "validation", "validation_en", "source_evidence",
    ):
        vals = []
        for item in _as_list(merged.get(key), 80) + _as_list(b.get(key), 80):
            if item not in vals:
                vals.append(item)
        if vals:
            merged[key] = vals[:40]
    title = str(merged.get("title_zh") or merged.get("title") or "")
    other_title = str(b.get("title_zh") or b.get("title") or "")
    if other_title and other_title not in title and len(title) < 16:
        merged["title_zh"] = f"{title} / {other_title}".strip(" /")
    merged["slug"] = _fallback_english_slug(str(merged.get("title_en") or merged.get("title_zh") or merged.get("title") or ""), fallback=str(merged.get("slug") or "merged_skill"))
    return merged


def _clean_generated_skill_markdown(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"```\{toctree\}.*?```", lambda m: _toctree_to_plain(m.group(0)), s, flags=re.DOTALL)
    s = re.sub(r"```\{eval-rst\}\s*(.*?)```", lambda m: _eval_rst_to_plain(m.group(1)), s, flags=re.DOTALL)
    s = re.sub(r"(?ms)^\\s*\\.\\.\\s+(?:toctree|hlist|only)::.*?(?=^\\S|\\Z)", "", s)
    s = re.sub(r":doc:`([^`<]+)<([^`>]+)>`", r"\1 (\2)", s)
    s = re.sub(r":doc:`([^`]+)`", r"\1", s)
    s = re.sub(r":file:`([^`]+)`", r"`\1`", s)
    return s.strip()


def _toctree_to_plain(block: str) -> str:
    items = []
    for line in block.splitlines():
        t = line.strip()
        if not t or t.startswith(("```", ":", "..")):
            continue
        items.append(t)
    return "文档目录线索：" + "、".join(items) if items else ""


def _eval_rst_to_plain(body: str) -> str:
    commands = []
    lines = []
    in_directive = False
    for raw in body.splitlines():
        t = raw.strip()
        if t.startswith(".. gmtplot::") or t.startswith(".. literalinclude::"):
            in_directive = True
            continue
        if in_directive and (not t or t.startswith(":")):
            continue
        if t.startswith("gmt "):
            commands.append(t)
        elif t:
            lines.append(t)
    if commands:
        return "示例命令：\n\n```bash\n" + "\n".join(commands) + "\n```"
    return "\n".join(lines)


def _render_llm_subskill(unit: dict) -> str:
    title = str(unit.get("title_zh") or unit.get("title") or "未命名章节").strip()
    title_en = str(unit.get("title_en") or "").strip()
    purpose = str(unit.get("purpose") or "").strip()
    purpose_en = str(unit.get("purpose_en") or "").strip()
    content = _clean_generated_skill_markdown(str(unit.get("content") or "").strip())
    content_en = _clean_generated_skill_markdown(str(unit.get("content_en") or "").strip())
    when = _as_list(unit.get("when_to_use"), 12)
    when_en = _as_list(unit.get("when_to_use_en"), 12)
    key_points = _as_list(unit.get("key_points"), 20)
    key_points_en = _as_list(unit.get("key_points_en"), 20)
    commands = _as_list(unit.get("commands_or_api"), 30)
    parameters = _as_list(unit.get("parameters"), 30)
    mini_workflow = _as_list(unit.get("mini_workflow"), 20)
    mini_workflow_en = _as_list(unit.get("mini_workflow_en"), 20)
    pitfalls = _as_list(unit.get("pitfalls"), 20)
    pitfalls_en = _as_list(unit.get("pitfalls_en"), 20)
    validation = _as_list(unit.get("validation"), 20)
    validation_en = _as_list(unit.get("validation_en"), 20)
    evidence = _as_list(unit.get("source_evidence"), 20)

    sections = [f"# {title}", ""]
    if title_en:
        sections += [f"**English:** {title_en}", ""]
    if purpose:
        sections += ["## 用途 / Purpose", "", purpose, ""]
        if purpose_en:
            sections += [f"**English:** {purpose_en}", ""]
    if when:
        sections += ["## 何时使用 / When To Use", "", _yaml_list(when), ""]
        if when_en:
            sections += ["**English**", "", _yaml_list(when_en), ""]
    sections += ["## 技能内容 / Skill Content", "", content or "文档未说明。", ""]
    if content_en:
        sections += ["**English**", "", content_en, ""]
    if key_points:
        sections += ["## 关键要点 / Key Points", "", _yaml_list(key_points), ""]
        if key_points_en:
            sections += ["**English**", "", _yaml_list(key_points_en), ""]
    if commands:
        sections += ["## 命令 / API / Commands", "", _yaml_list(commands), ""]
    if parameters:
        sections += ["## 参数与选项 / Parameters", "", _yaml_list(parameters), ""]
    if mini_workflow:
        sections += ["## 子技能工作流 / Mini Workflow", "", _yaml_list(mini_workflow), ""]
        if mini_workflow_en:
            sections += ["**English**", "", _yaml_list(mini_workflow_en), ""]
    if pitfalls:
        sections += ["## 注意事项 / Pitfalls", "", _yaml_list(pitfalls), ""]
        if pitfalls_en:
            sections += ["**English**", "", _yaml_list(pitfalls_en), ""]
    if validation:
        sections += ["## 验证方法 / Validation", "", _yaml_list(validation), ""]
        if validation_en:
            sections += ["**English**", "", _yaml_list(validation_en), ""]
    if evidence:
        sections += ["## 来源依据", "", _yaml_list(evidence), ""]
    return "\n".join(sections).strip() + "\n"


def _skill_builder_audit(plan: dict, generated_count: int) -> str:
    audit = plan.get("coverage_audit") if isinstance(plan.get("coverage_audit"), dict) else {}
    covered = _as_list(audit.get("covered_capabilities"), 40)
    weak = _as_list(audit.get("missing_or_weak"), 40)
    tests = _as_list(audit.get("suggested_tests"), 40)
    lines = [
        "# Skill Builder Agent 自检报告",
        "",
        f"- 生成子技能数：{generated_count}",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 已覆盖能力",
        "",
        _yaml_list(covered) if covered else "- 文档未说明",
        "",
        "## 缺失或薄弱内容",
        "",
        _yaml_list(weak) if weak else "- 暂未发现，仍建议人工抽查关键参数和命令。",
        "",
        "## 建议验证任务",
        "",
        _yaml_list(tests) if tests else "- 随机抽取 3 个用户问题，检查能否定位到对应 subskills 文档并给出有依据回答。",
        "",
    ]
    return "\n".join(lines)


def _split_document_into_skill_units(path_name: str, text: str, max_units: int = 12) -> List[dict]:
    """Fallback splitter when the LLM Skill Builder Agent is unavailable."""
    lines = [ln.rstrip() for ln in str(text or "").splitlines()]
    sections: List[Tuple[str, List[str]]] = []
    current_title = Path(path_name).stem
    current_body: List[str] = []

    heading_re = re.compile(r"^(#{1,4}\s+|第[一二三四五六七八九十\d]+[章节节、.]\s*|[0-9]+(?:\.[0-9]+)*\s+)(.+)$")
    for ln in lines:
        s = ln.strip()
        m = heading_re.match(s)
        if m and len(s) <= 120:
            if current_body:
                sections.append((current_title, current_body))
            current_title = m.group(2).strip() if m.lastindex and m.group(2).strip() else s.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(ln)
    if current_body:
        sections.append((current_title, current_body))

    if not sections:
        clean = str(text or "").strip()
        sections = [(Path(path_name).stem, [clean])]

    units = []
    for idx, (title, body_lines) in enumerate(sections[:max_units], 1):
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        title = title.strip() or f"文档内容 {idx}"
        units.append({
            "title_zh": title,
            "title_en": _fallback_english_slug(title, fallback=f"section_{idx}").replace("_", " ").title(),
            "slug": _fallback_english_slug(title, fallback=f"section_{idx}"),
            "purpose": "根据原始文档章节整理出的可复用技能资料。",
            "purpose_en": "Reusable skill notes converted from a source document section.",
            "content": body[:9000],
            "content_en": "English translation was not generated because the LLM planner was unavailable. Use the Chinese content as the source of truth.",
            "key_points": _fallback_key_points(body),
            "commands_or_api": _extract_command_like_items(body),
            "parameters": _extract_parameter_like_items(body),
            "mini_workflow": ["阅读本页技能内容", "提取文档明确给出的命令、参数或流程", "若用于编程，先写最小测试验证输出"],
            "mini_workflow_en": ["Read this subskill page", "Extract documented commands, parameters, or workflow steps", "When coding, write a minimal test before using the result"],
            "pitfalls": ["不要使用文档未说明的参数或命令", "如果信息不足，应明确说明文档未说明"],
            "pitfalls_en": ["Do not use undocumented parameters or commands", "State explicitly when the document does not provide enough information"],
            "validation": ["检查回答或代码是否能在本页找到依据", "检查是否包含必要参数和输入输出说明"],
            "validation_en": ["Check whether the answer or code is grounded in this page", "Check required parameters, inputs, and outputs"],
            "source_evidence": [f"Converted from `{path_name}` section `{title}`"],
        })
    return units


def _fallback_key_points(text: str, limit: int = 8) -> List[str]:
    points = []
    for ln in str(text or "").splitlines():
        s = ln.strip(" -*\t")
        if 18 <= len(s) <= 160:
            points.append(s)
        if len(points) >= limit:
            break
    return points


def _extract_command_like_items(text: str, limit: int = 20) -> List[str]:
    items = []
    for token in re.findall(r"(?:gmt\s+\w+|-[A-Za-z][A-Za-z0-9+/.-]*|[A-Za-z_][A-Za-z0-9_]+\([^)]*\))", str(text or "")):
        if token in {"-rst"}:
            continue
        if token not in items:
            items.append(token)
        if len(items) >= limit:
            break
    return items


def _extract_parameter_like_items(text: str, limit: int = 20) -> List[str]:
    items = []
    for token in re.findall(r"(?:(?:参数|选项|字段)\s*[:：]\s*[^\n]{1,80}|-[A-Za-z][A-Za-z0-9+/.-]*)", str(text or "")):
        token = token.strip()
        if token in {"-rst"}:
            continue
        if token and token not in items:
            items.append(token)
        if len(items) >= limit:
            break
    return items


def _render_converted_reference(path_name: str, unit: dict) -> str:
    return _render_llm_subskill(unit)


def _resolve_generated_skill_target(skill_name: str, overwrite: bool = True) -> Path:
    base = _USER_SKILL_DIR / skill_name
    if not base.exists():
        return base
    skill_md = base / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8", errors="ignore") if skill_md.exists() else ""
    if overwrite and (
        f"generated_by: {_DOC_SKILL_GENERATOR}" in text
        or (base.name.startswith("_gen_") and not skill_md.exists())
    ):
        return base
    for i in range(2, 100):
        candidate = _USER_SKILL_DIR / f"{skill_name}_{i}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Cannot find free skill folder for {skill_name}")


def _yaml_list(items: List[str], indent: str = "") -> str:
    return "\n".join(f"{indent}- {str(item).replace(chr(10), ' ').strip()}" for item in items)


def _render_folder_skill_md(proj_name: str, spec: dict, docs: List[dict]) -> str:
    raw_name = str(spec.get("name") or proj_name).strip()
    if raw_name.startswith("_gen_") or raw_name.startswith("gen_"):
        name = _generated_skill_slug(raw_name)
    else:
        name = _safe_skill_slug(raw_name)
    display = spec.get("display_name") or name.replace("_", " ").title()
    display_en = str(spec.get("display_name_en") or "").strip()
    keywords = [str(k) for k in spec.get("keywords", [])][:14]
    description = str(spec.get("description", "")).replace("\n", " ").strip()
    description_en = str(spec.get("description_en") or "").replace("\n", " ").strip()
    when = spec.get("when_to_use", [])
    when_en = spec.get("when_to_use_en", [])
    workflow = spec.get("workflow", [])
    workflow_en = spec.get("workflow_en", [])
    validation = spec.get("validation", [])
    validation_en = spec.get("validation_en", [])
    examples = spec.get("example_prompts", [])
    refs = [d["path"] for d in docs[:12]]
    return f"""---
name: {name}
description: >-
  {description}
category: generated
keywords:
{_yaml_list(keywords, "  ")}
source: generated
generated_by: {_DOC_SKILL_GENERATOR}
generated_from: seismo_skill/docs/{proj_name}/
generated_at: {datetime.now().isoformat(timespec="seconds")}
---

# {display}
{f"**English:** {display_en}" if display_en else ""}

## Purpose

这是由 LLM 从原始文档内容整理出的层级化中文技能包。使用它回答问题或编写代码时，应优先读取 `references/outline.md`，再打开相关章节页；这些章节页已经从 PDF/文档正文转换而来，不需要再读取原 PDF。
如需追溯原文，`references/manifest.md` 保留了参与构建的源文件相对路径和绝对路径。
{f"\\nEnglish: {description_en}" if description_en else ""}

## When To Use / 何时使用

{_yaml_list(when)}
{("**English**" + chr(10) + chr(10) + _yaml_list(when_en)) if when_en else ""}

## Workflow / 工作流

{_yaml_list(workflow)}
{("**English**" + chr(10) + chr(10) + _yaml_list(workflow_en)) if workflow_en else ""}

## Converted Skill References

1. 先打开 `references/outline.md` 判断该问题对应哪些子技能。
2. 再打开相关 `subskills/*.md`，从其中提取概念、参数、命令、流程、注意事项和验证方法。
3. 回答时明确区分“文档已说明”和“文档未说明”。不要编造缺失的命令、参数、实验结果或结论。
4. 如果用户要求编程，实现代码前先根据对应章节写一个最小检查或 mini test。
5. 可查看 `references/builder_audit.md` 了解 Skill Builder Agent 的覆盖范围和薄弱项。

{_yaml_list(refs)}

## Validation

{_yaml_list(validation)}

## Example Prompts

{_yaml_list(examples)}
"""


def _render_agent_yaml(spec: dict) -> str:
    display = str(spec.get("display_name") or spec.get("name") or "Generated Skill")
    desc = str(spec.get("description") or "").replace("\n", " ").strip()
    return f"""display_name: {display}
short_description: >-
  {desc}
default_prompt: >-
  Use the generated SKILL.md, references/outline.md, and relevant subskills/*.md files. Ground claims in the converted skill documents, produce runnable code when asked, and state missing information explicitly.
"""


def delete_generated_builtin_skill(name: str) -> bool:
    """Delete a docs-generated folder skill under project skill directories."""
    if str(name).startswith("_gen_") or str(name).startswith("gen_"):
        safe = _generated_skill_slug(name)
    else:
        safe = _safe_skill_slug(name)
    for root in (_USER_SKILL_DIR, _BUILTIN_SKILL_DIR):
        folders = root.iterdir() if root.exists() else []
        for folder in folders:
            if not folder.is_dir():
                continue
            skill_md = folder / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text(encoding="utf-8", errors="ignore")
            if f"name: {safe}" not in text and folder.name != safe:
                continue
            if f"generated_by: {_DOC_SKILL_GENERATOR}" not in text:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            _invalidate_skill_cache()
            return True
    return False


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _get_kb():
    """动态加载 KnowledgeBase，兼容 web_app/ 和独立运行两种路径。"""
    try:
        from web_app.rag_engine import get_knowledge_base
        return get_knowledge_base()
    except ImportError:
        pass
    _web = Path(__file__).parent.parent / "web_app"
    if str(_web) not in sys.path:
        sys.path.insert(0, str(_web))
    from rag_engine import get_knowledge_base
    return get_knowledge_base()


def _invalidate_skill_cache():
    try:
        from seismo_skill import skill_loader as _sl
        _sl.invalidate_cache()
    except Exception:
        pass


def _find_related_builtin_skills(keywords: List[str], top_n: int = 3) -> List[str]:
    """
    根据关键词列表在已有内置/用户技能中找出关联技能（关键词重叠度最高的）。
    用于生成项目技能时写入 related_skills 字段，建立正向链接。

    返回技能名列表（最多 top_n 个，不含 generated 技能）。
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _parent = _Path(__file__).parent.parent
        if str(_parent) not in _sys.path:
            _sys.path.insert(0, str(_parent))
        from seismo_skill.skill_loader import _get_skills
        skills = _get_skills()
    except Exception:
        return []

    kw_set = {k.lower() for k in keywords}
    scored: List[tuple] = []
    for s in skills:
        if s["source"] == "generated":
            continue
        overlap = sum(1 for k in s.get("keywords", []) if k.lower() in kw_set)
        if overlap > 0:
            scored.append((overlap, s["name"]))

    scored.sort(reverse=True)
    return [name for _, name in scored[:top_n]]


def _extract_keywords(stem: str, chunks: List[str]) -> List[str]:
    """从文件/文件夹名和前几个 chunk 中提取关键词列表（最多 10 个）。"""
    name_words = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", stem)
    keywords = list(dict.fromkeys(w.lower() for w in name_words))
    if chunks:
        text = " ".join(chunks[:3])
        en_words = re.findall(r"[A-Za-z]{4,}", text)
        freq: dict = {}
        for w in en_words:
            w_l = w.lower()
            freq[w_l] = freq.get(w_l, 0) + 1
        top = sorted(freq, key=lambda x: -freq[x])[:8]
        for w in top:
            if w not in keywords:
                keywords.append(w)
    return keywords[:10]


# ── Skill 模板 ────────────────────────────────────────────────────────────────

_PROJ_SKILL_TEMPLATE = """\
---
name: {name}
category: generated
keywords: {keywords}
source: generated
rag_sources: {proj_name}
related_skills: {related_skills}
generated_from: knowledge/{proj_name}/
generated_at: {generated_at}
---

# {title}

## 说明

本技能由文档项目文件夹 `{proj_name}` 自动生成（共 {file_count} 个文件），支持 **RAG 增强检索**。
{description}

## 包含文档

`{doc_names}`

## 文档摘要

```
{preview}
```

## 使用方式

直接提问即可，例如：

- "{title} 的基本用法是什么？"
- "如何用 {proj_name} 实现…？"
- "{title} 的参数格式是什么？"

系统将自动检索文档内容并结合上下文给出准确答案。

## 注意事项

- 本技能为自动生成，内容来自原始文档的 RAG 索引
- 如需修改，请在技能管理页面编辑（修改后 `source` 将变为 `user`）
"""

_FILE_SKILL_TEMPLATE = """\
---
name: {name}
category: generated
keywords: {keywords}
source: generated
rag_sources: {doc_name}
generated_from: {rel_path}
generated_at: {generated_at}
---

# {title}

## 说明

本技能由文档 `{doc_name}` 自动生成，支持 **RAG 增强检索**。

> 原文档：`{rel_path}`

## 文档摘要

```
{preview}
```

## 使用方式

直接提问，例如：

- "{title} 的输入格式是什么？"
- "如何使用 {title}？"

系统将自动检索文档内容并结合上下文回答。

## 注意事项

- 本技能为自动生成，内容来自原始文档的 RAG 索引
- 如需修改，请在技能管理页面编辑（修改后 `source` 将变为 `user`）
"""


# ── 模块级便捷函数 ────────────────────────────────────────────────────────────

def scan_and_report() -> ScanResult:
    """一行调用：扫描默认 knowledge/ 目录并返回结果。"""
    return KnowledgeIndexer().scan()


def build_all(progress_cb=None, stop_event=None) -> BuildResult:
    """一行调用：构建/更新默认 knowledge/ 目录的全部索引。"""
    return KnowledgeIndexer().build(progress_cb=progress_cb, stop_event=stop_event)
