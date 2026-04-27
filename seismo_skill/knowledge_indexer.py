"""
knowledge_indexer.py — seismo_skill/knowledge/ 目录文档索引器

功能
----
1. 扫描 seismo_skill/knowledge/ 下的文档（多层级目录，支持 PDF/DOCX/TXT/MD/RST/HTML）
2. 维护 manifest（seismo_rag/dir_manifest.json）检测新增/修改/删除
3. 每个顶级子文件夹作为一个"项目"，统一索引为 RAG，并由 LLM 生成一个 Skill
4. knowledge/ 根目录下的文件按原来的逐文件方式处理
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
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── 常量 ───────────────────────────────────────────────────────────────────────

# 支持的文档格式
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".rst", ".html", ".htm"}

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

# 自动生成的 Skill 存储目录
_USER_SKILL_DIR = Path.home() / ".seismicx" / "skills"

# 每个项目最多索引多少文件（优先高价值格式，避免处理海量叶子页）
MAX_FILES_PER_PROJECT = 120


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
    return (9, depth)


def _select_key_files(files: List[Path], max_count: int = MAX_FILES_PER_PROJECT) -> List[Path]:
    """
    从文件列表中智能选取最有价值的文件，最多 max_count 个。

    策略：
    - README / index 文件全部保留
    - 按优先级 + 目录深度排序
    - 深度较浅（章节级）的文件优先于叶子页
    - 若文件数 <= max_count，全部保留
    """
    if len(files) <= max_count:
        return files

    sorted_files = sorted(files, key=_file_priority)
    return sorted_files[:max_count]


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

                    if entry is None or entry.get("status") != "indexed":
                        result.new.append(path)
                        has_new = True
                    elif self._is_changed(path, entry):
                        result.modified.append(path)
                        has_new = True
                    elif entry.get("status") == "failed":
                        result.failed.append(rel)
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
                if rel in new_rels:
                    status = "new"
                elif rel in modified_rels:
                    status = "modified"
                elif entry.get("status") == "indexed":
                    status = "indexed"
                elif entry.get("status") == "failed":
                    status = "failed"
                else:
                    status = "new"
                proj_entry = self._proj_manifest.get(item.stem, {})
                projects.append({
                    "name": item.name,
                    "is_folder": False,
                    "total_files": 1,
                    "selected_files": 1,
                    "indexed_files": 1 if entry.get("status") == "indexed" else 0,
                    "failed_files": 1 if entry.get("status") == "failed" else 0,
                    "status": status,
                    "skill_name": entry.get("skill_name", ""),
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

                if indexed == 0:
                    status = "new"
                elif has_new:
                    status = "modified" if indexed > 0 else "new"
                elif indexed < len(selected):
                    status = "partial"
                else:
                    status = "indexed"

                proj_entry = self._proj_manifest.get(item.name, {})
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
        pending_projects = sum(1 for p in projects if p["status"] in ("new", "partial", "modified"))

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
