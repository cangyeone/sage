"""
rag_engine.py — BGE-M3 + FAISS knowledge base with TF-IDF fallback.

Storage layout (seismo_rag/):
    faiss_index.bin   — FAISS vectors
    metadata.json     — doc + chunk records
    id_map.json       — FAISS position → chunk_id
    docs/             — copies of indexed files

Public API
----------
    kb = get_knowledge_base()

    kb.add_document(path, ...)        ingest any supported file
    kb.delete_doc(doc_id)             remove a document
    kb.clear()                        wipe everything

    kb.retrieve(query, top_k, ...)    → [(DocChunk, score), ...]
    kb.retrieve_relevant_docs(query)  → [dict, ...]
    kb.build_rag_context(query)       → str  (LLM-ready)

    kb.list_docs()                    → [DocMeta, ...]
    kb.status()                       → dict

CLI (python rag_engine.py --help):
    --status          print knowledge base statistics
    --add FILE        index a document
    --query TEXT      run a retrieval query
    --test            run self-tests (no files required)
    --top-k N         number of results (default 5)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_WEB_APP_DIR = Path(__file__).parent
if str(_WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_APP_DIR))

# ---------------------------------------------------------------------------
# Re-export backends so callers can do: from rag_engine import EmbeddingModel
# ---------------------------------------------------------------------------

try:
    from rag_backends import EmbeddingModel, FaissIndex  # type: ignore  # noqa: F401
    from rag_extractors import extract_text, chunk_text  # type: ignore  # noqa: F401
except ImportError:
    pass  # graceful if run outside web_app/ context before path is set


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

KB_DIR     = Path(__file__).parent.parent / "seismo_rag"
INDEX_FILE = KB_DIR / "faiss_index.bin"
META_FILE  = KB_DIR / "metadata.json"
IDMAP_FILE = KB_DIR / "id_map.json"
DOCS_DIR   = KB_DIR / "docs"

KB_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Keyword retrieval helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # English high-frequency function words
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "with", "using", "use", "used", "into", "about", "than", "then",
    # Common Chinese filler words. Domain terms such as 震相/地震/速度 are kept.
    "的", "了", "和", "与", "或", "在", "是", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "这个", "那个", "一下", "一个", "一些", "进行",
    "使用", "用于", "可以", "需要", "如何", "什么", "怎么", "以及", "并且",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-/.]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


_RAG_CONTEXT_COMPLETE_ONLY_NOTE = (
    "Important citation rule: the passages below may be retrieval chunks. "
    "Use them as evidence only when the relevant sentence, paragraph, list item, "
    "table row, or code block is complete. Do not quote, reproduce, or rely on "
    "truncated tails, partial code fences, or half sentences; treat incomplete "
    "text only as a search clue.\n"
)


def _drop_incomplete_fenced_tail(text: str) -> str:
    """Avoid injecting an unterminated Markdown code fence into LLM context."""
    lines = str(text or "").splitlines()
    in_fence = False
    fence_char = ""
    fence_len = 0
    open_idx = -1
    for i, line in enumerate(lines):
        m = re.match(r"^\s*([`~]{3,})", line)
        if not m:
            continue
        marker = m.group(1)
        char = marker[0]
        if any(ch != char for ch in marker):
            continue
        if not in_fence:
            in_fence = True
            fence_char = char
            fence_len = len(marker)
            open_idx = i
        elif char == fence_char and len(marker) >= fence_len:
            in_fence = False
            fence_char = ""
            fence_len = 0
            open_idx = -1
    if in_fence and open_idx >= 0:
        return "\n".join(lines[:open_idx]).rstrip()
    return str(text or "").rstrip()


try:
    import jieba as _jieba  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    _jieba = None


def _keyword_tokens(text: str) -> List[str]:
    """
    Tokenize mixed English/Chinese scientific text for lexical retrieval.

    Chinese words are hard without a segmenter, so we keep the whole CJK span
    and add 2-4 character shingles. This catches terms such as 震相检测,
    长宁地震, 速度层析 without adding an external dependency.
    """
    tokens: List[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            if _jieba is not None:
                for word in _jieba.cut(raw, cut_all=False):
                    word = word.strip()
                    if len(word) >= 2 and word not in _STOPWORDS:
                        tokens.append(word)
            elif len(raw) <= 6 and raw not in _STOPWORDS:
                tokens.append(raw)
            for n in (2, 3, 4):
                if len(raw) >= n:
                    tokens.extend(raw[i:i + n] for i in range(len(raw) - n + 1))
        elif len(raw) >= 2 and raw not in _STOPWORDS:
            tokens.append(raw)
    return tokens


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocChunk:
    chunk_id:   str   # "{doc_id}_{section}_{chunk}"
    doc_id:     str   # 12-char MD5 of (proj_folder/doc_name + file bytes)
    doc_name:   str   # original filename
    page:       int   # 0-based section/page index
    text:       str   # chunk text
    char_start: int = 0


@dataclass
class DocMeta:
    doc_id:      str
    doc_name:    str
    file_path:   str        # path of stored copy (empty for TF-IDF-only docs)
    n_pages:     int
    n_chunks:    int
    added_at:    str        # ISO-8601
    size_bytes:  int
    proj_folder: str = ""   # originating skill/ref folder; empty = manual upload
    source_type: str = "upload"   # "upload" | "skill_docs" | "ref_knowledge"


# ---------------------------------------------------------------------------
# SimpleRAG accessor (TF-IDF fallback)
# ---------------------------------------------------------------------------

def _get_simple_rag():
    """Return the SimpleRAG singleton; works under both sys.path layouts."""
    # Ensure web_app/ is on sys.path so simple_rag can be imported directly
    _here = str(Path(__file__).parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        from simple_rag import get_simple_rag as _f   # Flask: web_app/ in sys.path
        return _f()
    except ImportError:
        from web_app.simple_rag import get_simple_rag as _f  # type: ignore
        return _f()


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Unified knowledge base.

    Primary backend  : BGE-M3 embeddings + FAISS IndexFlatIP.
    Fallback backend : TF-IDF via SimpleRAG (used when the embedding model
                       cannot be loaded, or FAISS has no vectors).

    The fallback is resolved in a single method (_retrieve_core) so the rest
    of the class never needs to branch on which backend is active.
    """

    _instance: Optional["KnowledgeBase"] = None

    def __init__(self):
        self._chunks: dict[str, DocChunk] = {}
        self._docs:   dict[str, DocMeta]  = {}
        self._faiss: Optional[object]     = None   # FaissIndex | None
        self._load_state()

    @classmethod
    def get(cls) -> "KnowledgeBase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        if META_FILE.exists():
            try:
                raw = json.loads(META_FILE.read_text())
                for did, d in raw.get("docs", {}).items():
                    self._docs[did] = DocMeta(**d)
                for cid, c in raw.get("chunks", {}).items():
                    self._chunks[cid] = DocChunk(**c)
            except Exception as exc:
                print(f"[RAG] Warning: could not load metadata — {exc}", file=sys.stderr)

        if INDEX_FILE.exists() and IDMAP_FILE.exists():
            try:
                from rag_backends import FaissIndex
                self._faiss = FaissIndex.load(str(INDEX_FILE), str(IDMAP_FILE))
            except Exception as exc:
                print(f"[RAG] Warning: could not load FAISS index — {exc}", file=sys.stderr)
                self._faiss = None

        self._evict_missing_files()

    def _save_state(self):
        meta = {
            "docs":   {did: asdict(d) for did, d in self._docs.items()},
            "chunks": {cid: asdict(c) for cid, c in self._chunks.items()},
        }
        META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if self._faiss:
            self._faiss.save(str(INDEX_FILE), str(IDMAP_FILE))

    def _evict_missing_files(self):
        """Remove records whose stored file copies have disappeared."""
        stale = [
            did for did, m in self._docs.items()
            if m.file_path and not Path(m.file_path).exists()
        ]
        if not stale:
            return
        for did in stale:
            self._docs.pop(did, None)
            for cid in [c for c, ch in self._chunks.items() if ch.doc_id == did]:
                del self._chunks[cid]
            print(f"[RAG] Evicted stale document: {did}", file=sys.stderr)
        self._rebuild_faiss()
        self._save_state()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        path: str,
        progress_cb=None,
        chunk_size: int = 500,
        overlap: int = 50,
        proj_folder: str = "",
        source_type: str = "upload",
    ) -> DocMeta:
        """
        Index any supported file format into the knowledge base.

        Supported: .pdf  .doc  .docx  .md  .txt  .rst  .html  .htm

        Parameters
        ----------
        path        : absolute or relative file path
        progress_cb : optional callable(str) for progress messages
        chunk_size  : target chars per chunk
        overlap     : overlap chars on hard cuts
        proj_folder : logical grouping label (e.g. skill folder name)
        source_type : "upload" | "skill_docs" | "ref_knowledge"
        """
        from rag_extractors import extract_text, chunk_text

        def log(msg: str):
            if progress_cb:
                progress_cb(msg)

        path     = str(path)
        doc_name = Path(path).name
        sig      = f"{proj_folder}/{doc_name}".encode()
        doc_id   = hashlib.md5(sig + b"|" + Path(path).read_bytes()).hexdigest()[:12]

        if doc_id in self._docs:
            log(f"[skip] Already indexed: {doc_name}")
            return self._docs[doc_id]

        log(f"[parse] {doc_name}")
        pages   = extract_text(path)
        n_pages = len(pages)
        log(f"  {n_pages} section(s)")

        # Build chunks
        all_chunks: List[DocChunk] = []
        for sec_idx, sec_text in pages:
            for i, ct in enumerate(chunk_text(sec_text, chunk_size, overlap)):
                all_chunks.append(DocChunk(
                    chunk_id  = f"{doc_id}_{sec_idx}_{i}",
                    doc_id    = doc_id,
                    doc_name  = doc_name,
                    page      = sec_idx,
                    text      = ct,
                ))
        log(f"  {len(all_chunks)} chunk(s)")

        if not all_chunks:
            raise ValueError(f"No text extracted from: {doc_name}")

        # Try vector backend; fall back to TF-IDF
        use_faiss = self._try_faiss_ingest(
            all_chunks, log, proj_folder=proj_folder, source_type=source_type
        )

        # Store file copy (only for FAISS path; TF-IDF stores its own data)
        if use_faiss:
            dest = DOCS_DIR / f"{doc_id}_{doc_name}"
            shutil.copy2(path, dest)
            file_path = str(dest)
        else:
            file_path = ""

        meta = DocMeta(
            doc_id      = doc_id,
            doc_name    = doc_name,
            file_path   = file_path,
            n_pages     = n_pages,
            n_chunks    = len(all_chunks),
            added_at    = datetime.now(timezone.utc).isoformat(timespec="seconds"),
            size_bytes  = Path(path).stat().st_size,
            proj_folder = proj_folder,
            source_type = source_type,
        )
        self._docs[doc_id] = meta
        self._save_state()
        log(f"[done] {doc_name} ({len(all_chunks)} chunks, backend={'faiss' if use_faiss else 'tfidf'})")
        return meta

    # Backward-compatible alias
    def add_pdf(self, pdf_path: str, progress_cb=None, **kwargs) -> DocMeta:
        return self.add_document(pdf_path, progress_cb, **kwargs)

    def _try_faiss_ingest(
        self,
        chunks: List[DocChunk],
        log,
        proj_folder: str = "",
        source_type: str = "upload",
    ) -> bool:
        """
        Try to encode chunks with BGE-M3 and add to FAISS.
        Returns True on success, False if the embedding model is unavailable
        (in which case chunks are added to the SimpleRAG TF-IDF backend instead).
        """
        try:
            from rag_backends import EmbeddingModel, FaissIndex

            model  = EmbeddingModel.get()
            texts  = [c.text for c in chunks]
            vecs   = model.encode(texts)
            log(f"  embedding backend: {model.backend or 'unknown'}")
            log(f"  embedding dim: {vecs.shape[1]}")

            if self._faiss is None:
                self._faiss = FaissIndex(dim=vecs.shape[1])

            self._faiss.add(vecs, [c.chunk_id for c in chunks])
            for c in chunks:
                self._chunks[c.chunk_id] = c

            return True

        except RuntimeError as exc:
            log(f"[warn] Embedding model unavailable: {exc}")
            log("[info] Using TF-IDF fallback (SimpleRAG)")

            first = chunks[0]
            _get_simple_rag().add_document(chunks, {
                "doc_id":    first.doc_id,
                "doc_name":  first.doc_name,
                "file_path": "",
                "n_pages":   max(c.page for c in chunks) + 1,
                "n_chunks":  len(chunks),
                "added_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "size_bytes": 0,
                "proj_folder": proj_folder,
                "source_type": source_type,
            })
            return False

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_doc(self, doc_id: str) -> bool:
        """
        Remove a document and all its chunks.
        FAISS requires a full index rebuild (IndexFlatIP has no remove).
        Also removes from the TF-IDF backend if present there.
        """
        found = False

        # FAISS backend
        if doc_id in self._docs:
            meta = self._docs.pop(doc_id)
            for cid in [c for c, ch in self._chunks.items() if ch.doc_id == doc_id]:
                del self._chunks[cid]
            try:
                Path(meta.file_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._rebuild_faiss()
            self._save_state()
            found = True

        # TF-IDF backend (may also hold a copy)
        try:
            sr = _get_simple_rag()
            if doc_id in getattr(sr, "_docs", {}):
                sr.delete_document(doc_id)
                found = True
        except Exception:
            pass

        return found

    def _rebuild_faiss(self):
        """Rebuild FAISS from self._chunks (called after deletion)."""
        if not self._chunks:
            self._faiss = None
            return
        try:
            from rag_backends import EmbeddingModel, FaissIndex
            model = EmbeddingModel.get()
            cids  = list(self._chunks)
            vecs  = model.encode([self._chunks[c].text for c in cids])
            self._faiss = FaissIndex(dim=vecs.shape[1])
            self._faiss.add(vecs, cids)
        except Exception as exc:
            print(f"[RAG] Warning: could not rebuild FAISS index — {exc}", file=sys.stderr)
            self._faiss = None

    def clear(self):
        """Wipe the entire knowledge base (FAISS + TF-IDF + files)."""
        self._chunks.clear()
        self._docs.clear()
        self._faiss = None
        for f in DOCS_DIR.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        for p in (INDEX_FILE, META_FILE, IDMAP_FILE):
            p.unlink(missing_ok=True)
        try:
            _get_simple_rag().clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Retrieval — single fallback point
    # ------------------------------------------------------------------

    def _retrieve_core(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[DocChunk, float]]:
        """
        Core retrieval: FAISS if available, otherwise TF-IDF.
        This is the only method that needs to know which backend is active.
        """
        # --- FAISS path -------------------------------------------------
        if self._faiss and self._faiss.n_vectors > 0:
            try:
                from rag_backends import EmbeddingModel
                qvec = EmbeddingModel.get().encode([query])
                hits = self._faiss.search(qvec, top_k=top_k)
                return [
                    (self._chunks[cid], score)
                    for cid, score in hits
                    if score >= score_threshold and cid in self._chunks
                ]
            except RuntimeError:
                pass   # embedding model went away; fall through to TF-IDF

        # --- TF-IDF fallback path ---------------------------------------
        try:
            sr      = _get_simple_rag()
            tfidf_t = min(score_threshold, 0.05)   # TF-IDF scores are much lower
            return [
                (
                    DocChunk(
                        chunk_id  = m.get("chunk_id", "v"),
                        doc_id    = m.get("doc_id",   "v"),
                        doc_name  = m.get("doc_name", "Unknown"),
                        page      = m.get("page", 0),
                        text      = text,
                    ),
                    score,
                )
                for text, score, m in sr.retrieve(query, top_k)
                if score >= tfidf_t
            ]
        except Exception:
            return []

    def _keyword_candidates(self) -> List[DocChunk]:
        """Return chunks available to lexical retrieval, including TF-IDF-only docs."""
        chunks = list(self._chunks.values())
        seen = {c.chunk_id for c in chunks}
        try:
            sr = _get_simple_rag()
            for item in getattr(sr.db, "items", []):
                meta = getattr(item, "metadata", {}) or {}
                cid = meta.get("chunk_id") or f"simple_{len(seen)}"
                if cid in seen:
                    continue
                seen.add(cid)
                chunks.append(DocChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id") or getattr(item, "doc_id", "") or "simple",
                    doc_name=meta.get("doc_name") or "Unknown",
                    page=int(meta.get("page") or 0),
                    text=getattr(item, "text", "") or "",
                ))
        except Exception:
            pass
        return [c for c in chunks if c.text]

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Tuple[DocChunk, float]]:
        """
        BM25-style keyword retrieval.

        This is intentionally not TF-IDF cosine search: it keeps exact scientific
        terms, abbreviations, phase names, and Chinese jieba tokens visible, while
        removing corpus-wide high-frequency tokens automatically.
        """
        query_terms = list(dict.fromkeys(_keyword_tokens(query)))
        if not query_terms:
            return []

        chunks = self._keyword_candidates()
        if not chunks:
            return []

        token_counts: list[Counter] = []
        doc_freq: Counter = Counter()
        doc_lens: list[int] = []
        for chunk in chunks:
            counts = Counter(_keyword_tokens(chunk.text))
            token_counts.append(counts)
            doc_lens.append(sum(counts.values()) or 1)
            for tok in counts:
                doc_freq[tok] += 1

        n_docs = len(chunks)
        # Drop corpus-wide generic words. If that removes every query token,
        # keep the original query terms so short technical queries still work.
        useful_terms = [
            t for t in query_terms
            if doc_freq.get(t, 0) > 0 and (doc_freq[t] / max(n_docs, 1)) <= 0.55
        ]
        if not useful_terms:
            useful_terms = [t for t in query_terms if doc_freq.get(t, 0) > 0]
        if not useful_terms:
            return []

        avgdl = sum(doc_lens) / max(len(doc_lens), 1)
        k1, b = 1.45, 0.72
        q_lower = query.lower()
        scored: list[tuple[DocChunk, float]] = []
        for chunk, counts, dl in zip(chunks, token_counts, doc_lens):
            score = 0.0
            for term in useful_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                score += idf * (tf * (k1 + 1)) / max(denom, 1e-9)
            text_lower = chunk.text.lower()
            if q_lower and q_lower in text_lower:
                score += 0.8
            score += 0.08 * sum(1 for t in useful_terms if t in text_lower)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return []
        top = scored[:top_k]
        max_score = max(s for _, s in top) or 1.0
        return [(chunk, min(1.0, score / max_score)) for chunk, score in top]

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[DocChunk, float]]:
        """Fuse dense retrieval with BM25 keyword retrieval."""
        dense = self._retrieve_core(query, top_k=max(top_k * 4, 20), score_threshold=score_threshold)
        keyword = self._keyword_retrieve(query, top_k=max(top_k * 4, 20))
        if not dense:
            return keyword[:top_k]
        if not keyword:
            return dense[:top_k]

        max_dense = max((s for _, s in dense), default=1.0) or 1.0
        merged: dict[str, dict] = {}
        for rank, (chunk, score) in enumerate(dense, 1):
            rec = merged.setdefault(chunk.chunk_id, {"chunk": chunk, "dense": 0.0, "kw": 0.0, "rrf": 0.0})
            rec["dense"] = max(rec["dense"], score / max_dense)
            rec["rrf"] += 1.0 / (60 + rank)
        for rank, (chunk, score) in enumerate(keyword, 1):
            rec = merged.setdefault(chunk.chunk_id, {"chunk": chunk, "dense": 0.0, "kw": 0.0, "rrf": 0.0})
            rec["kw"] = max(rec["kw"], score)
            rec["rrf"] += 1.0 / (60 + rank)

        fused: list[Tuple[DocChunk, float]] = []
        for rec in merged.values():
            # Keyword gets a strong vote because scientific terms and Chinese
            # method names are often more precise than embedding similarity.
            score = 0.62 * rec["dense"] + 0.58 * rec["kw"] + 2.5 * rec["rrf"]
            fused.append((rec["chunk"], min(1.0, score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    @staticmethod
    def _norm_source(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _chunk_matches_filters(
        self,
        chunk: DocChunk,
        sources: Optional[Iterable[str]] = None,
        source_types: Optional[Iterable[str]] = None,
    ) -> bool:
        """Return True when chunk belongs to one of the requested logical sources."""
        meta = self._docs.get(chunk.doc_id)
        if source_types:
            wanted_types = {str(s) for s in source_types if str(s)}
            if not meta or meta.source_type not in wanted_types:
                return False
        if not sources:
            return True

        wanted = {self._norm_source(s) for s in sources if str(s).strip()}
        if not wanted:
            return True

        fields = [
            chunk.doc_id,
            chunk.doc_name,
            meta.doc_id if meta else "",
            meta.doc_name if meta else "",
            meta.proj_folder if meta else "",
            meta.source_type if meta else "",
        ]
        hay = {self._norm_source(v) for v in fields if v}
        return any(w in hay or any(w in h or h in w for h in hay) for w in wanted)

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        sources: Optional[Iterable[str]] = None,
        source_types: Optional[Iterable[str]] = None,
    ) -> List[Tuple[DocChunk, float]]:
        """Return [(DocChunk, score), ...] sorted by score descending."""
        candidate_k = top_k
        if sources or source_types:
            candidate_k = max(top_k * 8, 40)
        results = self._hybrid_retrieve(query, candidate_k, score_threshold)
        if sources or source_types:
            results = [
                (chunk, score) for chunk, score in results
                if self._chunk_matches_filters(chunk, sources=sources, source_types=source_types)
            ][:top_k]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def retrieve_relevant_docs(
        self,
        query: str,
        top_k: int = 8,
        score_threshold: float = 0.5,
        sources: Optional[Iterable[str]] = None,
        source_types: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """
        Return structured retrieval results suitable for display.

        Each dict: doc_name, page (1-based), score, text, chunk_id, doc_id.
        """
        return [
            {
                "doc_name": c.doc_name,
                "page":     c.page + 1,
                "score":    round(s, 4),
                "text":     c.text,
                "chunk_id": c.chunk_id,
                "doc_id":   c.doc_id,
            }
            for c, s in self.retrieve(
                query, top_k, score_threshold, sources=sources, source_types=source_types
            )
        ]

    def build_rag_context(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 3000,
        score_threshold: float = 0.5,
        sources: Optional[Iterable[str]] = None,
        source_types: Optional[Iterable[str]] = None,
    ) -> str:
        """
        Retrieve and format as a concise LLM context block (English).

        Returns an empty string when no relevant chunks are found.
        """
        hits = self.retrieve(
            query, top_k, score_threshold, sources=sources, source_types=source_types
        )
        if not hits:
            return ""

        lines = [
            "The following passages were retrieved from the knowledge base:\n",
            _RAG_CONTEXT_COMPLETE_ONLY_NOTE,
        ]
        total = 0
        for chunk, score in hits:
            chunk_text = _drop_incomplete_fenced_tail(chunk.text)
            if not chunk_text.strip():
                continue
            entry = (
                f"[Source: {chunk.doc_name}, section {chunk.page + 1}, "
                f"relevance {score:.2f}]\n{chunk_text}\n"
            )
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Status / listing
    # ------------------------------------------------------------------

    def list_docs(self) -> List[DocMeta]:
        docs = dict(self._docs)
        # Include any docs that ended up only in the TF-IDF backend
        try:
            sr = _get_simple_rag()
            for did, dm in getattr(sr, "_docs", {}).items():
                if did not in docs:
                    docs[did] = dm
        except Exception:
            pass
        return sorted(docs.values(), key=lambda d: d.added_at, reverse=True)

    @property
    def n_docs(self) -> int:
        return len(self.list_docs())

    @property
    def n_chunks(self) -> int:
        count = len(self._chunks)
        if not (self._faiss and self._faiss.n_vectors > 0):
            try:
                sr = _get_simple_rag()
                if getattr(sr, "_docs", {}):
                    count = max(count, sr.db.count_items())
            except Exception:
                pass
        return count

    @property
    def is_empty(self) -> bool:
        return self.n_chunks == 0

    def status(self) -> dict:
        n_vecs = self._faiss.n_vectors if self._faiss else 0
        if n_vecs == 0:
            try:
                sr = _get_simple_rag()
                if getattr(sr, "_docs", {}):
                    n_vecs = sr.db.count_items()
            except Exception:
                pass
        n_docs = self.n_docs
        n_chunks = self.n_chunks
        if n_docs == 0 and n_chunks == 0:
            n_vecs = 0
        return {
            "n_docs":    n_docs,
            "n_chunks":  n_chunks,
            "n_vectors": n_vecs,
            "kb_dir":    str(KB_DIR),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.get()


# ---------------------------------------------------------------------------
# CLI + test harness
# ---------------------------------------------------------------------------

def _run_tests():
    """Quick self-tests that require no external files."""
    import textwrap

    print("=" * 60)
    print("RAG Engine — self-test")
    print("=" * 60)

    # --- chunker --------------------------------------------------------
    print("\n[1/4] chunk_text")
    from rag_extractors import chunk_text
    sample = "First sentence. " * 40 + "\n\nSecond paragraph. " * 20
    chunks = chunk_text(sample, chunk_size=200, overlap=30)
    assert chunks, "No chunks produced"
    assert all(len(c) <= 300 for c in chunks), "Chunk too large"
    print(f"  OK — {len(chunks)} chunks, max {max(len(c) for c in chunks)} chars")

    # --- extractors -----------------------------------------------------
    print("\n[2/4] Markdown extractor")
    from rag_extractors import _extract_md
    import tempfile, os
    md_content = "# Section 1\n\nHello world.\n\n## Section 2\n\nMore text.\n"
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(md_content)
        f.flush()
        sections = _extract_md(f.name)
    os.unlink(f.name)
    assert len(sections) >= 2, f"Expected ≥2 sections, got {len(sections)}"
    print(f"  OK — {len(sections)} sections extracted")

    # --- embedding model ------------------------------------------------
    print("\n[3/4] EmbeddingModel")
    try:
        from rag_backends import EmbeddingModel
        em = EmbeddingModel.get()
        vecs = em.encode(["earthquake waveform", "seismic velocity model"])
        assert vecs.shape == (2, em.dim), f"Shape mismatch: {vecs.shape}"
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1)
        assert all(abs(n - 1.0) < 1e-4 for n in norms), "Vectors not L2-normalised"
        print(f"  OK — backend={em.backend}, dim={em.dim}, shape={vecs.shape}")
    except RuntimeError as exc:
        print(f"  SKIP — embedding model not available: {exc!s:.80}")

    # --- end-to-end retrieval -------------------------------------------
    print("\n[4/4] End-to-end ingestion + retrieval (in-memory)")
    import tempfile, os, textwrap
    txt = textwrap.dedent("""\
        FAISS is a library for efficient similarity search and clustering of dense vectors.
        It contains algorithms that search in sets of vectors of any size, up to ones that
        possibly do not fit in RAM.

        BGE-M3 is a multilingual embedding model from BAAI that supports dense, sparse,
        and ColBERT retrieval. It outputs 1024-dimensional dense vectors.

        TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a numerical
        statistic that reflects how important a word is to a document in a collection.
    """)
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write(txt)
        tmpfile = f.name

    # Use a throw-away KB instance so we don't pollute the real one
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._chunks = {}
    kb._docs   = {}
    kb._faiss  = None

    try:
        kb.add_document(tmpfile)
        hits = kb.retrieve("FAISS similarity search", top_k=3, score_threshold=0.0)
        assert hits, "No results returned"
        top_text = hits[0][0].text.lower()
        assert "faiss" in top_text or "similar" in top_text, \
            f"Top result does not mention FAISS: {top_text[:80]}"
        print(f"  OK — {len(hits)} hit(s), top score={hits[0][1]:.3f}")
        print(f"  Top: {hits[0][0].text[:80]!r}")
    finally:
        os.unlink(tmpfile)

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Engine — knowledge base management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag_engine.py --status
              python rag_engine.py --add paper.pdf
              python rag_engine.py --query "seismic waveform processing" --top-k 5
              python rag_engine.py --test
        """) if False else "",
    )
    parser.add_argument("--status",  action="store_true", help="print KB stats")
    parser.add_argument("--add",     metavar="FILE",      help="index a document")
    parser.add_argument("--query",   metavar="TEXT",      help="run a retrieval query")
    parser.add_argument("--test",    action="store_true", help="run self-tests")
    parser.add_argument("--top-k",   type=int, default=5, dest="top_k")
    parser.add_argument("--threshold", type=float, default=0.0, dest="threshold",
                        help="score threshold for --query (default 0.0 = show all)")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        sys.exit(0)

    kb = get_knowledge_base()

    if args.status:
        print(json.dumps(kb.status(), indent=2))

    if args.add:
        kb.add_document(args.add, progress_cb=print)

    if args.query:
        hits = kb.retrieve(args.query, top_k=args.top_k, score_threshold=args.threshold)
        if not hits:
            print("No results found.")
        else:
            for rank, (chunk, score) in enumerate(hits, 1):
                print(f"\n[{rank}] score={score:.4f}  {chunk.doc_name}  p{chunk.page+1}")
                print(f"    {chunk.text[:160]}")

    if not any([args.status, args.add, args.query, args.test]):
        parser.print_help()
