#!/usr/bin/env python3
"""
rag_engine.py — BGE-M3 + FAISS 知识库引擎

功能：
  - 用 BGE-M3 对 PDF 文档进行 chunk-level 向量化
  - 用 FAISS 存储向量，支持持久化
  - 对话时检索最相关的 chunk 作为 RAG 上下文

存储位置：~/.seismicx/knowledge/
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ── 路径常量 ──────────────────────────────────────────────────────────────────

KB_DIR      = Path.home() / ".seismicx" / "knowledge"
INDEX_FILE  = KB_DIR / "faiss_index.bin"
META_FILE   = KB_DIR / "metadata.json"
DOCS_DIR    = KB_DIR / "docs"

for _d in [KB_DIR, DOCS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── 数据模型 ──────────────────────────────────────────────────────────────────

@dataclass
class DocChunk:
    chunk_id:  str          # 唯一 ID = doc_hash + "_" + chunk_index
    doc_id:    str          # 文档哈希
    doc_name:  str          # 原始文件名
    page:      int          # 页码（0-based）
    text:      str          # chunk 文本内容
    char_start: int = 0

@dataclass
class DocMeta:
    doc_id:     str
    doc_name:   str
    file_path:  str         # 副本在 DOCS_DIR 的路径
    n_pages:    int
    n_chunks:   int
    added_at:   str
    size_bytes: int


# ── 文本提取与分块 ─────────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    返回 [(page_index, text), ...] 列表。
    优先使用 pdfminer.six，回退到 PyMuPDF。
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        pages = []
        for i, layout in enumerate(extract_pages(pdf_path)):
            texts = []
            for element in layout:
                if isinstance(element, LTTextContainer):
                    texts.append(element.get_text())
            pages.append((i, " ".join(texts)))
        return pages
    except ImportError:
        pass

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = [(i, page.get_text()) for i, page in enumerate(doc)]
        doc.close()
        return pages
    except ImportError:
        raise RuntimeError(
            "需要 pdfminer.six 或 PyMuPDF 来解析 PDF：\n"
            "  pip install pdfminer.six\n"
            "  或 pip install pymupdf"
        )


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """简单按字符切分，保留语义边界（按句号换行切）。"""
    # 先按段落/句子切
    paragraphs = re.split(r'\n{2,}|(?<=[。！？.!?])\s', text)
    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) <= chunk_size:
            buf += (" " if buf else "") + para
        else:
            if buf:
                chunks.append(buf)
            # 如果单段落超长，强制切分
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            buf = para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) > 20]


# ── 嵌入模型（BGE-M3）─────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    懒加载 BGE-M3 模型。
    优先使用 FlagEmbedding，回退到 sentence-transformers。
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self):
        self._model = None
        self._backend = None
        self.dim = 1024  # BGE-M3 输出维度

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is not None:
            return

        # 尝试 FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=True,
                device="cpu",
            )
            self._backend = "flag"
            return
        except ImportError:
            pass

        # 回退到 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "BAAI/bge-m3",
                device="cpu",
            )
            self._backend = "st"
            return
        except ImportError:
            pass

        raise RuntimeError(
            "未找到嵌入模型库，请安装其中一个：\n"
            "  pip install FlagEmbedding\n"
            "  或 pip install sentence-transformers"
        )

    def encode(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        import numpy as np
        self._load()

        if self._backend == "flag":
            result = self._model.encode(
                texts,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = result["dense_vecs"]
        else:
            vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

        vecs = np.array(vecs, dtype="float32")
        # L2 归一化（内积 = cosine）
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms


# ── FAISS 索引封装 ─────────────────────────────────────────────────────────────

class FaissIndex:
    """轻量 FAISS 封装，支持增量添加与持久化。"""

    def __init__(self, dim: int = 1024):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 内积（向量已归一化，等价 cosine）
        self._id_map: List[str] = []         # faiss 位置 → chunk_id

    def add(self, vectors: "np.ndarray", chunk_ids: List[str]):
        import numpy as np
        vecs = np.asarray(vectors, dtype="float32")
        self.index.add(vecs)
        self._id_map.extend(chunk_ids)

    def search(self, query_vec: "np.ndarray", top_k: int = 5) -> List[Tuple[str, float]]:
        import numpy as np
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def save(self, index_path: str, idmap_path: str):
        import faiss, json
        faiss.write_index(self.index, index_path)
        with open(idmap_path, "w") as f:
            json.dump(self._id_map, f)

    @classmethod
    def load(cls, index_path: str, idmap_path: str, dim: int = 1024) -> "FaissIndex":
        import faiss, json
        obj = cls(dim)
        obj.index = faiss.read_index(index_path)
        with open(idmap_path) as f:
            obj._id_map = json.load(f)
        return obj

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal


# ── 主知识库类 ────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    统一知识库接口。

    Usage
    -----
    kb = KnowledgeBase.get()
    kb.add_pdf("/path/to/paper.pdf", progress_cb=print)
    chunks = kb.retrieve("如何计算b值", top_k=5)
    """

    _instance: Optional["KnowledgeBase"] = None

    def __init__(self):
        self._chunks: dict[str, DocChunk] = {}   # chunk_id → DocChunk
        self._docs:   dict[str, DocMeta]  = {}   # doc_id   → DocMeta
        self._faiss: Optional[FaissIndex] = None
        self._dirty = False
        self._load_state()

    @classmethod
    def get(cls) -> "KnowledgeBase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 状态持久化 ─────────────────────────────────────────────────────────────

    def _idmap_path(self) -> str:
        return str(KB_DIR / "id_map.json")

    def _load_state(self):
        if META_FILE.exists():
            try:
                raw = json.loads(META_FILE.read_text())
                for did, d in raw.get("docs", {}).items():
                    self._docs[did] = DocMeta(**d)
                for cid, c in raw.get("chunks", {}).items():
                    self._chunks[cid] = DocChunk(**c)
            except Exception:
                pass

        idmap = self._idmap_path()
        if INDEX_FILE.exists() and os.path.exists(idmap):
            try:
                self._faiss = FaissIndex.load(str(INDEX_FILE), idmap)
            except Exception:
                self._faiss = None

    def _save_state(self):
        meta = {
            "docs":   {did: asdict(d) for did, d in self._docs.items()},
            "chunks": {cid: asdict(c) for cid, c in self._chunks.items()},
        }
        META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if self._faiss:
            self._faiss.save(str(INDEX_FILE), self._idmap_path())
        self._dirty = False

    # ── 文档操作 ─────────────────────────────────────────────────────────────

    def add_pdf(
        self,
        pdf_path: str,
        progress_cb=None,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> DocMeta:
        """
        解析 PDF → 分块 → BGE-M3 编码 → 加入 FAISS。
        返回 DocMeta。
        """
        def _log(msg):
            if progress_cb:
                progress_cb(msg)

        pdf_path = str(pdf_path)
        doc_name = Path(pdf_path).name

        # 计算文档 hash
        h = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()[:12]
        doc_id = h

        if doc_id in self._docs:
            _log(f"⚠ 文档已存在于知识库：{doc_name}，跳过")
            return self._docs[doc_id]

        _log(f"📄 解析 PDF：{doc_name}")
        pages = _extract_pdf_text(pdf_path)
        n_pages = len(pages)
        _log(f"   共 {n_pages} 页")

        # 分块
        all_chunks: List[DocChunk] = []
        for page_idx, page_text in pages:
            sub_chunks = _chunk_text(page_text, chunk_size, overlap)
            for i, chunk_text in enumerate(sub_chunks):
                cid = f"{doc_id}_{page_idx}_{i}"
                all_chunks.append(DocChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_idx,
                    text=chunk_text,
                ))
        _log(f"   切分为 {len(all_chunks)} 个 chunk")

        if not all_chunks:
            raise ValueError("PDF 中未提取到有效文本，可能是扫描版 PDF（需 OCR）")

        # 编码
        _log("🔢 BGE-M3 向量化中…")
        model = EmbeddingModel.get()
        texts  = [c.text for c in all_chunks]
        vecs   = model.encode(texts)
        _log(f"   向量维度: {vecs.shape}")

        # 加入 FAISS
        if self._faiss is None:
            self._faiss = FaissIndex(dim=vecs.shape[1])
        chunk_ids = [c.chunk_id for c in all_chunks]
        self._faiss.add(vecs, chunk_ids)

        # 注册 chunk
        for chunk in all_chunks:
            self._chunks[chunk.chunk_id] = chunk

        # 复制 PDF 到知识库目录
        dest = DOCS_DIR / f"{doc_id}_{doc_name}"
        shutil.copy2(pdf_path, dest)

        # 注册文档
        meta = DocMeta(
            doc_id=doc_id,
            doc_name=doc_name,
            file_path=str(dest),
            n_pages=n_pages,
            n_chunks=len(all_chunks),
            added_at=datetime.now().isoformat(timespec="seconds"),
            size_bytes=Path(pdf_path).stat().st_size,
        )
        self._docs[doc_id] = meta
        self._save_state()
        _log(f"✅ 已加入知识库：{doc_name}（{len(all_chunks)} chunks）")
        return meta

    def delete_doc(self, doc_id: str) -> bool:
        """
        删除文档及其所有 chunk。
        注意：FAISS IndexFlatIP 不支持按 ID 删除，需要重建索引。
        """
        if doc_id not in self._docs:
            return False

        meta = self._docs.pop(doc_id)
        # 移除 chunk
        to_remove = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in to_remove:
            del self._chunks[cid]

        # 删除文件副本
        try:
            Path(meta.file_path).unlink(missing_ok=True)
        except Exception:
            pass

        # 重建 FAISS 索引（只保留剩余 chunk 的向量）
        self._rebuild_index()
        self._save_state()
        return True

    def _rebuild_index(self):
        """从剩余 chunks 重建 FAISS 索引。"""
        if not self._chunks:
            self._faiss = None
            return

        model  = EmbeddingModel.get()
        cids   = list(self._chunks.keys())
        texts  = [self._chunks[cid].text for cid in cids]
        vecs   = model.encode(texts)
        dim    = vecs.shape[1]
        self._faiss = FaissIndex(dim=dim)
        self._faiss.add(vecs, cids)

    def clear(self):
        """清空整个知识库。"""
        self._chunks.clear()
        self._docs.clear()
        self._faiss = None
        for f in DOCS_DIR.iterdir():
            f.unlink(missing_ok=True)
        INDEX_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        Path(self._idmap_path()).unlink(missing_ok=True)

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Tuple[DocChunk, float]]:
        """
        检索最相关的 chunk。
        返回 [(DocChunk, score), ...]，按 score 降序。
        """
        if not self._faiss or self._faiss.n_vectors == 0:
            return []

        model = EmbeddingModel.get()
        qvec  = model.encode([query])
        hits  = self._faiss.search(qvec, top_k=top_k)

        results = []
        for cid, score in hits:
            if score < score_threshold:
                continue
            if cid in self._chunks:
                results.append((self._chunks[cid], score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def build_rag_context(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> str:
        """
        检索 + 格式化为 LLM 系统提示可用的上下文段落。
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""

        lines = ["以下是从知识库中检索到的相关内容，请参考这些内容回答用户问题：\n"]
        total = 0
        for chunk, score in hits:
            entry = (
                f"【来源：{chunk.doc_name}，第 {chunk.page + 1} 页，"
                f"相关度 {score:.2f}】\n{chunk.text}\n"
            )
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def list_docs(self) -> List[DocMeta]:
        return sorted(self._docs.values(), key=lambda d: d.added_at, reverse=True)

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self.n_chunks == 0

    def status(self) -> dict:
        return {
            "n_docs":   self.n_docs,
            "n_chunks": self.n_chunks,
            "n_vectors": self._faiss.n_vectors if self._faiss else 0,
            "kb_dir":   str(KB_DIR),
        }


# ── 单例访问 ──────────────────────────────────────────────────────────────────

def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.get()


if __name__ == "__main__":
    import sys
    kb = get_knowledge_base()
    print("知识库状态:", kb.status())

    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        kb.add_pdf(pdf, progress_cb=print)
        print("\n检索测试:")
        hits = kb.retrieve("地震波形处理", top_k=3)
        for chunk, score in hits:
            print(f"  [{score:.3f}] {chunk.doc_name} p{chunk.page}: {chunk.text[:80]}...")
