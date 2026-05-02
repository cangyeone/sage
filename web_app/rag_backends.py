"""
rag_backends.py — Embedding model and FAISS vector index for the RAG pipeline.

EmbeddingModel
    Lazy-loads BGE-M3 with a three-step fallback chain:
      1. FlagEmbedding  (BGEM3FlagModel)
      2. sentence-transformers  (SentenceTransformer)
      3. transformers + safetensors  (for environments where torch < 2.6
         blocks torch.load due to CVE-2025-32434)
    Call EmbeddingModel.get().encode(texts) to use.

FaissIndex
    Thin wrapper around faiss.IndexFlatIP (inner-product; cosine similarity
    after L2-normalisation of all vectors).

get_embedding_model_path()
    Reads ~/.seismicx/config.json → embedding.model_path.
    Defaults to "BAAI/bge-m3" (HuggingFace hub).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def get_embedding_model_path() -> str:
    """
    Read the embedding model path from ~/.seismicx/config.json.
    Falls back to "BAAI/bge-m3" if the config is absent or the key is missing.
    """
    try:
        cfg_file = Path.home() / ".seismicx" / "config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            path = cfg.get("embedding", {}).get("model_path", "").strip()
            if path:
                return path
    except Exception:
        pass
    return "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# EmbeddingModel
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Singleton that lazily loads BGE-M3 and exposes a uniform encode() interface.

    Usage
    -----
    vecs = EmbeddingModel.get().encode(["text one", "text two"])
    # vecs is a float32 ndarray of shape (2, dim), L2-normalised.
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self):
        self._model   = None
        self._backend: Optional[str] = None
        self.dim = 1024  # BGE-M3 dense output dimension

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Drop the singleton so the next get() reloads the model."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return

        model_path = get_embedding_model_path()
        flag_err: Optional[str] = None
        st_err:   Optional[str] = None

        # --- Attempt 1: FlagEmbedding -----------------------------------
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
            self._model   = BGEM3FlagModel(model_path, use_fp16=True, device="cpu")
            self._backend = "flag"
            return
        except ImportError as e:
            flag_err = f"ImportError: {e}"
        except Exception as e:
            flag_err = f"{type(e).__name__}: {e}"

        # --- Attempt 2: sentence-transformers ---------------------------
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model   = SentenceTransformer(model_path, device="cpu")
            self._backend = "st"
            return
        except ImportError as e:
            st_err = f"ImportError: {e}"
        except Exception as e:
            st_err = f"{type(e).__name__}: {e}"

        # --- Attempt 3: transformers + safetensors (CVE-2025-32434) ----
        # torch < 2.6 forbids torch.load; safetensors format is unaffected.
        _is_cve = lambda msg: any(
            kw in str(msg)
            for kw in ("CVE-2025-32434", "torch.load", "weights_only")
        )
        if _is_cve(flag_err) or _is_cve(st_err):
            # 3a: sentence-transformers with use_safetensors=True
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model   = SentenceTransformer(
                    model_path, device="cpu",
                    model_kwargs={"use_safetensors": True},
                )
                self._backend = "st-safetensors"
                return
            except Exception:
                pass

            # 3b: raw transformers AutoModel + safetensors
            try:
                from transformers import AutoTokenizer, AutoModel  # type: ignore
                import torch

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(
                    model_path,
                    use_safetensors=True,
                    torch_dtype=torch.float32,
                )
                model.eval()

                class _Wrapper:
                    def __init__(self, tok, mod):
                        self._tok, self._mod = tok, mod

                    def encode(self, texts: List[str]):
                        import torch, numpy as np  # noqa: F811
                        inputs = self._tok(
                            texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt",
                        )
                        with torch.no_grad():
                            out = self._mod(**inputs)
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                        v = vecs.numpy()
                        norms = np.linalg.norm(v, axis=1, keepdims=True)
                        return v / np.maximum(norms, 1e-9)

                self._model   = _Wrapper(tokenizer, model)
                self._backend = "transformers-safetensors"
                return
            except Exception:
                pass

        # --- All attempts failed ----------------------------------------
        import sys
        python = sys.executable
        diag = ["Could not load the embedding model. Diagnostics:"]
        if flag_err:
            diag.append(f"  FlagEmbedding          → {flag_err}")
        else:
            diag.append("  FlagEmbedding          → not installed")
        if st_err:
            diag.append(f"  sentence-transformers  → {st_err}")
        else:
            diag.append("  sentence-transformers  → not installed")

        if _is_cve(flag_err) or _is_cve(st_err):
            diag += [
                "",
                "Cause: torch < 2.6 (CVE-2025-32434 blocks torch.load).",
                "Fix: upgrade torch:",
                f"  {python} -m pip install 'torch>=2.6'",
            ]
        else:
            diag += [
                "",
                "Install one of:",
                f"  {python} -m pip install FlagEmbedding",
                f"  {python} -m pip install sentence-transformers",
            ]
        raise RuntimeError("\n".join(diag))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        """
        Encode a list of strings into L2-normalised float32 vectors.

        Returns
        -------
        numpy.ndarray of shape (len(texts), dim)
        """
        import numpy as np  # type: ignore
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
        elif self._backend == "transformers-safetensors":
            vecs = self._model.encode(texts)          # already L2-normalised
        else:
            vecs = self._model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True
            )

        vecs = np.array(vecs, dtype="float32")
        # Ensure L2 normalisation regardless of backend
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    @property
    def backend(self) -> Optional[str]:
        return self._backend


# ---------------------------------------------------------------------------
# FaissIndex
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    Thin wrapper around faiss.IndexFlatIP.

    All vectors must be L2-normalised before insertion; inner product then
    equals cosine similarity.
    """

    def __init__(self, dim: int = 1024):
        import faiss  # type: ignore
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)
        self._id_map: List[str] = []   # FAISS position → chunk_id

    # ------------------------------------------------------------------

    def add(self, vectors: "np.ndarray", chunk_ids: List[str]):
        import numpy as np  # type: ignore
        self.index.add(np.asarray(vectors, dtype="float32"))
        self._id_map.extend(chunk_ids)

    def search(self, query_vec: "np.ndarray", top_k: int = 5) -> List[Tuple[str, float]]:
        import numpy as np  # type: ignore
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))
        return results

    # ------------------------------------------------------------------

    def save(self, index_path: str, idmap_path: str):
        import faiss, json  # type: ignore  # noqa: F811
        faiss.write_index(self.index, index_path)
        Path(idmap_path).write_text(json.dumps(self._id_map))

    @classmethod
    def load(cls, index_path: str, idmap_path: str, dim: int = 1024) -> "FaissIndex":
        import faiss, json  # type: ignore  # noqa: F811
        obj = cls(dim)
        obj.index    = faiss.read_index(index_path)
        obj._id_map  = json.loads(Path(idmap_path).read_text())
        return obj

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal
