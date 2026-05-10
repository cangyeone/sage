"""
rag_backends.py — Embedding model and FAISS vector index for the RAG pipeline.

EmbeddingModel
    Lazy-loads BGE-M3 with a fallback chain:
      1. local ONNX model  (onnxruntime + tokenizers, no transformers needed)
      2. FlagEmbedding  (BGEM3FlagModel)
      3. sentence-transformers  (SentenceTransformer)
      4. transformers + safetensors  (for environments where torch < 2.6
         blocks torch.load due to CVE-2025-32434)
    Call EmbeddingModel.get().encode(texts) to use.

FaissIndex
    Thin wrapper around faiss.IndexFlatIP (inner-product; cosine similarity
    after L2-normalisation of all vectors).

get_embedding_model_path()
    Reads ~/.seismicx/config.json → embedding.model_path.
    Defaults to project-local "open_models/bge-m3" when present, otherwise
    "BAAI/bge-m3" (HuggingFace hub).
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
    Falls back to project-local open_models/bge-m3 if present; otherwise
    "BAAI/bge-m3" if the config is absent or the key is missing.
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
    try:
        project_model = Path(__file__).resolve().parents[1] / "open_models" / "bge-m3"
        if project_model.exists():
            return str(project_model)
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
        onnx_err: Optional[str] = None
        flag_err: Optional[str] = None
        st_err:   Optional[str] = None

        # --- Attempt 0: local ONNX BGE-M3 --------------------------------
        # ModelScope/HuggingFace snapshots often include onnx/model.onnx.
        # This path avoids transformers + huggingface_hub version conflicts.
        try:
            local_root = Path(model_path).expanduser()
            if local_root.exists():
                wrapper = _load_local_onnx_bge_m3(local_root)
                if wrapper is not None:
                    self._model = wrapper
                    self._backend = "onnx"
                    return
        except ImportError as e:
            onnx_err = f"ImportError: {e}"
        except Exception as e:
            onnx_err = f"{type(e).__name__}: {e}"

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
        diag.append(f"  model_path             → {model_path}")
        if onnx_err:
            diag.append(f"  local ONNX             → {onnx_err}")
        elif Path(str(model_path)).expanduser().exists():
            diag.append("  local ONNX             → onnx/model.onnx or tokenizer.json not found")
        else:
            diag.append("  local ONNX             → skipped; model_path is not a local directory")
        if flag_err:
            diag.append(f"  FlagEmbedding          → {flag_err}")
        else:
            diag.append("  FlagEmbedding          → not installed")
        if st_err:
            diag.append(f"  sentence-transformers  → {st_err}")
        else:
            diag.append("  sentence-transformers  → not installed")

        _is_hub_conflict = lambda msg: "huggingface-hub" in str(msg) and ("<1.0" in str(msg) or "found huggingface-hub" in str(msg))
        if _is_cve(flag_err) or _is_cve(st_err):
            diag += [
                "",
                "Cause: torch < 2.6 (CVE-2025-32434 blocks torch.load).",
                "Fix: upgrade torch:",
                f"  {python} -m pip install 'torch>=2.6'",
            ]
        elif _is_hub_conflict(flag_err) or _is_hub_conflict(st_err):
            diag += [
                "",
                "Cause: dependency version conflict in the current Python environment.",
                "Fix one of:",
                f"  {python} -m pip install 'huggingface-hub>=0.34,<1.0'",
                "  Or download a local BGE-M3 directory containing onnx/model.onnx and tokenizer.json:",
                "    modelscope download --model BAAI/bge-m3 --local_dir open_models/bge-m3",
            ]
        else:
            diag += [
                "",
                "Recommended local download:",
                "  modelscope download --model BAAI/bge-m3 --local_dir open_models/bge-m3",
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
        elif self._backend in {"transformers-safetensors", "onnx"}:
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


def _load_local_onnx_bge_m3(model_root: Path):
    """Return an ONNX wrapper when a local BGE-M3 snapshot has ONNX assets."""
    model_root = model_root.expanduser().resolve()
    onnx_path = model_root / "onnx" / "model.onnx"
    tokenizer_path = model_root / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = model_root / "onnx" / "tokenizer.json"
    if not onnx_path.exists() or not tokenizer_path.exists():
        return None

    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    from tokenizers import Tokenizer  # type: ignore

    class _OnnxBgeM3Wrapper:
        def __init__(self, root: Path, model_file: Path, tok_file: Path):
            self.root = root
            self.tokenizer = Tokenizer.from_file(str(tok_file))
            self.session = ort.InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )
            self.input_names = {i.name for i in self.session.get_inputs()}
            self.output_names = [o.name for o in self.session.get_outputs()]

        def encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512):
            all_vecs = []
            clean_texts = [str(t or "") for t in texts]
            for start in range(0, len(clean_texts), max(1, batch_size)):
                batch = clean_texts[start:start + max(1, batch_size)]
                encs = self.tokenizer.encode_batch(batch)
                ids = [list(enc.ids)[:max_length] for enc in encs]
                if not ids:
                    continue
                max_len = max(1, max(len(x) for x in ids))
                input_ids = np.zeros((len(ids), max_len), dtype=np.int64)
                attention_mask = np.zeros((len(ids), max_len), dtype=np.int64)
                for row, row_ids in enumerate(ids):
                    if not row_ids:
                        row_ids = [0]
                    n = min(len(row_ids), max_len)
                    input_ids[row, :n] = np.asarray(row_ids[:n], dtype=np.int64)
                    attention_mask[row, :n] = 1

                feed = {"input_ids": input_ids, "attention_mask": attention_mask}
                feed = {k: v for k, v in feed.items() if k in self.input_names}
                if "sentence_embedding" in self.output_names:
                    vec = self.session.run(["sentence_embedding"], feed)[0]
                else:
                    token_embeddings = self.session.run(None, feed)[0]
                    mask = attention_mask[..., None].astype("float32")
                    vec = (token_embeddings * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
                all_vecs.append(vec.astype("float32"))
            if not all_vecs:
                return np.zeros((0, 1024), dtype="float32")
            vecs = np.vstack(all_vecs).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.maximum(norms, 1e-9)

    return _OnnxBgeM3Wrapper(model_root, onnx_path, tokenizer_path)


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
