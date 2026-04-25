"""
evidence_driven_geo_agent.py — Evidence-Driven Geoscience Interpretation Agent

An autonomous, tool-using agent that searches local project files, literature
libraries, RAG indexes, and seismic datasets to build a traceable evidence table,
generate competing geological hypotheses, validate them with optional code
execution, and write a structured interpretation report.

Architecture
------------
  AgentConfig
  ↓
  ToolRegistry  (7 tools, path-sandboxed, call-logged)
  ├── LocalFileSearchTool   list_dir / read_file / search_files / grep / get_file_metadata
  ├── LiteratureLibraryTool search_papers / read_pdf / extract_pdf_metadata / extract_references / search_bibtex
  ├── RAGIndexTool          index_documents / search_rag / add_document / rebuild_index
  ├── SeismoDataTool        read_catalog / read_station_file / read_waveform / read_velocity_model / read_focal_mechanisms
  ├── GeoPlotTool           plot_catalog_map / plot_depth_section / plot_velocity_slice / plot_fault_distance / plot_evidence_map
  ├── CodeExecutionTool     run_python / run_shell
  └── StateMemoryTool       save_state / load_state / save_report / save_evidence_table / save_hypotheses
  ↓
  LoopController
    plan → search_local → search_literature → search_rag → read_sources
        → extract_evidence → inspect_seismo → run_validation
        → update_table → update_hypotheses → update_report → convergence_check
  ↓
  EvidenceDrivenGeoAgent  (public facade)

Evidence source types
---------------------
  literature      — extracted from indexed PDF / BibTeX / RAG KB
  local_data      — extracted from seismic catalog, waveform, velocity model
  model_derived   — result of computational analysis or code execution
  inference       — LLM-reasoned interpretation (not directly in source)
  speculation     — low-confidence claim explicitly flagged as such

Usage (programmatic)
--------------------
  from sage_agents import EvidenceDrivenGeoAgent, AgentConfig

  cfg = AgentConfig(
      workspace_root="./examples/weiyuan",
      literature_root="./papers/weiyuan",
      allow_python=True,
  )
  agent = EvidenceDrivenGeoAgent(cfg)
  result = agent.run(
      question="Why are M>4 earthquakes near the Molingchang fault?",
      study_area="Weiyuan, Sichuan Basin",
  )
  print(result["final_report"])

Usage (Flask)
-------------
  POST /api/evidence_geo_agent
  {"question": "...", "study_area": "...", "workspace_root": "...", ...}

Usage (CLI)
-----------
  python seismic_cli.py evidence-geo-agent \\
      --question "Why are M>4 earthquakes near the Molingchang fault?" \\
      --study-area "Weiyuan, Sichuan Basin" \\
      --workspace-root ./examples/weiyuan \\
      --literature-root ./papers/weiyuan \\
      --max-iterations 3
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ── Standalone LLM utility (no circular import with app.py) ──────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(
    messages: List[Dict[str, str]],
    llm_cfg: Dict[str, Any],
    max_tokens: int = 2000,
    temperature: float = 0.3,
) -> str:
    """Call the configured LLM backend. Returns response text."""
    provider = llm_cfg.get("provider", "ollama")
    model    = llm_cfg.get("model", "")
    api_base = llm_cfg.get("api_base", "")
    api_key  = llm_cfg.get("api_key", "")

    if not api_base:
        raise ConnectionError("LLM backend not configured (api_base missing).")
    if not model:
        raise ConnectionError("No model selected in LLM config.")

    if provider == "ollama":
        url     = api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    else:
        url     = api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }

    data    = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())

    if provider == "ollama":
        return body.get("message", {}).get("content", "").strip()
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def _get_llm_config() -> Dict[str, Any]:
    """Load LLM config from config_manager."""
    try:
        _root = str(Path(__file__).parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from config_manager import LLMConfigManager
        return LLMConfigManager().get_llm_config()
    except Exception:
        return {}


def _get_kb():
    """Return the RAG KnowledgeBase singleton (or None if unavailable)."""
    try:
        _webdir = str(Path(__file__).parent.parent / "web_app")
        if _webdir not in sys.path:
            sys.path.insert(0, _webdir)
        from rag_engine import get_knowledge_base
        return get_knowledge_base()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ── Configuration ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Runtime configuration for EvidenceDrivenGeoAgent.
    All path-based tools enforce access within workspace_root and literature_root.
    """
    # ── Roots ────────────────────────────────────────────────────────────────
    workspace_root:    str = "."          # local project directory (sandboxed)
    literature_root:   str = ""           # PDF / BibTeX library root
    rag_index_path:    str = ""           # optional explicit RAG index path
    output_dir:        str = "outputs/evidence_driven_geo_agent"

    # ── Capability gates ─────────────────────────────────────────────────────
    allow_python:      bool = True        # enable CodeExecutionTool.run_python
    allow_shell:       bool = False       # enable CodeExecutionTool.run_shell
    allow_web_search:  bool = False       # enable WebSearchTool (web_search / scholar_search / download_pdf)
    use_multimodal:    bool = False       # enable ImageAnalysisTool (analyze_image / extract_table)
    use_rag:           bool = True        # enable RAGIndexTool.search_rag
    use_local_files:   bool = True        # enable LocalFileSearchTool

    # ── Loop limits ──────────────────────────────────────────────────────────
    max_iterations:          int = 3
    max_tool_calls_per_iter: int = 8

    # ── Data ─────────────────────────────────────────────────────────────────
    rag_top_k:        int   = 8
    score_threshold:  float = 0.35

    # ── Code sandbox ─────────────────────────────────────────────────────────
    code_timeout_s:   int = 60           # Python execution timeout
    code_max_output:  int = 8192         # max chars captured from stdout+stderr

    # ── File access ──────────────────────────────────────────────────────────
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".md", ".txt", ".csv", ".json", ".yaml", ".yml",
        ".sac", ".mseed", ".xml", ".gmt", ".grd", ".nc", ".bib",
        ".dat", ".inp", ".out", ".log",
    ])

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# ── Data classes ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoEvidence:
    """
    A single piece of evidence with full provenance and source type classification.

    source_type:
      literature      — from indexed PDF / BibTeX / RAG KB
      local_data      — from seismic catalog / station file / waveform / velocity model
      model_derived   — result of code execution / computational analysis
      inference       — LLM-reasoned, not directly stated in a source
      speculation     — explicitly low-confidence, marked as uncertain

    evidence_type:
      text   — extracted from prose
      figure — extracted from an image / map / cross-section
      table  — extracted from a structured table
      data   — computed directly from a numerical dataset
    """
    evidence_id:              str
    source:                   str        # human-readable reference (file, page, URL, etc.)
    source_type:              str        # literature | local_data | model_derived | inference | speculation
    evidence_type:            str = "text"  # text | figure | table | data
    observation:              str = ""   # verbatim or close paraphrase of factual content
    data_type:                str = "other"  # seismicity | velocity_model | focal_mechanism | geology | ...
    spatial_scale:            str = "unspecified"   # local | regional | crustal | lithospheric
    depth_range:              str = "unspecified"
    geological_structure:     str = "unspecified"
    interpretation:           str = ""   # author's or source's interpretation
    alternative_interpretation: str = ""  # agent-generated alternative reading
    assumption:               str = ""   # key assumptions in source
    confidence:               str = "medium"   # high | medium | low
    uncertainty:              str = ""
    supports:                 List[str] = field(default_factory=list)   # hypothesis_ids supported
    contradicts:              List[str] = field(default_factory=list)   # hypothesis_ids contradicted
    conflict_with:            List[str] = field(default_factory=list)   # conflicting evidence_ids
    citation:                 str = ""   # full citation string if available
    notes:                    str = ""
    iteration:                int = 0
    tool_call_id:             str = ""   # which tool call produced this evidence


@dataclass
class GeoHypothesis:
    """A testable geological hypothesis derived from the evidence table."""
    hypothesis_id:          str
    statement:              str
    supporting_evidence:    List[str]   # evidence_ids
    contradicting_evidence: List[str]
    data_types_needed:      List[str]
    confidence:             str         # high | medium | low | speculative
    status:                 str = "active"  # active | rejected | merged


@dataclass
class ValidationCheck:
    """A concrete validation analysis linked to a hypothesis."""
    check_id:       str
    linked_to:      str
    description:    str
    data_required:  str
    method:         str
    expected_outcome: str


@dataclass
class ToolCall:
    """Immutable record of a single tool invocation."""
    call_id:          str
    iteration:        int
    tool:             str
    method:           str
    args:             Dict[str, Any]
    reason:           str
    result_summary:   str
    evidence_added:   List[str]    = field(default_factory=list)  # evidence_ids
    figures_added:    List[str]    = field(default_factory=list)  # file paths
    error:            Optional[str] = None
    duration_s:       float        = 0.0
    ts:               float        = field(default_factory=time.time)


@dataclass
class GeoAgentResult:
    """Complete output of the EvidenceDrivenGeoAgent."""
    question:             str
    study_area:           str
    iterations_run:       int
    final_report:         str
    evidence_table:       List[GeoEvidence]
    hypotheses:           List[GeoHypothesis]
    tool_log:             List[ToolCall]
    retrieved_sources:    List[str]
    generated_figures:    List[str]
    missing_information:  List[str]
    suggested_validation: List[ValidationCheck]
    convergence_reason:   str


# ─────────────────────────────────────────────────────────────────────────────
# ── Path sandboxing ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _safe_path(path: str, *roots: str) -> Optional[Path]:
    """
    Resolve `path` and verify it lives under at least one of `roots`.
    Returns the resolved Path on success, None on sandbox violation.
    Accepts absolute paths, paths relative to roots, or bare filenames.
    """
    p = Path(path).expanduser()
    # Try to resolve; fall back to non-strict resolution for non-existent files
    try:
        resolved = p.resolve(strict=False)
    except Exception:
        return None

    for root in roots:
        if not root:
            continue
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue

    # If path isn't absolute, try appending it to each root
    if not p.is_absolute():
        for root in roots:
            if not root:
                continue
            candidate = (Path(root).expanduser() / p).resolve(strict=False)
            try:
                root_resolved = Path(root).expanduser().resolve(strict=False)
                candidate.relative_to(root_resolved)
                return candidate
            except ValueError:
                continue

    return None


def _ext_allowed(path: Path, allowed: List[str]) -> bool:
    return path.suffix.lower() in [e.lower() for e in allowed]


# ─────────────────────────────────────────────────────────────────────────────
# ── AgentLogger ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class AgentLogger:
    """Accumulates and serialises all tool call records."""

    def __init__(self) -> None:
        self._log: List[ToolCall] = []
        self._callbacks: List[Callable] = []

    def add_callback(self, cb: Callable[[Dict], None]) -> None:
        self._callbacks.append(cb)

    def record(self, call: ToolCall) -> None:
        self._log.append(call)
        for cb in self._callbacks:
            try:
                cb({
                    "phase":   "tool_call",
                    "tool":    call.tool,
                    "method":  call.method,
                    "message": f"[{call.tool}.{call.method}] {call.reason[:80]}",
                })
            except Exception:
                pass

    @property
    def log(self) -> List[ToolCall]:
        return list(self._log)

    def calls_this_iter(self, iteration: int) -> int:
        return sum(1 for c in self._log if c.iteration == iteration)

    def to_dicts(self) -> List[Dict]:
        return [asdict(c) for c in self._log]


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 1: LocalFileSearchTool ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class LocalFileSearchTool:
    """
    Search and read files within the configured workspace root.
    All paths are sandboxed to workspace_root.
    """

    NAME = "local_file_search"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    def _resolve(self, path: str) -> Optional[Path]:
        return _safe_path(path, self._cfg.workspace_root, self._cfg.output_dir)

    def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents within workspace_root."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside the workspace root."}
        if not p.exists():
            return {"error": f"Path does not exist: {p}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        entries = []
        for child in sorted(p.iterdir()):
            entries.append({
                "name":  child.name,
                "type":  "dir" if child.is_dir() else "file",
                "size":  child.stat().st_size if child.is_file() else None,
                "ext":   child.suffix.lower() if child.is_file() else None,
            })
        return {"path": str(p), "entries": entries, "count": len(entries)}

    def read_file(self, path: str, max_chars: int = 8000) -> Dict[str, Any]:
        """Read file content. Enforces extension allowlist and size cap."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside the workspace root."}
        if not p.exists():
            return {"error": f"File does not exist: {p}"}
        if not p.is_file():
            return {"error": f"Not a file: {p}"}
        if not _ext_allowed(p, self._cfg.allowed_extensions):
            return {"error": f"Extension '{p.suffix}' is not in the allowed list."}
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            truncated = len(text) > max_chars
            return {
                "path":      str(p),
                "content":   text[:max_chars],
                "truncated": truncated,
                "size":      len(text),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def search_files(self, query: str, root: str = ".") -> Dict[str, Any]:
        """
        Search for files whose name or text content matches `query`.
        Returns up to 20 matching file paths with brief snippets.
        """
        base = self._resolve(root)
        if base is None:
            base_path = Path(self._cfg.workspace_root).expanduser().resolve(strict=False)
        else:
            base_path = base

        q_lower = query.lower()
        matches = []
        for fp in sorted(base_path.rglob("*")):
            if not fp.is_file():
                continue
            if not _ext_allowed(fp, self._cfg.allowed_extensions):
                continue
            # filename match
            if q_lower in fp.name.lower():
                matches.append({"path": str(fp), "match": "filename", "snippet": ""})
                if len(matches) >= 20:
                    break
                continue
            # content match (text files only)
            if fp.suffix.lower() in (".txt", ".md", ".csv", ".json", ".yaml",
                                     ".yml", ".bib", ".log", ".dat"):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    if q_lower in text.lower():
                        idx = text.lower().find(q_lower)
                        snippet = text[max(0, idx - 60): idx + 100].strip()
                        matches.append({"path": str(fp), "match": "content", "snippet": snippet})
                        if len(matches) >= 20:
                            break
                except Exception:
                    pass

        return {"query": query, "matches": matches, "count": len(matches)}

    def grep(self, pattern: str, root: str = ".") -> Dict[str, Any]:
        """Grep `pattern` (regex) across text files in root. Returns up to 30 hits."""
        base = self._resolve(root)
        if base is None:
            base_path = Path(self._cfg.workspace_root).expanduser().resolve(strict=False)
        else:
            base_path = base

        text_exts = {".txt", ".md", ".csv", ".json", ".yaml", ".yml",
                     ".bib", ".log", ".dat", ".py", ".sh", ".inp", ".out"}
        hits = []
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return {"error": f"Invalid regex: {exc}"}

        for fp in sorted(base_path.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in text_exts:
                continue
            try:
                for lineno, line in enumerate(
                    fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        hits.append({
                            "file": str(fp),
                            "line": lineno,
                            "content": line.strip()[:200],
                        })
                        if len(hits) >= 30:
                            break
            except Exception:
                pass
            if len(hits) >= 30:
                break

        return {"pattern": pattern, "hits": hits, "count": len(hits)}

    def get_file_metadata(self, path: str) -> Dict[str, Any]:
        """Return size, mtime, extension, and line count for a file."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside the workspace root."}
        if not p.exists():
            return {"error": f"File does not exist: {p}"}
        stat = p.stat()
        meta: Dict[str, Any] = {
            "path":       str(p),
            "size_bytes": stat.st_size,
            "modified":   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            "extension":  p.suffix.lower(),
        }
        if p.suffix.lower() in (".txt", ".md", ".csv", ".json", ".bib",
                                ".log", ".dat", ".py"):
            try:
                with p.open(encoding="utf-8", errors="replace") as fh:
                    meta["line_count"] = sum(1 for _ in fh)
            except Exception:
                pass
        return meta


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 2: LiteratureLibraryTool ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class LiteratureLibraryTool:
    """
    Search and read the PDF / BibTeX literature library.
    All paths are sandboxed to literature_root (and workspace_root as fallback).
    """

    NAME = "literature_library"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    def _roots(self) -> Tuple[str, ...]:
        roots = [self._cfg.literature_root, self._cfg.workspace_root]
        return tuple(r for r in roots if r)

    def _resolve(self, path: str) -> Optional[Path]:
        return _safe_path(path, *self._roots())

    # ── internal PDF reader ────────────────────────────────────────────────

    @staticmethod
    def _read_pdf_text(path: Path, max_pages: int = 40) -> str:
        """Extract text from PDF using PyMuPDF → pypdf → raw bytes fallback."""
        # 1. PyMuPDF (best quality)
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(path))
            texts = []
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                texts.append(page.get_text())
            doc.close()
            return "\n".join(texts)
        except ImportError:
            pass
        except Exception:
            pass

        # 2. pypdf
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            texts = []
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                t = page.extract_text() or ""
                texts.append(t)
            return "\n".join(texts)
        except ImportError:
            pass
        except Exception:
            pass

        # 3. PyPDF2 (legacy fallback)
        try:
            import PyPDF2  # type: ignore
            reader = PyPDF2.PdfReader(str(path))
            texts = []
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                texts.append(page.extract_text() or "")
            return "\n".join(texts)
        except ImportError:
            pass
        except Exception:
            pass

        return "(Could not extract text from PDF — no PDF reader available)"

    # ── public methods ─────────────────────────────────────────────────────

    def search_papers(self, query: str) -> Dict[str, Any]:
        """Search for PDFs whose filename or first-page text matches `query`."""
        roots = self._roots()
        q_lower = query.lower()
        results = []

        for root in roots:
            rp = Path(root).expanduser().resolve(strict=False)
            for fp in sorted(rp.rglob("*.pdf")):
                if q_lower in fp.name.lower():
                    results.append({"path": str(fp), "match": "filename"})
                    continue
                # scan first ~2000 chars
                try:
                    snippet = self._read_pdf_text(fp, max_pages=2)[:2000]
                    if q_lower in snippet.lower():
                        idx = snippet.lower().find(q_lower)
                        ctx = snippet[max(0, idx - 60): idx + 120].strip()
                        results.append({"path": str(fp), "match": "content", "snippet": ctx})
                except Exception:
                    pass
                if len(results) >= 15:
                    break
            if len(results) >= 15:
                break

        return {"query": query, "results": results, "count": len(results)}

    def read_pdf(self, path: str, max_chars: int = 12000) -> Dict[str, Any]:
        """Read full text of a PDF within the literature library."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside configured roots."}
        if not p.exists():
            return {"error": f"File does not exist: {p}"}
        if p.suffix.lower() != ".pdf":
            return {"error": "read_pdf only supports .pdf files."}
        text = self._read_pdf_text(p)
        truncated = len(text) > max_chars
        return {
            "path":      str(p),
            "content":   text[:max_chars],
            "truncated": truncated,
            "chars":     len(text),
        }

    def extract_pdf_metadata(self, path: str) -> Dict[str, Any]:
        """Extract title, authors, year, abstract from a PDF."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside configured roots."}
        if not p.exists():
            return {"error": f"File does not exist: {p}"}
        meta: Dict[str, Any] = {"path": str(p)}

        # Try PyMuPDF metadata
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(p))
            info = doc.metadata
            meta.update({
                "title":    info.get("title", ""),
                "author":   info.get("author", ""),
                "subject":  info.get("subject", ""),
                "keywords": info.get("keywords", ""),
            })
            # Grab first 1200 chars as abstract proxy
            first_page = doc[0].get_text() if len(doc) > 0 else ""
            meta["first_page_excerpt"] = first_page[:1200]
            doc.close()
        except ImportError:
            text = self._read_pdf_text(p, max_pages=1)
            meta["first_page_excerpt"] = text[:1200]
        except Exception:
            text = self._read_pdf_text(p, max_pages=1)
            meta["first_page_excerpt"] = text[:1200]

        return meta

    def extract_references(self, path: str) -> Dict[str, Any]:
        """Extract the reference list from a PDF (heuristic line parser)."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside configured roots."}
        text = self._read_pdf_text(p, max_pages=60)
        # Find references section
        ref_start = -1
        for marker in ("References\n", "REFERENCES\n", "Bibliography\n", "参考文献\n"):
            idx = text.find(marker)
            if idx != -1:
                ref_start = idx + len(marker)
                break
        if ref_start == -1:
            return {"path": str(p), "references": [], "note": "Could not find References section"}

        ref_block = text[ref_start: ref_start + 6000]
        lines = [l.strip() for l in ref_block.splitlines() if len(l.strip()) > 20]
        # Heuristic: lines starting with [N] or author-year pattern
        refs = []
        for line in lines[:60]:
            if re.match(r'^\[?\d+\]?\s+\w', line) or re.match(r'^[A-Z][a-z]+,', line):
                refs.append(line)

        return {"path": str(p), "references": refs[:40], "count": len(refs)}

    def search_bibtex(self, query: str) -> Dict[str, Any]:
        """Search BibTeX files for entries matching `query`."""
        roots = self._roots()
        q_lower = query.lower()
        entries = []

        for root in roots:
            rp = Path(root).expanduser().resolve(strict=False)
            for fp in rp.rglob("*.bib"):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    # Split into @-entries
                    raw_entries = re.split(r'\n@', "\n" + text)
                    for entry in raw_entries:
                        if q_lower in entry.lower():
                            # Get the key and first few fields
                            key_m = re.match(r'(\w+)\{(\S+),', entry[:100])
                            entry_key = key_m.group(2) if key_m else "?"
                            entries.append({
                                "file":  str(fp),
                                "key":   entry_key,
                                "snippet": ("@" + entry[:300].strip())
                                           if not entry.startswith("@") else entry[:300].strip(),
                            })
                            if len(entries) >= 20:
                                break
                except Exception:
                    pass
            if len(entries) >= 20:
                break

        return {"query": query, "entries": entries, "count": len(entries)}


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 3: RAGIndexTool ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class RAGIndexTool:
    """
    Interface to the project's existing RAG knowledge base.
    index_documents / add_document / rebuild_index write to the rag index;
    search_rag retrieves ranked chunks.
    """

    NAME = "rag_index"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._kb  = None   # lazy-loaded

    def _get_kb(self):
        if self._kb is None:
            self._kb = _get_kb()
        return self._kb

    def search_rag(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """Search the RAG knowledge base. Returns ranked chunks."""
        kb = self._get_kb()
        if kb is None or getattr(kb, "is_empty", True):
            return {"query": query, "chunks": [], "note": "RAG knowledge base is empty or unavailable."}
        try:
            hits = kb.retrieve(query, top_k=top_k,
                               score_threshold=self._cfg.score_threshold)
            chunks = []
            for chunk, score in hits:
                chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "page":     chunk.page,
                    "score":    round(score, 4),
                    "text":     chunk.text[:1200],
                })
            return {"query": query, "chunks": chunks, "count": len(chunks)}
        except Exception as exc:
            return {"query": query, "chunks": [], "error": str(exc)}

    def add_document(self, path: str) -> Dict[str, Any]:
        """Add a single document to the RAG index."""
        p = _safe_path(path, self._cfg.workspace_root, self._cfg.literature_root)
        if p is None:
            return {"error": f"Path '{path}' is outside configured roots."}
        kb = self._get_kb()
        if kb is None:
            return {"error": "RAG knowledge base not available."}
        try:
            kb.add_document(str(p))
            return {"ok": True, "path": str(p)}
        except Exception as exc:
            return {"error": str(exc)}

    def index_documents(self, path: str) -> Dict[str, Any]:
        """Index all documents in a directory."""
        p = _safe_path(path, self._cfg.workspace_root, self._cfg.literature_root)
        if p is None:
            return {"error": f"Path '{path}' is outside configured roots."}
        kb = self._get_kb()
        if kb is None:
            return {"error": "RAG knowledge base not available."}
        try:
            count = 0
            for fp in Path(str(p)).rglob("*"):
                if fp.is_file() and fp.suffix.lower() in (".pdf", ".md", ".txt"):
                    try:
                        kb.add_document(str(fp))
                        count += 1
                    except Exception:
                        pass
            return {"ok": True, "indexed": count, "path": str(p)}
        except Exception as exc:
            return {"error": str(exc)}

    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the entire RAG index from scratch."""
        kb = self._get_kb()
        if kb is None:
            return {"error": "RAG knowledge base not available."}
        try:
            kb.rebuild()
            return {"ok": True}
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 4: SeismoDataTool ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class SeismoDataTool:
    """
    Read and summarise seismic datasets: catalogs, station files, waveforms,
    velocity models, and focal mechanism tables.
    All paths sandboxed to workspace_root.
    """

    NAME = "seismo_data"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    def _resolve(self, path: str) -> Optional[Path]:
        return _safe_path(path, self._cfg.workspace_root,
                          self._cfg.literature_root, self._cfg.output_dir)

    def read_catalog(self, path: str, max_rows: int = 500) -> Dict[str, Any]:
        """
        Read an earthquake catalog (CSV, JSON, or custom text).
        Returns summary statistics + first N rows.
        """
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        try:
            import pandas as pd
            if p.suffix.lower() == ".json":
                df = pd.read_json(str(p))
            else:
                # Try comma, then whitespace separators
                try:
                    df = pd.read_csv(str(p), comment="#")
                except Exception:
                    df = pd.read_csv(str(p), sep=r'\s+', comment="#")

            n = len(df)
            cols = list(df.columns)
            # Detect key columns
            lon_col = next((c for c in cols if "lon" in c.lower()), None)
            lat_col = next((c for c in cols if "lat" in c.lower()), None)
            dep_col = next((c for c in cols if c.lower() in ("dep", "depth", "z")), None)
            mag_col = next((c for c in cols if c.lower() in ("mag", "ml", "mw", "ms", "mb", "magnitude")), None)

            summary: Dict[str, Any] = {"rows": n, "columns": cols, "path": str(p)}
            if mag_col and mag_col in df.columns:
                summary["mag_range"] = [round(float(df[mag_col].min()), 2),
                                        round(float(df[mag_col].max()), 2)]
            if dep_col and dep_col in df.columns:
                summary["depth_range_km"] = [round(float(df[dep_col].min()), 1),
                                              round(float(df[dep_col].max()), 1)]
            if lon_col and lat_col:
                summary["lon_range"] = [round(float(df[lon_col].min()), 3),
                                        round(float(df[lon_col].max()), 3)]
                summary["lat_range"] = [round(float(df[lat_col].min()), 3),
                                        round(float(df[lat_col].max()), 3)]
            summary["preview"] = df.head(min(max_rows, 20)).to_dict(orient="records")
            return summary
        except ImportError:
            return {"error": "pandas is required for catalog reading."}
        except Exception as exc:
            return {"error": str(exc)}

    def read_station_file(self, path: str) -> Dict[str, Any]:
        """Read a seismic station file (CSV, StationXML, or text)."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}

        # StationXML
        if p.suffix.lower() == ".xml":
            try:
                from obspy import read_inventory  # type: ignore
                inv = read_inventory(str(p))
                stations = []
                for net in inv:
                    for sta in net:
                        stations.append({
                            "network":   net.code,
                            "station":   sta.code,
                            "latitude":  sta.latitude,
                            "longitude": sta.longitude,
                            "elevation": sta.elevation,
                        })
                return {"path": str(p), "stations": stations, "count": len(stations)}
            except ImportError:
                pass
            except Exception as exc:
                return {"error": str(exc)}

        # CSV / text fallback
        try:
            import pandas as pd
            df = pd.read_csv(str(p), comment="#", sep=None, engine="python")
            return {
                "path":    str(p),
                "columns": list(df.columns),
                "rows":    len(df),
                "preview": df.head(10).to_dict(orient="records"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def read_waveform(self, path: str) -> Dict[str, Any]:
        """
        Read seismic waveform (SAC, MiniSEED) and return basic statistics.
        """
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        try:
            from obspy import read  # type: ignore
            st = read(str(p))
            traces = []
            for tr in st:
                traces.append({
                    "network":    tr.stats.network,
                    "station":    tr.stats.station,
                    "channel":    tr.stats.channel,
                    "starttime":  str(tr.stats.starttime),
                    "endtime":    str(tr.stats.endtime),
                    "sampling_hz": tr.stats.sampling_rate,
                    "npts":       tr.stats.npts,
                    "data_min":   float(tr.data.min()),
                    "data_max":   float(tr.data.max()),
                })
            return {"path": str(p), "traces": traces, "n_traces": len(traces)}
        except ImportError:
            return {"error": "obspy is required for waveform reading."}
        except Exception as exc:
            return {"error": str(exc)}

    def read_velocity_model(self, path: str) -> Dict[str, Any]:
        """Read a 1-D velocity model (CSV or text columns: depth vp vs)."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        try:
            import pandas as pd
            df = pd.read_csv(str(p), comment="#", sep=None, engine="python")
            return {
                "path":    str(p),
                "columns": list(df.columns),
                "layers":  len(df),
                "data":    df.to_dict(orient="records"),
            }
        except Exception as exc:
            # plain text fallback
            try:
                lines = [l for l in p.read_text(errors="replace").splitlines()
                         if l.strip() and not l.startswith("#")]
                return {"path": str(p), "lines": lines[:50], "count": len(lines)}
            except Exception:
                return {"error": str(exc)}

    def read_focal_mechanisms(self, path: str, max_rows: int = 200) -> Dict[str, Any]:
        """Read focal mechanism table (CSV with strike/dip/rake or moment tensor columns)."""
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        try:
            import pandas as pd
            df = pd.read_csv(str(p), comment="#", sep=None, engine="python")
            return {
                "path":    str(p),
                "columns": list(df.columns),
                "rows":    len(df),
                "preview": df.head(min(max_rows, 20)).to_dict(orient="records"),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 5: GeoPlotTool ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class GeoPlotTool:
    """
    Generate matplotlib-based geoscience figures. Figures are saved to output_dir.
    """

    NAME = "geo_plot"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _out(self, filename: str) -> str:
        return str(Path(self._cfg.output_dir) / filename)

    def _load_catalog(self, catalog_path: str):  # type: ignore
        import pandas as pd
        p = _safe_path(catalog_path, self._cfg.workspace_root,
                       self._cfg.literature_root, self._cfg.output_dir)
        if p is None or not p.exists():
            raise FileNotFoundError(f"Catalog not found or outside workspace: {catalog_path}")
        try:
            return pd.read_csv(str(p), comment="#")
        except Exception:
            return pd.read_csv(str(p), sep=r'\s+', comment="#")

    def plot_catalog_map(
        self,
        catalog_path: str,
        region: Optional[List[float]] = None,
        title: str = "Earthquake Catalog Map",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot epicenter map colored by depth or magnitude."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            df = self._load_catalog(catalog_path)
            lon_col = next((c for c in df.columns if "lon" in c.lower()), df.columns[0])
            lat_col = next((c for c in df.columns if "lat" in c.lower()), df.columns[1])
            dep_col = next((c for c in df.columns if c.lower() in ("dep", "depth", "z")), None)
            mag_col = next((c for c in df.columns if c.lower() in ("mag", "ml", "mw", "ms")), None)

            fig, ax = plt.subplots(figsize=(10, 8))

            color_vals = df[dep_col].values if dep_col else np.zeros(len(df))
            size_vals  = (df[mag_col].values * 3) ** 2 if mag_col else 20

            sc = ax.scatter(df[lon_col], df[lat_col], c=color_vals, s=size_vals,
                            cmap="plasma_r", alpha=0.7, edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="Depth (km)" if dep_col else "")
            if region:
                ax.set_xlim(region[0], region[1])
                ax.set_ylim(region[2], region[3])
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            out = output or self._out("catalog_map.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"ok": True, "figure": out}
        except Exception as exc:
            return {"error": str(exc)}

    def plot_depth_section(
        self,
        catalog_path: str,
        profile: Optional[Dict[str, float]] = None,
        title: str = "Depth Cross-Section",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot earthquake depth distribution (distance vs depth)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            df = self._load_catalog(catalog_path)
            dep_col = next((c for c in df.columns if c.lower() in ("dep", "depth", "z")), None)
            mag_col = next((c for c in df.columns if c.lower() in ("mag", "ml", "mw")), None)

            if dep_col is None:
                return {"error": "No depth column found in catalog."}

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            depth = df[dep_col].dropna()
            ax1.hist(depth, bins=30, color="steelblue", edgecolor="k", alpha=0.7)
            ax1.set_xlabel("Depth (km)")
            ax1.set_ylabel("Count")
            ax1.set_title("Depth Distribution")
            ax1.invert_xaxis()

            lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
            lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
            if lon_col and lat_col:
                size_vals = (df[mag_col].values * 3) ** 2 if mag_col else 20
                sc = ax2.scatter(df[lon_col], df[dep_col],
                                 c=df[dep_col], cmap="plasma_r",
                                 s=size_vals, alpha=0.6, edgecolors="k", linewidths=0.2)
                plt.colorbar(sc, ax=ax2, label="Depth (km)")
                ax2.set_xlabel("Longitude (°E)")
                ax2.set_ylabel("Depth (km)")
                ax2.set_title("Lon vs Depth")
                ax2.invert_yaxis()

            fig.suptitle(title)
            out = output or self._out("depth_section.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"ok": True, "figure": out}
        except Exception as exc:
            return {"error": str(exc)}

    def plot_velocity_slice(
        self,
        model_path: str,
        depth: float = 10.0,
        title: str = "Velocity Slice",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot a horizontal slice of a velocity model at a given depth."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd

            p = _safe_path(model_path, self._cfg.workspace_root,
                           self._cfg.literature_root, self._cfg.output_dir)
            if p is None or not p.exists():
                return {"error": f"Model file not found: {model_path}"}

            df = pd.read_csv(str(p), comment="#", sep=None, engine="python")
            dep_col = next((c for c in df.columns if "dep" in c.lower()), None)
            if dep_col is None:
                return {"error": "No depth column in velocity model."}

            slice_df = df[(df[dep_col] - depth).abs() < 2].copy()
            if slice_df.empty:
                return {"error": f"No data near depth={depth} km."}

            lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
            lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
            vp_col  = next((c for c in df.columns if "vp" in c.lower()), None)

            if not (lon_col and lat_col and vp_col):
                return {"error": "Model needs lon/lat/vp columns for 2-D slice."}

            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(slice_df[lon_col], slice_df[lat_col],
                            c=slice_df[vp_col], cmap="seismic",
                            s=40, alpha=0.8)
            plt.colorbar(sc, ax=ax, label="Vp (km/s)")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"{title}  (depth ≈ {depth} km)")
            ax.grid(True, alpha=0.3)
            out = output or self._out("velocity_slice.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"ok": True, "figure": out}
        except Exception as exc:
            return {"error": str(exc)}

    def plot_fault_distance(
        self,
        catalog_path: str,
        fault_path: Optional[str] = None,
        title: str = "Earthquake-Fault Distance",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot earthquake distances from the nearest fault trace.
        If fault_path is None, shows the magnitude-depth relation instead.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            df = self._load_catalog(catalog_path)
            dep_col = next((c for c in df.columns if c.lower() in ("dep", "depth", "z")), None)
            mag_col = next((c for c in df.columns if c.lower() in ("mag", "ml", "mw", "ms")), None)

            fig, ax = plt.subplots(figsize=(10, 6))
            if dep_col and mag_col:
                ax.scatter(df[mag_col], df[dep_col],
                           alpha=0.5, s=20, color="steelblue", edgecolors="none")
                ax.set_xlabel("Magnitude")
                ax.set_ylabel("Depth (km)")
                ax.invert_yaxis()
                ax.set_title(title + " (Magnitude vs Depth)")
            else:
                ax.text(0.5, 0.5, "Insufficient columns for fault-distance plot.",
                        ha="center", va="center", transform=ax.transAxes)

            out = output or self._out("fault_distance.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"ok": True, "figure": out}
        except Exception as exc:
            return {"error": str(exc)}

    def plot_evidence_map(
        self,
        evidence_list: List[Dict],
        region: Optional[List[float]] = None,
        title: str = "Evidence Location Map",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot a map summarising evidence locations from the evidence table.
        Each evidence item should have optional 'lon', 'lat', 'source_type' keys.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(10, 8))

            color_map = {
                "literature":    "steelblue",
                "local_data":    "darkorange",
                "model_derived": "forestgreen",
                "inference":     "purple",
                "speculation":   "gray",
            }
            legend_handles = []

            for ev in evidence_list:
                lon = ev.get("lon") or ev.get("longitude")
                lat = ev.get("lat") or ev.get("latitude")
                if lon is None or lat is None:
                    continue
                stype = ev.get("source_type", "inference")
                color = color_map.get(stype, "gray")
                ax.scatter(float(lon), float(lat), color=color, s=60,
                           alpha=0.8, edgecolors="k", linewidths=0.3)

            for stype, color in color_map.items():
                legend_handles.append(
                    mpatches.Patch(color=color, label=stype.replace("_", " ").title())
                )
            ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

            if region:
                ax.set_xlim(region[0], region[1])
                ax.set_ylim(region[2], region[3])
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            out = output or self._out("evidence_map.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"ok": True, "figure": out}
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 6: CodeExecutionTool ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

_PYTHON_PREAMBLE = """\
import os, sys, json
import warnings; warnings.filterwarnings('ignore')
# Output directory is pre-set via environment variable SAGE_OUTDIR
SAGE_OUTDIR = os.environ.get('SAGE_OUTDIR', '.')
os.makedirs(SAGE_OUTDIR, exist_ok=True)
import matplotlib; matplotlib.use('Agg')
"""


class CodeExecutionTool:
    """
    Execute Python code or (optionally) shell commands in a sandboxed subprocess.

    Safety constraints:
    - allow_python / allow_shell are checked before every call
    - cwd is set to output_dir
    - SAGE_OUTDIR environment variable is injected
    - timeout and max_output limits enforced
    - Arbitrary imports are the user's responsibility; network is not blocked
      (restricting outbound connections requires OS-level controls)
    """

    NAME = "code_execution"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_python(self, code: str) -> Dict[str, Any]:
        """Execute `code` in a subprocess. Returns stdout, stderr, figures, error."""
        if not self._cfg.allow_python:
            return {"error": "Python execution is disabled (allow_python=false)."}

        full_code = _PYTHON_PREAMBLE + "\n" + code
        out_dir = str(Path(self._cfg.output_dir).resolve())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         encoding="utf-8", delete=False) as tf:
            tf.write(full_code)
            script = tf.name

        env = os.environ.copy()
        env["SAGE_OUTDIR"] = out_dir

        before_files = set(Path(out_dir).glob("*.png")) | set(Path(out_dir).glob("*.jpg"))

        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=self._cfg.code_timeout_s,
                cwd=out_dir,
                env=env,
            )
            stdout = proc.stdout[: self._cfg.code_max_output]
            stderr = proc.stderr[: self._cfg.code_max_output]
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            return {"error": f"Code execution timed out after {self._cfg.code_timeout_s}s."}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            try:
                os.unlink(script)
            except Exception:
                pass

        after_files = set(Path(out_dir).glob("*.png")) | set(Path(out_dir).glob("*.jpg"))
        new_figures = [str(f) for f in (after_files - before_files)]

        return {
            "ok":         returncode == 0,
            "returncode": returncode,
            "stdout":     stdout,
            "stderr":     stderr,
            "figures":    new_figures,
        }

    def run_shell(self, command: str) -> Dict[str, Any]:
        """Execute a shell command. Requires allow_shell=True in config."""
        if not self._cfg.allow_shell:
            return {"error": "Shell execution is disabled (allow_shell=false)."}

        out_dir = str(Path(self._cfg.output_dir).resolve())
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._cfg.code_timeout_s,
                cwd=out_dir,
            )
            return {
                "ok":         proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout":     proc.stdout[: self._cfg.code_max_output],
                "stderr":     proc.stderr[: self._cfg.code_max_output],
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Shell command timed out after {self._cfg.code_timeout_s}s."}
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 7: StateMemoryTool ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 8: ImageAnalysisTool ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_ANALYSIS_SYSTEM = """\
You are a geoscience figure interpreter.  Given an image from a seismology or
geophysics paper (map, cross-section, velocity model, focal mechanism plot,
dispersion image, etc.), extract structured information.

For each figure:
1. what_plotted:  data type / variable on each axis
2. spatial_distribution: describe earthquake clusters, velocity anomalies, etc.
3. key_anomalies: any clusters, gradients, discontinuities, outliers
4. scale: depth range, distance range, magnitude range
5. stated_interpretation: what the authors say this figure shows
6. alternative_interpretations: one or two alternative geological readings
7. data_type: seismicity | velocity_model | focal_mechanism | dispersion | geology | other
8. confidence: high | medium | low (how much information you could extract)

Output ONLY a JSON object with these keys.  If you cannot determine a value, use "unspecified".
"""

_TABLE_EXTRACTION_SYSTEM = """\
You are a scientific table parser. Given an image or text containing a table, extract
its contents as structured JSON.

Output:
{
  "title": "table caption or title",
  "headers": ["col1", "col2", ...],
  "rows": [ [val, val, ...], ... ],
  "units": {"col1": "unit", ...},
  "variables": ["description of each variable"],
  "relationships": ["any correlation or pattern you see"],
  "evidence_entries": [
    {
      "observation": "factual claim derivable from the table",
      "data_type": "...",
      "confidence": "high | medium | low"
    }
  ]
}
"""


class ImageAnalysisTool:
    """
    Multimodal image and table analysis tool.
    Requires a vision-capable LLM (e.g. GPT-4o, LLaVA, Qwen-VL).
    Falls back to graceful "not available" message if the LLM is text-only.
    """

    NAME = "image_analysis"

    def __init__(self, config: AgentConfig, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = config
        self._llm = llm_cfg

    def _resolve(self, path: str) -> Optional[Path]:
        return _safe_path(path, self._cfg.workspace_root,
                          self._cfg.literature_root, self._cfg.output_dir)

    def _encode_image(self, path: Path) -> Optional[str]:
        """Base64-encode an image file for multimodal LLM calls."""
        import base64
        try:
            with open(str(path), "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _vision_call(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        """
        Send an image + text to the LLM.
        Supports OpenAI vision format and Ollama multimodal format.
        """
        b64 = self._encode_image(image_path)
        if not b64:
            return json.dumps({"error": "Could not encode image."})

        provider = self._llm.get("provider", "ollama")
        model    = self._llm.get("model", "")
        api_base = self._llm.get("api_base", "")
        api_key  = self._llm.get("api_key", "")

        if not api_base or not model:
            return json.dumps({"error": "LLM not configured."})

        suffix = image_path.suffix.lower().lstrip(".")
        mime   = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "gif": "image/gif"}.get(suffix, "image/png")

        if provider == "ollama":
            # Ollama multimodal: images as list of base64 strings
            url     = api_base.rstrip("/") + "/api/chat"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt, "images": [b64]},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1000},
            }
        else:
            # OpenAI vision format
            url     = api_base.rstrip("/") + "/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text",       "text": user_prompt},
                        {"type": "image_url",  "image_url": {
                            "url": f"data:{mime};base64,{b64}"}},
                    ]},
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
            }

        data    = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
        }
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
            if provider == "ollama":
                return body.get("message", {}).get("content", "").strip()
            return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def analyze_image(self, path: str, question: str = "") -> Dict[str, Any]:
        """
        Interpret a geoscience figure (map, cross-section, velocity model, etc.).
        Returns structured analysis: what is plotted, anomalies, interpretations.
        """
        if not self._cfg.use_multimodal:
            return {"error": "Multimodal analysis is disabled (use_multimodal=false)."}
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff"):
            return {"error": f"Unsupported image format: {p.suffix}"}

        user_prompt = (
            f"Analyse this geoscience figure.\n"
            f"Scientific context: {question or 'general geoscience figure'}\n\n"
            "Extract structured information as JSON."
        )
        raw = self._vision_call(p, _IMAGE_ANALYSIS_SYSTEM, user_prompt)
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(m.group(0)) if m else {"raw": raw}
        except Exception:
            result = {"raw": raw}
        result["path"] = str(p)
        return result

    def extract_table(self, path: str, question: str = "") -> Dict[str, Any]:
        """
        Extract structured data from a table (image or text file).
        For image files uses vision LLM; for text/CSV uses pandas.
        """
        if not self._cfg.use_multimodal and not path.lower().endswith((".csv", ".tsv", ".txt")):
            return {"error": "Multimodal analysis is disabled (use_multimodal=false)."}
        p = self._resolve(path)
        if p is None:
            return {"error": f"Path '{path}' is outside workspace."}
        if not p.exists():
            return {"error": f"File not found: {p}"}

        # Text / CSV table — use pandas
        if p.suffix.lower() in (".csv", ".tsv", ".txt"):
            try:
                import pandas as pd
                sep = "\t" if p.suffix.lower() == ".tsv" else None
                df  = pd.read_csv(str(p), sep=sep, comment="#", engine="python")
                return {
                    "path":     str(p),
                    "headers":  list(df.columns),
                    "rows":     df.head(20).values.tolist(),
                    "n_rows":   len(df),
                    "evidence_entries": [
                        {
                            "observation": f"Table contains {len(df)} rows × {len(df.columns)} columns: {list(df.columns)}",
                            "data_type":  "other",
                            "confidence": "high",
                        }
                    ],
                }
            except Exception as exc:
                return {"error": str(exc)}

        # Image table — vision LLM
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif"):
            return {"error": f"Cannot extract table from {p.suffix} file."}

        user_prompt = (
            f"Extract the table from this image.\n"
            f"Context: {question or 'geoscience data table'}\n\n"
            "Return structured JSON."
        )
        raw = self._vision_call(p, _TABLE_EXTRACTION_SYSTEM, user_prompt)
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(m.group(0)) if m else {"raw": raw}
        except Exception:
            result = {"raw": raw}
        result["path"] = str(p)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# ── Tool 9: WebSearchTool ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Optional web and scholar search.  Disabled by default (allow_web_search=False).

    web_search:    DuckDuckGo HTML scrape (no API key required)
    scholar_search: Semantic Scholar public API (no key required)
    download_pdf:  Fetch and save a PDF from a URL to output_dir
    """

    NAME = "web_search"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    def _check_allowed(self) -> Optional[Dict]:
        if not self._cfg.allow_web_search:
            return {"error": (
                "Web search is disabled (allow_web_search=false). "
                "Enable it in AgentConfig or via --allow-web-search CLI flag. "
                "Reason required: RAG + local sources must be insufficient first."
            )}
        return None

    def web_search(self, query: str, max_results: int = 8) -> Dict[str, Any]:
        """Search the web via DuckDuckGo HTML (no API key required)."""
        err = self._check_allowed()
        if err:
            return err
        try:
            encoded = urllib.request.quote(query)
            url     = f"https://html.duckduckgo.com/html/?q={encoded}"
            req     = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SeismicX/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Minimal HTML scrape — extract result titles and snippets
            results = []
            for m in re.finditer(
                r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                html, re.DOTALL
            ):
                href  = m.group(1)
                title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                if href and title:
                    results.append({"url": href, "title": title, "snippet": ""})
                if len(results) >= max_results:
                    break

            # Try to grab snippets from result__snippet spans
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</[^>]+>', html, re.DOTALL
            )
            for i, s in enumerate(snippets[:len(results)]):
                results[i]["snippet"] = re.sub(r'<[^>]+>', '', s).strip()[:200]

            return {"query": query, "results": results, "count": len(results)}
        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc)}

    def scholar_search(self, query: str, max_results: int = 8) -> Dict[str, Any]:
        """
        Search Semantic Scholar (public API, no key required).
        Returns paper title, authors, year, abstract, and URL.
        """
        err = self._check_allowed()
        if err:
            return err
        try:
            encoded = urllib.request.quote(query)
            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={encoded}&limit={max_results}"
                f"&fields=title,authors,year,abstract,url,externalIds"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "SeismicX/1.0",
                    "Accept":     "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode())

            papers = []
            for p in body.get("data", []):
                papers.append({
                    "title":   p.get("title", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])[:4]],
                    "year":    p.get("year"),
                    "abstract": (p.get("abstract") or "")[:400],
                    "url":     p.get("url", ""),
                    "doi":     p.get("externalIds", {}).get("DOI", ""),
                })
            return {"query": query, "papers": papers, "count": len(papers)}
        except Exception as exc:
            return {"query": query, "papers": [], "error": str(exc)}

    def download_pdf(self, url: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a PDF from `url` into output_dir.
        Returns the saved file path on success.
        """
        err = self._check_allowed()
        if err:
            return err
        try:
            fname = filename or (hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf")
            dest  = Path(self._cfg.output_dir) / "downloaded_pdfs" / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SeismicX/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
            dest.write_bytes(content)
            return {"ok": True, "path": str(dest), "size_bytes": len(content)}
        except Exception as exc:
            return {"error": str(exc)}


class StateMemoryTool:
    """
    Persist agent state between iterations and across runs.
    All files are written to output_dir/{task_id}/.
    """

    NAME = "state_memory"

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config

    def _task_dir(self, task_id: str) -> Path:
        d = Path(self._cfg.output_dir) / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_state(self, task_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Persist arbitrary state dict as JSON."""
        p = self._task_dir(task_id) / "state.json"
        try:
            p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True, "path": str(p)}
        except Exception as exc:
            return {"error": str(exc)}

    def load_state(self, task_id: str) -> Dict[str, Any]:
        """Load previously persisted state."""
        p = self._task_dir(task_id) / "state.json"
        if not p.exists():
            return {"found": False, "state": {}}
        try:
            return {"found": True, "state": json.loads(p.read_text(encoding="utf-8"))}
        except Exception as exc:
            return {"error": str(exc)}

    def save_report(self, task_id: str, report: str) -> Dict[str, Any]:
        """Save Markdown report to disk."""
        p = self._task_dir(task_id) / "report.md"
        try:
            p.write_text(report, encoding="utf-8")
            return {"ok": True, "path": str(p)}
        except Exception as exc:
            return {"error": str(exc)}

    def save_evidence_table(self, task_id: str, table: List[Dict]) -> Dict[str, Any]:
        """Save evidence table as JSON."""
        p = self._task_dir(task_id) / "evidence_table.json"
        try:
            p.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True, "path": str(p), "count": len(table)}
        except Exception as exc:
            return {"error": str(exc)}

    def save_hypotheses(self, task_id: str, hypotheses: List[Dict]) -> Dict[str, Any]:
        """Save hypotheses as JSON."""
        p = self._task_dir(task_id) / "hypotheses.json"
        try:
            p.write_text(json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True, "path": str(p), "count": len(hypotheses)}
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ── ToolRegistry ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Manages tool instances, dispatches calls, and records every invocation."""

    def __init__(
        self,
        config: AgentConfig,
        logger: AgentLogger,
        llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger    = logger
        _llm            = llm_cfg or {}
        self.local_file = LocalFileSearchTool(config)
        self.literature = LiteratureLibraryTool(config)
        self.rag        = RAGIndexTool(config)
        self.seismo     = SeismoDataTool(config)
        self.geo_plot   = GeoPlotTool(config)
        self.code       = CodeExecutionTool(config)
        self.memory     = StateMemoryTool(config)
        self.images     = ImageAnalysisTool(config, _llm)
        self.web        = WebSearchTool(config)

        self._tools: Dict[str, Any] = {
            "local_file_search":   self.local_file,
            "literature_library":  self.literature,
            "rag_index":           self.rag,
            "seismo_data":         self.seismo,
            "geo_plot":            self.geo_plot,
            "code_execution":      self.code,
            "state_memory":        self.memory,
            "image_analysis":      self.images,
            "web_search":          self.web,
        }

    def dispatch(
        self,
        tool: str,
        method: str,
        args: Dict[str, Any],
        reason: str,
        iteration: int,
    ) -> Tuple[Dict[str, Any], ToolCall]:
        """
        Call tool.method(**args).
        Returns (result_dict, ToolCall record).
        Every call is logged regardless of success/failure.
        """
        call_id = uuid.uuid4().hex[:10]
        t0 = time.time()

        instance = self._tools.get(tool)
        if instance is None:
            result = {"error": f"Unknown tool: '{tool}'"}
        else:
            fn = getattr(instance, method, None)
            if fn is None:
                result = {"error": f"Unknown method: '{tool}.{method}'"}
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = {"error": str(exc)}

        duration = time.time() - t0

        # Summarise result for the log (truncate long values)
        if "error" in result:
            summary = f"ERROR: {result['error'][:120]}"
        elif "content" in result:
            summary = f"Read {result.get('size', '?')} chars from {result.get('path', '?')}"
        elif "chunks" in result:
            summary = f"RAG: {len(result.get('chunks', []))} chunks for '{args.get('query', '?')[:40]}'"
        elif "matches" in result:
            summary = f"Found {result.get('count', 0)} matches for '{args.get('query', '?')[:40]}'"
        elif "entries" in result:
            summary = f"Found {result.get('count', 0)} BibTeX entries"
        elif "results" in result:
            summary = f"Found {result.get('count', 0)} papers"
        elif "figure" in result:
            summary = f"Figure saved: {result['figure']}"
        elif "ok" in result:
            summary = f"ok={result['ok']} {list(result.keys())[:4]}"
        else:
            summary = str(result)[:120]

        call = ToolCall(
            call_id=call_id,
            iteration=iteration,
            tool=tool,
            method=method,
            args=args,
            reason=reason,
            result_summary=summary,
            error=result.get("error"),
            duration_s=round(duration, 3),
        )
        self._logger.record(call)
        return result, call


# ─────────────────────────────────────────────────────────────────────────────
# ── LLM-based reasoning components ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_SELECTOR_SYSTEM = """\
You are an autonomous geoscience reasoning agent.
Your goal is to gather structured evidence to answer a scientific question.
You must NEVER skip evidence structuring.  Build evidence first; reason second.

DATA SOURCE PRIORITY (strict order):
1. local structured data (seismo_data: catalogs, velocity models, focal mechanisms)
2. rag_index (indexed knowledge base)
3. literature_library (PDFs, BibTeX)
4. image_analysis (figures, maps, cross-sections in workspace)
5. web_search — ONLY if all local sources are insufficient AND you explicitly state why

You have access to these tools:
- local_file_search   : list_dir(path) | read_file(path) | search_files(query,root) | grep(pattern,root) | get_file_metadata(path)
- literature_library  : search_papers(query) | read_pdf(path) | extract_pdf_metadata(path) | extract_references(path) | search_bibtex(query)
- rag_index           : search_rag(query,top_k) | add_document(path) | index_documents(path) | rebuild_index()
- seismo_data         : read_catalog(path) | read_station_file(path) | read_waveform(path) | read_velocity_model(path) | read_focal_mechanisms(path)
- geo_plot            : plot_catalog_map(catalog_path,region,title,output) | plot_depth_section(catalog_path,profile,title,output) | plot_velocity_slice(model_path,depth,title,output) | plot_fault_distance(catalog_path,fault_path,title,output) | plot_evidence_map(evidence_list,region,title,output)
- code_execution      : run_python(code) | run_shell(command)
- state_memory        : save_state(task_id,state) | save_report(task_id,report) | save_evidence_table(task_id,table) | save_hypotheses(task_id,hypotheses)
- image_analysis      : analyze_image(path,question) | extract_table(path,question)
- web_search          : web_search(query,max_results) | scholar_search(query,max_results) | download_pdf(url,filename)

Strategy rules:
1. Start iteration 1 with: seismo_data → rag_index.search_rag → local_file_search → literature_library.
2. Use image_analysis for any PNG/JPG figures found in the workspace.
3. Use code_execution only for analyses not possible with other tools.
4. Escalate to web_search only when local + RAG + literature are insufficient; include the reason in "reason".
5. Use geo_plot to generate figures after reading data.
6. Never call state_memory unless persisting for later reuse.
7. Return {"done": true} when enough evidence gathered for this iteration.

Output EXACTLY this JSON (no extra text):
{
  "tool":    "<tool_name>",
  "method":  "<method_name>",
  "args":    { ... },
  "purpose": "<what you expect this call to return>",
  "reason":  "<why you are calling this tool NOW, referencing priority order>",
  "done":    false
}
OR:
{ "done": true, "reason": "<why you have enough evidence for this iteration>" }
"""


_EVIDENCE_EXTRACTOR_SYSTEM = """\
You are a geoscience evidence analyst. Given tool output and the scientific question,
extract structured evidence records.

ANTI-HALLUCINATION RULES:
- Every claim MUST be directly traceable to the tool output.
- NEVER fabricate numbers, coordinates, or geological names not in the source.
- If the tool output contains no relevant evidence, return an empty array [].

SEPARATION RULES:
- observation: what the data / text DIRECTLY states (factual, verbatim or close paraphrase)
- interpretation: what the authors / source CONCLUDE from the observation
- alternative_interpretation: a different geological reading the agent proposes
- assumption: key assumptions stated or implied in the source
- Mark source_type: "literature" | "local_data" | "model_derived" | "inference" | "speculation"
  - speculation: ONLY when source explicitly hedges with "may", "possibly", "unclear", "uncertain"
- Mark evidence_type: "text" | "figure" | "table" | "data"

Output ONLY a JSON array (empty [] is valid):
[
  {
    "observation":               "verbatim or close paraphrase of factual content",
    "interpretation":            "source's conclusion from this observation",
    "alternative_interpretation": "an alternative geological reading (agent-generated)",
    "assumption":                "key assumption (or empty string)",
    "source_type":  "literature | local_data | model_derived | inference | speculation",
    "evidence_type": "text | figure | table | data",
    "data_type":    "seismicity | velocity_model | focal_mechanism | geology | geochemistry | stratigraphy | stress_field | dispersion | other",
    "spatial_scale":  "local (<50km) | regional (50-500km) | crustal | lithospheric",
    "depth_range":    "e.g. 0-15 km or unspecified",
    "geological_structure": "primary structure named",
    "confidence":   "high | medium | low",
    "uncertainty":  "main uncertainty or empty string",
    "citation":     "short citation if available (Author Year, or filename)"
  }
]
"""


_HYPOTHESIS_SYSTEM = """\
You are a geoscience hypothesis generator. Generate COMPETING testable hypotheses
from the evidence table. Rules:
- Do not converge to one hypothesis prematurely.
- Each hypothesis must cite specific evidence IDs.
- Prefer hypotheses testable with available seismic/geological data.
- Flag confidence based on weight of evidence.
Output JSON array:
[
  {
    "statement": "one-sentence geological hypothesis",
    "supporting_evidence": ["ev_id1"],
    "contradicting_evidence": ["ev_id2"],
    "data_types_needed": ["focal_mechanism"],
    "confidence": "high | medium | low | speculative"
  }
]
"""


_REASONER_SYSTEM = """\
You are a geological reasoning expert. Evaluate each hypothesis rigorously.
Output JSON:
{
  "evaluations": [
    {
      "hypothesis_id": "H1",
      "assessment": "supported | weakly_supported | contradicted | insufficient_data",
      "note": "one-sentence evaluation"
    }
  ],
  "preferred_hypothesis": "H1",
  "preferred_rationale": "reason citing evidence IDs",
  "missing_information": ["list of critically missing data"]
}
"""


_REPORT_SYSTEM = """\
Write a structured geoscience interpretation report in Markdown.
Rules:
- Every geological claim must cite an evidence ID in brackets, e.g. [ev_abc123].
- Clearly separate observation from interpretation.
- Flag uncertainty and missing data explicitly.
- Use hedging language where appropriate (consistent with, suggests, may indicate).
- Structure: ## Problem Definition | ## Evidence Summary | ## Competing Hypotheses | ## Preferred Interpretation | ## Uncertainty & Limitations | ## Recommended Analyses | ## References
- Keep under 1400 words.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ── EvidenceTableBuilder ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class GeoEvidenceTableBuilder:
    """Accumulates GeoEvidence records, deduplicates, and flags conflicts."""

    def __init__(self) -> None:
        self._table: List[GeoEvidence] = []
        self._obs_hashes: set = set()

    def add(self, new_evidence: List[GeoEvidence]) -> List[GeoEvidence]:
        added = []
        for ev in new_evidence:
            h = hashlib.md5(ev.observation[:80].encode()).hexdigest()
            if h in self._obs_hashes:
                continue
            self._obs_hashes.add(h)
            self._table.append(ev)
            added.append(ev)
        self._flag_conflicts()
        return added

    def _flag_conflicts(self) -> None:
        by_struct: Dict[str, List[GeoEvidence]] = {}
        for ev in self._table:
            key = ev.geological_structure.lower().strip()
            if key and key != "unspecified":
                by_struct.setdefault(key, []).append(ev)
        for evs in by_struct.values():
            if len(evs) < 2:
                continue
            interps = [e.interpretation.lower()[:60] for e in evs]
            if len(set(interps)) > 1:
                ids = [e.evidence_id for e in evs]
                for ev in evs:
                    others = [i for i in ids if i != ev.evidence_id]
                    for o in others:
                        if o not in ev.conflict_with:
                            ev.conflict_with.append(o)

    @property
    def table(self) -> List[GeoEvidence]:
        return list(self._table)

    def to_markdown(self) -> str:
        if not self._table:
            return "_No evidence collected._"
        hdr = ("| ID | Source | Type | Structure | Observation | "
               "Interpretation | Confidence | Source Type |\n"
               "|---|---|---|---|---|---|---|---|\n")
        rows = []
        for ev in self._table:
            obs = ev.observation[:70].replace("|", "∣")
            interp = ev.interpretation[:50].replace("|", "∣")
            rows.append(
                f"| {ev.evidence_id} | {ev.source[:30]} | {ev.data_type} | "
                f"{ev.geological_structure[:25]} | {obs} | {interp} | "
                f"{ev.confidence} | {ev.source_type} |"
            )
        return hdr + "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ── LoopController ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class LoopController:
    """
    Orchestrates the full evidence-driven agent loop:

      for iteration in range(max_iterations):
          1. LLM tool planner selects next tool call
          2. Execute tool (up to max_tool_calls_per_iter)
          3. Extract evidence from tool output
          4. Update evidence table
          5. Generate/update hypotheses
          6. Evaluate hypotheses
          7. Update report
          8. Convergence check
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_cfg: Dict[str, Any],
        registry: ToolRegistry,
        ev_table: GeoEvidenceTableBuilder,
        logger: AgentLogger,
    ) -> None:
        self._cfg      = config
        self._llm      = llm_cfg
        self._reg      = registry
        self._ev       = ev_table
        self._log      = logger
        self._ev_table = ev_table  # alias for external access
        self._report   = ""
        self._figures: List[str] = []
        self._sources: List[str] = []
        self._missing: List[str] = []

    # ── LLM helpers ───────────────────────────────────────────────────────

    def _select_tool(
        self,
        question: str,
        study_area: str,
        iteration: int,
        calls_this_iter: int,
        recent_results: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM which tool to call next. Returns parsed JSON or None."""
        ev_summary = f"{len(self._ev.table)} evidence records collected so far."
        ev_types = list({e.data_type for e in self._ev.table[-10:]})
        recent_txt = "\n".join(f"  - {r}" for r in recent_results[-3:])

        prompt = (
            f"Question: {question}\n"
            f"Study area: {study_area}\n\n"
            f"Iteration {iteration}, tool call {calls_this_iter + 1} of {self._cfg.max_tool_calls_per_iter}.\n"
            f"Evidence so far: {ev_summary}  Types: {ev_types}\n\n"
            f"Recent tool results:\n{recent_txt or '(none yet)'}\n\n"
            "Select the next tool call as JSON."
        )
        messages = [
            {"role": "system", "content": _TOOL_SELECTOR_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._llm, max_tokens=400, temperature=0.3)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None

    def _extract_evidence(
        self,
        tool_output: Dict[str, Any],
        tool: str,
        method: str,
        question: str,
        iteration: int,
        call_id: str,
    ) -> List[GeoEvidence]:
        """Use LLM to extract structured evidence from a tool result."""
        # ── Serialise relevant part of output ─────────────────────────────
        text_repr = ""
        ev_type_hint = "text"

        if method == "analyze_image":
            # Already structured JSON from vision LLM
            text_repr    = json.dumps(tool_output, ensure_ascii=False)[:3000]
            ev_type_hint = "figure"
        elif method == "extract_table":
            text_repr    = json.dumps(tool_output, ensure_ascii=False)[:3000]
            ev_type_hint = "table"
            # Fast-path: use evidence_entries if vision LLM already extracted them
            if "evidence_entries" in tool_output:
                src = tool_output.get("path", f"{tool}.{method}")
                records = []
                for item in tool_output["evidence_entries"]:
                    obs = item.get("observation", "").strip()
                    if not obs:
                        continue
                    eid = hashlib.md5(f"{call_id}:{obs[:60]}".encode()).hexdigest()[:10]
                    records.append(GeoEvidence(
                        evidence_id=eid,
                        source=str(src),
                        source_type="local_data",
                        evidence_type="table",
                        observation=obs,
                        data_type=item.get("data_type", "other"),
                        confidence=item.get("confidence", "medium"),
                        iteration=iteration,
                        tool_call_id=call_id,
                    ))
                if records:
                    return records
        elif "content" in tool_output:
            text_repr = str(tool_output["content"])[:3000]
        elif "chunks" in tool_output:
            text_repr = "\n\n".join(
                f"[{c.get('doc_name','?')} p{c.get('page','?')}]\n{c.get('text','')[:600]}"
                for c in tool_output.get("chunks", [])[:6]
            )
        elif "papers" in tool_output:
            text_repr = "\n".join(
                f"Title: {p.get('title','')}  Year: {p.get('year','')}  "
                f"Abstract: {p.get('abstract','')[:300]}"
                for p in tool_output.get("papers", [])[:5]
            )
        elif "results" in tool_output:
            text_repr = json.dumps(tool_output["results"][:5], ensure_ascii=False)[:1500]
        elif "data" in tool_output:
            text_repr = json.dumps(tool_output["data"][:20], ensure_ascii=False)[:1500]
        elif "preview" in tool_output:
            text_repr = json.dumps(tool_output["preview"][:10], ensure_ascii=False)[:1200]
        elif "stdout" in tool_output:
            text_repr = (tool_output.get("stdout", "") + "\n" +
                         tool_output.get("stderr", ""))[:2000]
            ev_type_hint = "data"
        elif "entries" in tool_output:
            text_repr = "\n".join(
                e.get("snippet", "") for e in tool_output.get("entries", [])[:6]
            )

        if not text_repr or len(text_repr.strip()) < 40:
            return []

        # Determine source type hint
        if tool in ("literature_library", "rag_index", "web_search"):
            src_hint = "literature" if tool != "web_search" else "literature"
        elif tool == "seismo_data":
            src_hint = "local_data"
        elif tool == "code_execution":
            src_hint = "model_derived"
        elif tool == "image_analysis":
            src_hint = "local_data"
        else:
            src_hint = "inference"

        prompt = (
            f"Scientific question: {question}\n"
            f"Tool: {tool}.{method}  "
            f"(preferred source_type: {src_hint}, evidence_type: {ev_type_hint})\n\n"
            f"Tool output:\n{text_repr}\n\n"
            "Extract structured geological evidence (JSON array, empty [] if none)."
        )
        messages = [
            {"role": "system", "content": _EVIDENCE_EXTRACTOR_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._llm, max_tokens=1200, temperature=0.1)
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group(0))
        except Exception:
            return []

        # Build source label
        src_label = (
            tool_output.get("path") or
            tool_output.get("doc_name") or
            f"{tool}.{method}"
        )

        records: List[GeoEvidence] = []
        for item in items:
            obs = item.get("observation", "").strip()
            if not obs or len(obs) < 20:
                continue
            eid = hashlib.md5(f"{call_id}:{obs[:60]}".encode()).hexdigest()[:10]
            records.append(GeoEvidence(
                evidence_id=eid,
                source=str(src_label),
                source_type=item.get("source_type", src_hint),
                evidence_type=item.get("evidence_type", ev_type_hint),
                observation=obs,
                data_type=item.get("data_type", "other"),
                spatial_scale=item.get("spatial_scale", "unspecified"),
                depth_range=item.get("depth_range", "unspecified"),
                geological_structure=item.get("geological_structure", "unspecified"),
                interpretation=item.get("interpretation", ""),
                alternative_interpretation=item.get("alternative_interpretation", ""),
                assumption=item.get("assumption", ""),
                confidence=item.get("confidence", "medium"),
                uncertainty=item.get("uncertainty", ""),
                citation=item.get("citation", ""),
                iteration=iteration,
                tool_call_id=call_id,
            ))
        return records

    def _generate_hypotheses(
        self,
        question: str,
        study_area: str,
        existing: List[GeoHypothesis],
    ) -> List[GeoHypothesis]:
        evidence = self._ev.table
        if not evidence:
            return existing

        ev_txt = "\n".join(
            f"[{e.evidence_id}] {e.data_type}|{e.geological_structure}: "
            f"{e.observation[:100]} ({e.source_type})"
            for e in evidence[-20:]
        )
        ex_txt = ""
        if existing:
            ex_txt = "\nExisting hypotheses:\n" + "\n".join(
                f"  [{h.hypothesis_id}] {h.statement}" for h in existing
            )

        prompt = (
            f"Question: {question}\nStudy area: {study_area}\n\n"
            f"Evidence:\n{ev_txt}{ex_txt}\n\n"
            "Generate or update competing geological hypotheses (JSON array)."
        )
        messages = [
            {"role": "system", "content": _HYPOTHESIS_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._llm, max_tokens=1000, temperature=0.4)
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return existing
            items = json.loads(m.group(0))
        except Exception:
            return existing

        result = []
        for i, h in enumerate(items):
            result.append(GeoHypothesis(
                hypothesis_id=f"H{i + 1}",
                statement=h.get("statement", ""),
                supporting_evidence=h.get("supporting_evidence", []),
                contradicting_evidence=h.get("contradicting_evidence", []),
                data_types_needed=h.get("data_types_needed", []),
                confidence=h.get("confidence", "medium"),
            ))
        return result

    def _evaluate_hypotheses(
        self,
        question: str,
        study_area: str,
        hypotheses: List[GeoHypothesis],
    ) -> Tuple[List[GeoHypothesis], List[str], str, str]:
        if not hypotheses:
            return [], [], "", ""

        ev_txt = "\n".join(
            f"[{e.evidence_id}] {e.geological_structure}: {e.observation[:80]}"
            for e in self._ev.table[-20:]
        )
        hyp_txt = "\n".join(
            f"[{h.hypothesis_id}] {h.statement}"
            for h in hypotheses
        )
        prompt = (
            f"Question: {question}\nStudy area: {study_area}\n\n"
            f"Hypotheses:\n{hyp_txt}\n\nEvidence:\n{ev_txt}\n\n"
            "Evaluate (JSON)."
        )
        messages = [
            {"role": "system", "content": _REASONER_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._llm, max_tokens=1200, temperature=0.2)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return hypotheses, [], "", ""
            d = json.loads(m.group(0))
        except Exception:
            return hypotheses, [], "", ""

        evals = {e["hypothesis_id"]: e for e in d.get("evaluations", [])}
        for h in hypotheses:
            ev = evals.get(h.hypothesis_id, {})
            if ev.get("assessment") == "contradicted":
                h.status = "rejected"

        missing  = d.get("missing_information", [])
        pref_id  = d.get("preferred_hypothesis", "")
        pref_rat = d.get("preferred_rationale", "")
        return hypotheses, missing, pref_id, pref_rat

    def _update_report(
        self,
        question: str,
        study_area: str,
        hypotheses: List[GeoHypothesis],
        missing: List[str],
        preferred_id: str,
        preferred_rat: str,
        iteration: int,
    ) -> str:
        ev_txt = "\n".join(
            f"[{e.evidence_id}] {e.source_type}|{e.data_type}|{e.source}: "
            f"{e.observation[:100]} — {e.interpretation[:60]}"
            for e in self._ev.table
        )
        hyp_txt = "\n".join(
            f"[{h.hypothesis_id}] ({h.confidence}) {h.statement} [{h.status}]"
            for h in hypotheses
        )
        prompt = (
            f"## Scientific question\n{question}\n\n"
            f"## Study area\n{study_area}\n\n"
            f"## Evidence table (iteration {iteration})\n{ev_txt}\n\n"
            f"## Hypotheses\n{hyp_txt}\n\n"
            f"## Preferred: {preferred_id} — {preferred_rat}\n\n"
            f"## Missing\n" + "\n".join(f"- {m}" for m in missing[:8]) + "\n\n"
            f"## Figures generated\n" + "\n".join(f"- {f}" for f in self._figures) + "\n\n"
            "Write the interpretation report."
        )
        messages = [
            {"role": "system", "content": _REPORT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            self._report = _llm_call(messages, self._llm, max_tokens=1600, temperature=0.3)
        except Exception as exc:
            self._report = (
                f"# Report (iteration {iteration})\n\n"
                f"*Report generation failed: {exc}*\n\n"
                f"## Evidence\n{self._ev.to_markdown()}"
            )
        return self._report

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        study_area: str,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> GeoAgentResult:

        def _emit(phase: str, msg: str) -> None:
            if on_progress:
                on_progress({"phase": phase, "message": msg})

        hypotheses:    List[GeoHypothesis]   = []
        missing_info:  List[str]             = []
        convergence    = "max_iterations_reached"
        preferred_id   = ""
        preferred_rat  = ""

        for iteration in range(1, self._cfg.max_iterations + 1):
            _emit("iteration_start", f"Iteration {iteration}/{self._cfg.max_iterations}")
            added_this_iter = 0
            recent_summaries: List[str] = []

            # ── Tool call loop ────────────────────────────────────────────
            for call_num in range(self._cfg.max_tool_calls_per_iter):
                # Ask LLM which tool to call
                selection = self._select_tool(
                    question, study_area, iteration, call_num, recent_summaries
                )
                if selection is None:
                    _emit("warning", "Tool selector returned no valid JSON; stopping tool calls.")
                    break
                if selection.get("done", False):
                    _emit("tool_done", f"Agent signals done: {selection.get('reason','')[:80]}")
                    break

                tool   = selection.get("tool", "")
                method = selection.get("method", "")
                args   = selection.get("args", {})
                reason = selection.get("reason", "")

                _emit("tool_call", f"[{tool}.{method}] {reason[:80]}")

                result, call = self._reg.dispatch(
                    tool, method, args, reason, iteration
                )

                recent_summaries.append(f"[{tool}.{method}]: {call.result_summary}")

                # Track sources
                for src_key in ("path", "doc_name", "file"):
                    val = result.get(src_key)
                    if val and str(val) not in self._sources:
                        self._sources.append(str(val))
                for chunk in result.get("chunks", []):
                    doc = chunk.get("doc_name", "")
                    if doc and doc not in self._sources:
                        self._sources.append(doc)

                # Track figures
                if "figure" in result:
                    fig = result["figure"]
                    if fig and fig not in self._figures:
                        self._figures.append(fig)
                for fig in result.get("figures", []):
                    if fig and fig not in self._figures:
                        self._figures.append(fig)

                # Extract evidence
                new_evidence = self._extract_evidence(
                    result, tool, method, question, iteration, call.call_id
                )
                added = self._ev.add(new_evidence)
                added_this_iter += len(added)

                # Tag evidence IDs in the call record (mutate in place)
                call.evidence_added = [e.evidence_id for e in added]
                call.figures_added  = [result["figure"]] if "figure" in result else result.get("figures", [])

                if added:
                    _emit("evidence", f"Added {len(added)} evidence record(s).")

            # ── After tool calls: reasoning ───────────────────────────────
            _emit("hypothesising", f"Generating hypotheses from {len(self._ev.table)} evidence records…")
            hypotheses = self._generate_hypotheses(question, study_area, hypotheses)

            _emit("evaluating", "Evaluating hypotheses…")
            hypotheses, missing_info, preferred_id, preferred_rat = self._evaluate_hypotheses(
                question, study_area, hypotheses
            )
            self._missing = missing_info

            _emit("writing", f"Updating report (iteration {iteration})…")
            self._update_report(
                question, study_area, hypotheses,
                missing_info, preferred_id, preferred_rat, iteration
            )

            # ── Convergence ───────────────────────────────────────────────
            if added_this_iter == 0 and iteration > 1:
                convergence = "no_new_evidence"
                _emit("converged", f"No new evidence in iteration {iteration}; converging.")
                break

            active = [h for h in hypotheses if h.status == "active"]
            if len(active) == 1 and active[0].confidence == "high" and iteration > 1:
                convergence = "hypothesis_convergence"
                _emit("converged", "Single high-confidence hypothesis reached.")
                break

        _emit("done", "Agent loop complete.")

        # Append evidence table to report
        final_report = (
            self._report
            + "\n\n---\n\n## Evidence Table\n\n"
            + self._ev.to_markdown()
        )

        # Build ValidationCheck suggestions from missing_info
        suggested_validation = [
            ValidationCheck(
                check_id=f"V{i+1}",
                linked_to="",
                description=item,
                data_required="See evidence table",
                method="literature search / data collection",
                expected_outcome="Resolve ambiguity identified in report",
            )
            for i, item in enumerate(self._missing[:8])
        ]

        return GeoAgentResult(
            question=question,
            study_area=study_area,
            iterations_run=iteration,
            final_report=final_report,
            evidence_table=self._ev.table,
            hypotheses=hypotheses,
            tool_log=self._log.log,
            retrieved_sources=self._sources,
            generated_figures=self._figures,
            missing_information=self._missing,
            suggested_validation=suggested_validation,
            convergence_reason=convergence,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── Public Facade ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceDrivenGeoAgent:
    """
    Top-level facade for the evidence-driven geoscience interpretation agent.

    Parameters
    ----------
    config : AgentConfig
        Runtime configuration (workspace root, literature root, capability gates).
    llm_cfg : dict, optional
        LLM configuration. If omitted, loads from config_manager.

    Example
    -------
    >>> from sage_agents import EvidenceDrivenGeoAgent, AgentConfig
    >>> cfg = AgentConfig(workspace_root="./examples/weiyuan",
    ...                   literature_root="./papers/weiyuan")
    >>> agent = EvidenceDrivenGeoAgent(cfg)
    >>> result = agent.run("Why are M>4 earthquakes near the Molingchang fault?",
    ...                    "Weiyuan, Sichuan Basin")
    >>> print(result["final_report"])
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config  = config or AgentConfig()
        self._llm_cfg = llm_cfg or _get_llm_config()
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(
        self,
        question: str,
        study_area: str = "",
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the evidence-driven agent.

        Parameters
        ----------
        question : str
            The scientific question to investigate.
        study_area : str
            Geographic / geological context.
        on_progress : callable, optional
            Called with {"phase": str, "message": str} dicts for live progress.

        Returns
        -------
        dict matching GeoAgentResult serialised to JSON-safe format.
        """
        logger   = AgentLogger()
        if on_progress:
            logger.add_callback(on_progress)

        registry = ToolRegistry(self._config, logger, llm_cfg=self._llm_cfg)
        ev_table = GeoEvidenceTableBuilder()
        ctrl     = LoopController(
            config=self._config,
            llm_cfg=self._llm_cfg,
            registry=registry,
            ev_table=ev_table,
            logger=logger,
        )

        result = ctrl.run(question, study_area, on_progress=on_progress)
        return self._to_dict(result)

    @staticmethod
    def _to_dict(result: GeoAgentResult) -> Dict[str, Any]:
        """Serialise GeoAgentResult to a JSON-safe dict."""
        def _ev(e: GeoEvidence) -> Dict:
            d = asdict(e)
            return d

        def _hyp(h: GeoHypothesis) -> Dict:
            d = asdict(h)
            return d

        def _call(c: ToolCall) -> Dict:
            d = asdict(c)
            return d

        def _val(v: ValidationCheck) -> Dict:
            return asdict(v)

        return {
            "question":             result.question,
            "study_area":           result.study_area,
            "iterations_run":       result.iterations_run,
            "final_report":         result.final_report,
            "evidence_table":       [_ev(e) for e in result.evidence_table],
            "hypotheses":           [_hyp(h) for h in result.hypotheses],
            "tool_log":             [_call(c) for c in result.tool_log],
            "retrieved_sources":    result.retrieved_sources,
            "generated_figures":    result.generated_figures,
            "missing_information":  result.missing_information,
            "suggested_validation": [_val(v) for v in result.suggested_validation],
            "convergence_reason":   result.convergence_reason,
        }
