"""
literature_loop_agent.py — Iterative Geoscience Literature Interpretation Agent

Architecture
------------
  TaskPlanner → LiteratureRetriever → PaperReader → EvidenceTableBuilder
      → HypothesisGenerator → GeologicalReasoner → ValidationPlanner
      → ReportWriter → LoopController (orchestrates all, runs N loops)

Each component is a stateless class that takes structured input and returns
structured output via LLM calls.  The LoopController maintains all state and
drives the iteration loop.

Usage (programmatic)
--------------------
  from sage_agents import LiteratureLoopAgent

  agent = LiteratureLoopAgent(llm_config)
  result = agent.run(
      question="Why are M>4 earthquakes near the Molingchang fault?",
      study_area="Weiyuan, Sichuan Basin",
      max_iterations=3,
  )
  print(result.final_report)

Usage (Flask)
-------------
  POST /api/literature_loop
  {"question": "...", "study_area": "...", "max_iterations": 3}

Usage (CLI)
-----------
  python seismic_cli.py literature-loop --question "..." --study-area "..."
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ── LLM utility (standalone, no circular imports with app.py) ──────────────
# ---------------------------------------------------------------------------

def _llm_call(
    messages: List[Dict[str, str]],
    llm_cfg: Dict[str, Any],
    max_tokens: int = 2000,
    temperature: float = 0.3,
) -> str:
    """Call the configured LLM backend.  Returns the response text."""
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
    """Load LLM config from config_manager (same as app.py)."""
    try:
        # resolve path: sage_agents/ is one level below project root
        _root = str(Path(__file__).parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from config_manager import LLMConfigManager
        return LLMConfigManager().get_llm_config()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# ── RAG retrieval helper ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _get_kb():
    """Return the KnowledgeBase singleton (or None if unavailable)."""
    try:
        _webdir = str(Path(__file__).parent.parent / "web_app")
        if _webdir not in sys.path:
            sys.path.insert(0, _webdir)
        from rag_engine import get_knowledge_base
        return get_knowledge_base()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ── Data classes ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTask:
    """A single reasoning step in the decomposed plan."""
    task_id:     str
    task_type:   str   # "define_problem" | "retrieve" | "read" | "compare" | ...
    description: str
    status:      str = "pending"   # pending | running | done


@dataclass
class RetrievedChunk:
    """A single document chunk retrieved from the RAG knowledge base."""
    chunk_id:  str
    doc_name:  str
    page:      int
    text:      str
    score:     float = 0.0
    query:     str   = ""


@dataclass
class Evidence:
    """A single piece of geological/geophysical evidence extracted from literature."""
    evidence_id:        str
    source:             str            # doc_name (page N)
    observation:        str            # factual observation, verbatim or close paraphrase
    data_type:          str            # seismicity | velocity | focal_mechanism | geology | ...
    spatial_scale:      str            # local | regional | crustal | lithospheric
    depth_range:        str            # e.g. "0–15 km" or "lower crust"
    geological_structure: str          # fault | basin | Moho | volcanic arc | ...
    interpretation:     str            # author's interpretation (may differ from observation)
    confidence:         str            # high | medium | low
    uncertainty:        str            # what the authors acknowledge as uncertain
    conflict_with:      List[str]      # evidence_ids this conflicts with
    notes:              str = ""
    iteration:          int = 0        # which loop iteration added this


@dataclass
class Hypothesis:
    """A testable geological hypothesis generated from the evidence."""
    hypothesis_id:       str
    statement:           str           # one-sentence hypothesis
    supporting_evidence: List[str]     # evidence_ids
    contradicting_evidence: List[str]  # evidence_ids
    data_types_needed:   List[str]     # what additional data would strengthen/falsify
    confidence:          str           # high | medium | low | speculative
    status:              str = "active"  # active | rejected | merged


@dataclass
class ValidationCheck:
    """A concrete analysis suggested to test a hypothesis."""
    check_id:      str
    linked_to:     str        # hypothesis_id
    description:   str        # what to do
    data_required: str        # what data/figures/files are needed
    method:        str        # how to do it (plot, statistical test, comparison, ...)
    expected_outcome: str     # what a positive/negative result would mean


@dataclass
class AgentResult:
    """Final output of the literature loop agent."""
    question:             str
    study_area:           str
    iterations_run:       int
    final_report:         str              # Markdown report
    evidence_table:       List[Evidence]
    hypotheses:           List[Hypothesis]
    missing_information:  List[str]
    suggested_validation: List[ValidationCheck]
    retrieved_sources:    List[str]
    convergence_reason:   str             # why the loop stopped


# ---------------------------------------------------------------------------
# ── Component 1: TaskPlanner ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a scientific reasoning planner for geoscience and seismology.
Given a scientific question and a study area, decompose the problem into a
prioritised list of reasoning tasks.

Output ONLY a JSON array of task objects, strictly in this format:
[
  {"task_id": "T1", "task_type": "define_problem",   "description": "..."},
  {"task_id": "T2", "task_type": "retrieve",         "description": "..."},
  {"task_id": "T3", "task_type": "read_evidence",    "description": "..."},
  {"task_id": "T4", "task_type": "compare",          "description": "..."},
  {"task_id": "T5", "task_type": "hypothesise",      "description": "..."},
  {"task_id": "T6", "task_type": "evaluate",         "description": "..."},
  {"task_id": "T7", "task_type": "validate",         "description": "..."},
  {"task_id": "T8", "task_type": "write_report",     "description": "..."}
]
Valid task_types: define_problem, retrieve, read_evidence, compare,
  hypothesise, evaluate, validate, write_report.
No extra text outside the JSON block.
"""


class TaskPlanner:
    """Decomposes a scientific question into ordered reasoning tasks."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = llm_cfg

    def plan(self, question: str, study_area: str) -> List[ReasoningTask]:
        prompt = (
            f"Scientific question: {question}\n"
            f"Study area: {study_area}\n\n"
            "Decompose this into reasoning tasks (JSON array)."
        )
        messages = [
            {"role": "system",  "content": _PLANNER_SYSTEM},
            {"role": "user",    "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._cfg, max_tokens=800, temperature=0.2)
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            tasks_raw = json.loads(m.group(0)) if m else []
            return [ReasoningTask(**t) for t in tasks_raw]
        except Exception:
            # Fallback: generic task list
            return [
                ReasoningTask("T1", "define_problem",  "Define the geological problem"),
                ReasoningTask("T2", "retrieve",        "Retrieve relevant literature"),
                ReasoningTask("T3", "read_evidence",   "Extract evidence from papers"),
                ReasoningTask("T4", "compare",         "Compare evidence across papers"),
                ReasoningTask("T5", "hypothesise",     "Generate geological hypotheses"),
                ReasoningTask("T6", "evaluate",        "Evaluate hypotheses against evidence"),
                ReasoningTask("T7", "validate",        "Design validation checks"),
                ReasoningTask("T8", "write_report",    "Write interpretation report"),
            ]


# ---------------------------------------------------------------------------
# ── Component 2: LiteratureRetriever ───────────────────────────────────────
# ---------------------------------------------------------------------------

class LiteratureRetriever:
    """Wraps the existing RAG knowledge base for iterative retrieval."""

    def __init__(self, top_k: int = 8, score_threshold: float = 0.35) -> None:
        self._top_k     = top_k
        self._threshold = score_threshold
        self._kb        = _get_kb()
        self._seen_ids: set = set()   # prevent exact-duplicate chunks across iterations

    @property
    def available(self) -> bool:
        return self._kb is not None and not self._kb.is_empty

    def retrieve(
        self,
        query: str,
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks for one or more queries, de-duplicated against
        previously returned chunks.
        """
        if not self.available:
            return []

        queries = [query] + (extra_queries or [])
        seen_this_call: set = set()
        results: List[RetrievedChunk] = []

        for q in queries:
            try:
                hits = self._kb.retrieve(
                    q,
                    top_k=self._top_k,
                    score_threshold=self._threshold,
                )
            except Exception:
                continue

            for chunk, score in hits:
                if chunk.chunk_id in self._seen_ids:
                    continue
                if chunk.chunk_id in seen_this_call:
                    continue
                seen_this_call.add(chunk.chunk_id)
                results.append(RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    page=chunk.page,
                    text=chunk.text,
                    score=score,
                    query=q,
                ))

        # Mark all as seen for next iteration
        for r in results:
            self._seen_ids.add(r.chunk_id)

        return sorted(results, key=lambda x: x.score, reverse=True)

    def refine_query(
        self,
        original_question: str,
        evidence_so_far: List[Evidence],
        llm_cfg: Dict[str, Any],
    ) -> List[str]:
        """Ask the LLM to generate more targeted sub-queries."""
        if not evidence_so_far:
            return [original_question]

        obs_sample = "\n".join(
            f"- [{e.source}] {e.observation[:120]}"
            for e in evidence_so_far[-6:]
        )
        prompt = (
            f"Original question: {original_question}\n\n"
            f"Evidence collected so far:\n{obs_sample}\n\n"
            "Generate 3 targeted retrieval sub-queries to find missing evidence. "
            "Output as a JSON array of strings only."
        )
        messages = [
            {"role": "system", "content": "You are a literature search strategist for geoscience."},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, llm_cfg, max_tokens=300, temperature=0.4)
            m   = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))[:3]
        except Exception:
            pass
        return [original_question]


# ---------------------------------------------------------------------------
# ── Component 3: PaperReader ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_READER_SYSTEM = """\
You are a geoscience literature analyst. Given a text chunk from a research paper
and the scientific question being investigated, extract structured information.

RULES:
- Strictly separate OBSERVATION (what the data shows) from INTERPRETATION (author's explanation).
- Never add your own geological interpretations — only extract what the text states.
- Mark anything speculative or uncertain explicitly.
- If a field cannot be determined from the text, use "unspecified".

Output a JSON object with these exact keys:
{
  "observation": "factual observation from the data (verbatim or close paraphrase)",
  "data_type": "seismicity | velocity_model | focal_mechanism | geology | tomography | geodesy | geochemistry | stratigraphy | stress_field | other",
  "spatial_scale": "local (<50 km) | regional (50-500 km) | crustal | lithospheric",
  "depth_range": "e.g. 0-15 km, lower crust, or unspecified",
  "geological_structure": "primary structure discussed (fault name, basin, Moho, etc.)",
  "interpretation": "author's geological interpretation of the observation",
  "assumptions": "key assumptions stated or implied",
  "limitations": "limitations acknowledged by the authors",
  "confidence": "high | medium | low",
  "uncertainty": "main source of uncertainty"
}
No extra text outside the JSON object.
"""


class PaperReader:
    """Extracts structured evidence from retrieved document chunks."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = llm_cfg

    def read(
        self,
        chunk: RetrievedChunk,
        question: str,
        iteration: int = 0,
    ) -> Optional[Evidence]:
        """Parse a single chunk into an Evidence record."""
        if len(chunk.text.strip()) < 80:
            return None   # too short to be meaningful

        prompt = (
            f"Scientific question: {question}\n\n"
            f"Paper: {chunk.doc_name}  (page {chunk.page + 1})\n\n"
            f"Text excerpt:\n{chunk.text[:2500]}\n\n"
            "Extract structured evidence (JSON)."
        )
        messages = [
            {"role": "system", "content": _READER_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._cfg, max_tokens=600, temperature=0.1)
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            d = json.loads(m.group(0))
        except Exception:
            return None

        obs = d.get("observation", "").strip()
        if not obs or obs.lower() in ("", "unspecified", "n/a"):
            return None

        eid = hashlib.md5(f"{chunk.chunk_id}:{obs[:60]}".encode()).hexdigest()[:10]
        return Evidence(
            evidence_id=eid,
            source=f"{chunk.doc_name} (p.{chunk.page + 1})",
            observation=obs,
            data_type=d.get("data_type", "unspecified"),
            spatial_scale=d.get("spatial_scale", "unspecified"),
            depth_range=d.get("depth_range", "unspecified"),
            geological_structure=d.get("geological_structure", "unspecified"),
            interpretation=d.get("interpretation", ""),
            confidence=d.get("confidence", "medium"),
            uncertainty=d.get("uncertainty", ""),
            conflict_with=[],
            notes=d.get("assumptions", "") + " | " + d.get("limitations", ""),
            iteration=iteration,
        )

    def read_batch(
        self,
        chunks: List[RetrievedChunk],
        question: str,
        iteration: int = 0,
        on_progress: Optional[Callable] = None,
    ) -> List[Evidence]:
        """Read multiple chunks, skipping failures."""
        results = []
        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress({"phase": "reading", "msg": f"Reading chunk {i+1}/{len(chunks)}: {chunk.doc_name}"})
            ev = self.read(chunk, question, iteration)
            if ev is not None:
                results.append(ev)
        return results


# ---------------------------------------------------------------------------
# ── Component 4: EvidenceTableBuilder ──────────────────────────────────────
# ---------------------------------------------------------------------------

class EvidenceTableBuilder:
    """
    Accumulates Evidence records across iterations; detects conflicts.
    Maintains a canonical evidence list (no duplicate observation+source pairs).
    """

    def __init__(self) -> None:
        self._table: List[Evidence] = []
        self._obs_hashes: set = set()

    def add(self, new_evidence: List[Evidence]) -> List[Evidence]:
        """
        Merge new evidence into the table. Returns only genuinely new records.
        """
        added = []
        for ev in new_evidence:
            h = hashlib.md5(ev.observation[:80].encode()).hexdigest()
            if h in self._obs_hashes:
                continue
            self._obs_hashes.add(h)
            self._table.append(ev)
            added.append(ev)
        # Detect conflicts (same structure, different interpretations)
        self._flag_conflicts()
        return added

    def _flag_conflicts(self) -> None:
        """Mark evidence pairs that share structure but have different interpretations."""
        by_structure: Dict[str, List[Evidence]] = {}
        for ev in self._table:
            key = ev.geological_structure.lower().strip()
            if key and key != "unspecified":
                by_structure.setdefault(key, []).append(ev)

        for evs in by_structure.values():
            if len(evs) < 2:
                continue
            interps = [e.interpretation.lower()[:60] for e in evs]
            # Simple heuristic: if interpretations differ substantially, flag conflict
            if len(set(interps)) > 1:
                ids = [e.evidence_id for e in evs]
                for ev in evs:
                    others = [i for i in ids if i != ev.evidence_id]
                    for o in others:
                        if o not in ev.conflict_with:
                            ev.conflict_with.append(o)

    @property
    def table(self) -> List[Evidence]:
        return list(self._table)

    def to_markdown(self) -> str:
        if not self._table:
            return "_No evidence collected yet._"
        header = (
            "| ID | Source | Observation | Data Type | Structure | "
            "Interpretation | Confidence | Conflicts |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for ev in self._table:
            conflicts = ", ".join(ev.conflict_with) or "—"
            obs = ev.observation[:80].replace("|", "∣")
            interp = ev.interpretation[:60].replace("|", "∣")
            rows.append(
                f"| {ev.evidence_id} | {ev.source} | {obs} | "
                f"{ev.data_type} | {ev.geological_structure} | "
                f"{interp} | {ev.confidence} | {conflicts} |"
            )
        return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# ── Component 5: HypothesisGenerator ───────────────────────────────────────
# ---------------------------------------------------------------------------

_HYPOTHESIS_SYSTEM = """\
You are a geoscience hypothesis generator for tectonic, seismological, and
fault-zone interpretation problems.

RULES:
- Generate COMPETING hypotheses — do not converge prematurely.
- Each hypothesis must cite specific evidence IDs that support or contradict it.
- Do not generate hypotheses unsupported by the evidence table.
- Clearly distinguish between hypothesis and observation.
- Prefer hypotheses testable with available geoscience data.
- Indicate confidence level based on weight of evidence.

Output a JSON array of hypothesis objects:
[
  {
    "statement": "one-sentence geological hypothesis",
    "supporting_evidence": ["ev_id1", "ev_id2"],
    "contradicting_evidence": ["ev_id3"],
    "data_types_needed": ["focal_mechanisms", "tomography"],
    "confidence": "high | medium | low | speculative"
  }
]
No extra text outside the JSON array.
"""


class HypothesisGenerator:
    """Generates competing geological hypotheses from the evidence table."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = llm_cfg

    def generate(
        self,
        question: str,
        study_area: str,
        evidence: List[Evidence],
        existing_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> List[Hypothesis]:
        if not evidence:
            return []

        ev_summary = "\n".join(
            f"[{e.evidence_id}] {e.data_type} | {e.geological_structure} | "
            f"OBS: {e.observation[:120]} | INTERP: {e.interpretation[:80]}"
            for e in evidence[-20:]   # last 20 to keep prompt manageable
        )
        existing_txt = ""
        if existing_hypotheses:
            existing_txt = "\n\nExisting hypotheses (refine or replace these):\n" + "\n".join(
                f"- [{h.hypothesis_id}] {h.statement}" for h in existing_hypotheses
            )

        prompt = (
            f"Scientific question: {question}\n"
            f"Study area: {study_area}\n\n"
            f"Evidence table:\n{ev_summary}"
            f"{existing_txt}\n\n"
            "Generate competing geological hypotheses (JSON array)."
        )
        messages = [
            {"role": "system", "content": _HYPOTHESIS_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._cfg, max_tokens=1000, temperature=0.4)
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return existing_hypotheses or []
            hyps_raw = json.loads(m.group(0))
        except Exception:
            return existing_hypotheses or []

        hypotheses = []
        for i, h in enumerate(hyps_raw):
            hid = f"H{i+1}"
            hypotheses.append(Hypothesis(
                hypothesis_id=hid,
                statement=h.get("statement", ""),
                supporting_evidence=h.get("supporting_evidence", []),
                contradicting_evidence=h.get("contradicting_evidence", []),
                data_types_needed=h.get("data_types_needed", []),
                confidence=h.get("confidence", "medium"),
            ))
        return hypotheses


# ---------------------------------------------------------------------------
# ── Component 6: GeologicalReasoner ────────────────────────────────────────
# ---------------------------------------------------------------------------

_REASONER_SYSTEM = """\
You are a geological reasoning expert in seismology, tectonics, fault mechanics,
and crustal structure.

Given hypotheses and an evidence table, evaluate each hypothesis rigorously:
- Identify which evidence is DIRECT (directly constrains the hypothesis),
  INDIRECT (consistent but not conclusive), SPECULATIVE, or MISSING.
- Flag hypotheses that overclaim beyond available data.
- Identify the preferred interpretation (most parsimonious, best-supported).
- List critical missing data that would resolve ambiguity.

Output JSON with this structure:
{
  "evaluations": [
    {
      "hypothesis_id": "H1",
      "direct_evidence": ["ev_id1"],
      "indirect_evidence": ["ev_id2"],
      "speculation": "what is speculative in this hypothesis",
      "missing_data": ["what data would confirm or refute this"],
      "assessment": "supported | weakly_supported | contradicted | insufficient_data",
      "note": "one-sentence evaluation"
    }
  ],
  "preferred_hypothesis": "H1",
  "preferred_rationale": "reason, citing evidence IDs",
  "missing_information": ["list of critically missing data"]
}
"""


class GeologicalReasoner:
    """Evaluates hypotheses against collected evidence."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = llm_cfg

    def evaluate(
        self,
        question: str,
        study_area: str,
        hypotheses: List[Hypothesis],
        evidence: List[Evidence],
    ) -> Tuple[List[Hypothesis], List[str], str, str]:
        """
        Returns (updated_hypotheses, missing_information, preferred_id, rationale).
        """
        if not hypotheses:
            return [], ["No hypotheses to evaluate"], "", ""

        ev_txt = "\n".join(
            f"[{e.evidence_id}] {e.data_type} | {e.geological_structure}: "
            f"{e.observation[:100]}"
            for e in evidence[-20:]
        )
        hyp_txt = "\n".join(
            f"[{h.hypothesis_id}] {h.statement} "
            f"(support: {h.supporting_evidence}, contra: {h.contradicting_evidence})"
            for h in hypotheses
        )
        prompt = (
            f"Question: {question}\nStudy area: {study_area}\n\n"
            f"Hypotheses:\n{hyp_txt}\n\n"
            f"Evidence:\n{ev_txt}\n\n"
            "Evaluate each hypothesis (JSON)."
        )
        messages = [
            {"role": "system", "content": _REASONER_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._cfg, max_tokens=1200, temperature=0.2)
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return hypotheses, [], "", ""
            d = json.loads(m.group(0))
        except Exception:
            return hypotheses, [], "", ""

        evals = {e["hypothesis_id"]: e for e in d.get("evaluations", [])}
        for h in hypotheses:
            ev = evals.get(h.hypothesis_id, {})
            assessment = ev.get("assessment", "insufficient_data")
            if assessment == "contradicted":
                h.status = "rejected"
            note = ev.get("note", "")
            if note:
                h.notes = note  # type: ignore[attr-defined]

        missing  = d.get("missing_information", [])
        pref_id  = d.get("preferred_hypothesis", "")
        pref_rat = d.get("preferred_rationale", "")
        return hypotheses, missing, pref_id, pref_rat


# ---------------------------------------------------------------------------
# ── Component 7: ValidationPlanner ─────────────────────────────────────────
# ---------------------------------------------------------------------------

_VALIDATION_SYSTEM = """\
You are a geoscience validation expert.  Given hypotheses and available evidence,
suggest concrete, executable validation checks that could confirm or refute each
hypothesis.

Focus on analyses possible with:
  - earthquake catalogs and hypocenter locations
  - focal mechanism solutions
  - seismic velocity tomography models
  - surface geological maps and fault traces
  - cross-sections and depth profiles
  - stress field data
  - user-provided CSV files, figures, or velocity models

Output a JSON array:
[
  {
    "linked_to": "H1",
    "description": "specific analysis to perform",
    "data_required": "what data or files are needed",
    "method": "how to perform the check (plot, statistical test, comparison, ...)",
    "expected_outcome": "what a positive or negative result would mean for the hypothesis"
  }
]
"""


class ValidationPlanner:
    """Designs concrete validation checks for each hypothesis."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg = llm_cfg

    def plan(
        self,
        hypotheses: List[Hypothesis],
        evidence: List[Evidence],
        study_area: str,
    ) -> List[ValidationCheck]:
        if not hypotheses:
            return []

        hyp_txt = "\n".join(
            f"[{h.hypothesis_id}] {h.statement} (confidence: {h.confidence})"
            for h in hypotheses if h.status == "active"
        )
        ev_types = list({e.data_type for e in evidence})

        prompt = (
            f"Study area: {study_area}\n\n"
            f"Active hypotheses:\n{hyp_txt}\n\n"
            f"Available data types in evidence: {', '.join(ev_types)}\n\n"
            "Suggest validation checks (JSON array)."
        )
        messages = [
            {"role": "system", "content": _VALIDATION_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = _llm_call(messages, self._cfg, max_tokens=900, temperature=0.3)
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return []
            checks_raw = json.loads(m.group(0))
        except Exception:
            return []

        checks = []
        for i, c in enumerate(checks_raw):
            checks.append(ValidationCheck(
                check_id=f"V{i+1}",
                linked_to=c.get("linked_to", ""),
                description=c.get("description", ""),
                data_required=c.get("data_required", ""),
                method=c.get("method", ""),
                expected_outcome=c.get("expected_outcome", ""),
            ))
        return checks


# ---------------------------------------------------------------------------
# ── Component 8: ReportWriter ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are a geoscience report writer.  Write in a clear, academic style appropriate
for seismology, tectonics, and crustal structure research.

RULES:
- Every geological claim must cite an evidence ID in brackets, e.g. [ev_abc123].
- Clearly separate observation from interpretation.
- Explicitly flag uncertainty and missing data.
- Do not overclaim — use hedging language where appropriate
  (e.g. "consistent with", "suggests", "may indicate").
- Structure the report with Markdown headings.
- Keep total length under 1200 words.
"""


class ReportWriter:
    """Maintains and updates the living interpretation report."""

    def __init__(self, llm_cfg: Dict[str, Any]) -> None:
        self._cfg    = llm_cfg
        self._report = ""

    def write(
        self,
        question: str,
        study_area: str,
        evidence: List[Evidence],
        hypotheses: List[Hypothesis],
        missing: List[str],
        validation: List[ValidationCheck],
        preferred_hyp_id: str,
        preferred_rationale: str,
        iteration: int,
    ) -> str:
        ev_txt = "\n".join(
            f"[{e.evidence_id}] {e.data_type} | {e.source}: "
            f"{e.observation[:120]} — {e.interpretation[:80]}"
            for e in evidence
        )
        hyp_txt = "\n".join(
            f"[{h.hypothesis_id}] ({h.confidence}) {h.statement} "
            f"[status: {h.status}]"
            for h in hypotheses
        )
        val_txt = "\n".join(
            f"[{v.check_id}→{v.linked_to}] {v.description}"
            for v in validation[:6]
        )
        missing_txt = "\n".join(f"- {m}" for m in missing[:8])
        pref_note = (
            f"Preferred hypothesis: {preferred_hyp_id} — {preferred_rationale}"
            if preferred_hyp_id else ""
        )

        prompt = (
            f"## Scientific question\n{question}\n\n"
            f"## Study area\n{study_area}\n\n"
            f"## Evidence table (iteration {iteration})\n{ev_txt}\n\n"
            f"## Hypotheses\n{hyp_txt}\n\n"
            f"{pref_note}\n\n"
            f"## Missing information\n{missing_txt or 'None identified'}\n\n"
            f"## Suggested validation\n{val_txt or 'None yet'}\n\n"
            "Write or update a structured interpretation report in Markdown."
            " Include: problem definition, literature synthesis, evidence summary,"
            " competing hypotheses, preferred interpretation, uncertainty,"
            " recommended analyses, and citations."
        )
        messages = [
            {"role": "system", "content": _REPORT_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        try:
            self._report = _llm_call(messages, self._cfg, max_tokens=1600, temperature=0.3)
        except Exception as exc:
            self._report = (
                f"# Interpretation Report (iteration {iteration})\n\n"
                f"*Report generation failed: {exc}*\n\n"
                f"## Evidence summary\n{self._build_fallback_summary(evidence)}"
            )
        return self._report

    def _build_fallback_summary(self, evidence: List[Evidence]) -> str:
        return "\n".join(
            f"- [{e.evidence_id}] {e.source}: {e.observation[:100]}"
            for e in evidence[:10]
        )

    @property
    def current_report(self) -> str:
        return self._report


# ---------------------------------------------------------------------------
# ── Component 9: LoopController ────────────────────────────────────────────
# ---------------------------------------------------------------------------

class LoopController:
    """
    Orchestrates the full literature-loop:

      for iteration in range(max_iterations):
          1. Refine retrieval queries
          2. Retrieve literature
          3. Read chunks → extract evidence
          4. Update evidence table
          5. Generate/update hypotheses
          6. Evaluate hypotheses
          7. Plan validation
          8. Update report
          9. Check convergence
    """

    def __init__(
        self,
        llm_cfg: Dict[str, Any],
        top_k: int = 8,
        score_threshold: float = 0.35,
    ) -> None:
        self._cfg       = llm_cfg
        self._planner   = TaskPlanner(llm_cfg)
        self._retriever = LiteratureRetriever(top_k=top_k, score_threshold=score_threshold)
        self._reader    = PaperReader(llm_cfg)
        self._ev_table  = EvidenceTableBuilder()
        self._hyp_gen   = HypothesisGenerator(llm_cfg)
        self._reasoner  = GeologicalReasoner(llm_cfg)
        self._val_plan  = ValidationPlanner(llm_cfg)
        self._reporter  = ReportWriter(llm_cfg)

    def run(
        self,
        question: str,
        study_area: str,
        max_iterations: int = 3,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> AgentResult:
        def _emit(phase: str, msg: str) -> None:
            if on_progress:
                on_progress({"phase": phase, "message": msg})

        _emit("planning", f"Decomposing question: {question[:80]}…")
        tasks = self._planner.plan(question, study_area)

        hypotheses:    List[Hypothesis]       = []
        missing_info:  List[str]              = []
        validation:    List[ValidationCheck]  = []
        sources_seen:  List[str]              = []
        preferred_id   = ""
        preferred_rat  = ""
        convergence    = "max_iterations_reached"

        for iteration in range(1, max_iterations + 1):
            _emit("retrieving", f"Iteration {iteration}/{max_iterations}: retrieving literature…")

            # 1. Build queries
            extra_queries = self._retriever.refine_query(question, self._ev_table.table, self._cfg)
            main_query    = f"{question} {study_area}"
            chunks        = self._retriever.retrieve(main_query, extra_queries)

            if not chunks and iteration == 1:
                _emit("warning", "No chunks retrieved from knowledge base.")

            # Track sources
            for c in chunks:
                if c.doc_name not in sources_seen:
                    sources_seen.append(c.doc_name)

            # 2. Read chunks
            _emit("reading", f"Reading {len(chunks)} document chunk(s)…")
            new_evidence = self._reader.read_batch(
                chunks, question, iteration=iteration, on_progress=on_progress
            )
            added = self._ev_table.add(new_evidence)

            _emit("evidence", f"Added {len(added)} new evidence record(s).")

            # 3. Generate / update hypotheses
            _emit("hypothesising", "Generating geological hypotheses…")
            hypotheses = self._hyp_gen.generate(
                question, study_area,
                self._ev_table.table,
                existing_hypotheses=hypotheses,
            )

            # 4. Evaluate hypotheses
            _emit("evaluating", "Evaluating hypotheses against evidence…")
            hypotheses, missing_info, preferred_id, preferred_rat = self._reasoner.evaluate(
                question, study_area, hypotheses, self._ev_table.table
            )

            # 5. Plan validation
            _emit("validating", "Planning validation checks…")
            validation = self._val_plan.plan(hypotheses, self._ev_table.table, study_area)

            # 6. Write / update report
            _emit("writing", f"Updating interpretation report (iteration {iteration})…")
            self._reporter.write(
                question, study_area,
                self._ev_table.table, hypotheses,
                missing_info, validation,
                preferred_id, preferred_rat,
                iteration,
            )

            # 7. Convergence check
            if len(added) == 0 and iteration > 1:
                convergence = "no_new_evidence"
                _emit("converged", f"Converged: no new evidence after iteration {iteration}.")
                break

            active_hyps = [h for h in hypotheses if h.status == "active"]
            if len(active_hyps) == 1 and active_hyps[0].confidence in ("high",) and iteration > 1:
                convergence = "hypothesis_convergence"
                _emit("converged", "Converged: single high-confidence hypothesis.")
                break

        _emit("done", "Literature loop complete.")

        # Build evidence table appendix in report
        final_report = (
            self._reporter.current_report
            + "\n\n---\n\n## Evidence Table\n\n"
            + self._ev_table.to_markdown()
        )

        return AgentResult(
            question=question,
            study_area=study_area,
            iterations_run=iteration,
            final_report=final_report,
            evidence_table=self._ev_table.table,
            hypotheses=hypotheses,
            missing_information=missing_info,
            suggested_validation=validation,
            retrieved_sources=sources_seen,
            convergence_reason=convergence,
        )


# ---------------------------------------------------------------------------
# ── Public facade ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class LiteratureLoopAgent:
    """
    Top-level facade.  Create once; call run() as many times as needed.

    Parameters
    ----------
    llm_cfg : dict, optional
        LLM configuration dict (provider, model, api_base, api_key).
        If omitted, reads from config_manager automatically.
    top_k : int
        Number of RAG chunks to retrieve per query (default 8).
    score_threshold : float
        Minimum cosine similarity for RAG retrieval (default 0.35).
    """

    def __init__(
        self,
        llm_cfg: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
        score_threshold: float = 0.35,
    ) -> None:
        self._cfg = llm_cfg or _get_llm_config()
        self._top_k     = top_k
        self._threshold = score_threshold

    def run(
        self,
        question: str,
        study_area: str = "",
        max_iterations: int = 3,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> AgentResult:
        """
        Run the iterative literature-loop agent.

        Parameters
        ----------
        question : str
            The scientific question to investigate.
        study_area : str
            Geographic / geological context (helps focus retrieval).
        max_iterations : int
            Maximum reasoning loops (default 3).
        on_progress : callable, optional
            Called with {"phase": str, "message": str} dicts.

        Returns
        -------
        AgentResult
        """
        ctrl = LoopController(self._cfg, self._top_k, self._threshold)
        return ctrl.run(question, study_area, max_iterations, on_progress)

    def result_to_dict(self, result: AgentResult) -> Dict[str, Any]:
        """Serialise an AgentResult to a JSON-safe dict."""
        return {
            "final_report":         result.final_report,
            "evidence_table":       [asdict(e) for e in result.evidence_table],
            "hypotheses":           [asdict(h) for h in result.hypotheses],
            "missing_information":  result.missing_information,
            "suggested_validation": [asdict(v) for v in result.suggested_validation],
            "retrieved_sources":    result.retrieved_sources,
            "iterations_run":       result.iterations_run,
            "convergence_reason":   result.convergence_reason,
        }
