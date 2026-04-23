"""
paper_reader.py — 地震学文献读取与结构化解析

支持输入
--------
1. 本地 PDF 文件
2. arXiv 论文（arXiv ID 或完整 URL）
3. DOI 链接（通过 Sci-Hub 镜像或 Unpaywall）
4. 直接粘贴的文本/摘要

输出
----
Paper 对象，包含：
  - 标题、作者、年份、期刊
  - 按章节分割的正文（Abstract / Introduction / Methods / Results / ...）
  - 提取的公式、表格、算法描述
  - 原始全文字符串
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PaperSection:
    title: str           # 节标题（如 "2. Methods"）
    content: str         # 正文内容
    level: int = 1       # 标题层级（1=章, 2=节, 3=子节）


@dataclass
class Paper:
    """结构化的科学论文对象。"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    journal: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    sections: List[PaperSection] = field(default_factory=list)
    full_text: str = ""
    source: str = ""        # 来源描述（文件路径/URL）
    keywords: List[str] = field(default_factory=list)

    # ---- computed ----
    def get_methods_text(self) -> str:
        """提取方法章节文本（模糊匹配）。"""
        method_kw = {"method", "methodology", "approach", "algorithm",
                     "technique", "procedure", "formulation", "theory",
                     "数据处理", "方法", "理论", "算法"}
        texts = []
        for sec in self.sections:
            t = sec.title.lower()
            if any(k in t for k in method_kw):
                texts.append(f"[{sec.title}]\n{sec.content}")
        return "\n\n".join(texts) if texts else ""

    def get_key_content(self, max_chars: int = 8000) -> str:
        """
        返回对 LLM 最有价值的内容（摘要 + 方法 + 结论），
        控制在 max_chars 以内。
        """
        parts = []
        if self.abstract:
            parts.append(f"[Abstract]\n{self.abstract}")
        methods = self.get_methods_text()
        if methods:
            parts.append(methods)
        # Conclusion / Discussion
        for sec in self.sections:
            t = sec.title.lower()
            if any(k in t for k in {"conclusion", "discussion", "result", "summary", "结论", "讨论", "结果"}):
                parts.append(f"[{sec.title}]\n{sec.content[:1000]}")
        combined = "\n\n".join(parts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...[truncated]"
        return combined

    def summary(self) -> str:
        lines = [f"📄 {self.title or '(无标题)'}"]
        if self.authors:
            lines.append(f"   作者: {', '.join(self.authors[:4])}"
                         + (" et al." if len(self.authors) > 4 else ""))
        if self.year:
            lines.append(f"   年份: {self.year}")
        if self.journal:
            lines.append(f"   期刊: {self.journal}")
        if self.doi:
            lines.append(f"   DOI: {self.doi}")
        if self.arxiv_id:
            lines.append(f"   arXiv: {self.arxiv_id}")
        lines.append(f"   章节: {len(self.sections)} 个")
        lines.append(f"   全文: {len(self.full_text)} 字符")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF reader
# ---------------------------------------------------------------------------

def _read_pdf_text(path: str) -> str:
    """Extract raw text from a PDF using pdfminer (preferred) or PyMuPDF."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path)
        if text and len(text.strip()) > 100:
            return text
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except ImportError:
        pass
    except Exception:
        pass

    raise ImportError(
        "无法读取 PDF：请安装 pdfminer.six (pip install pdfminer.six) "
        "或 PyMuPDF (pip install pymupdf)"
    )


def _split_sections(text: str) -> List[PaperSection]:
    """
    Heuristically split paper text into sections by detecting headings.
    Works for most two-column journal PDFs and preprints.
    """
    # Common section heading patterns
    heading_pattern = re.compile(
        r'^(?:'
        r'\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z ]{3,60}'  # "1. Introduction" / "2.1 Methods"
        r'|[A-Z][A-Z ]{4,40}$'                         # "ABSTRACT", "METHODS"
        r')',
        re.MULTILINE
    )

    lines = text.splitlines()
    sections: List[PaperSection] = []
    current_title = "Preamble"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if heading_pattern.match(stripped) and len(stripped) < 80:
            # Save previous section
            content = "\n".join(current_lines).strip()
            if content:
                level = 1 if not re.match(r'\d+\.\d+', stripped) else 2
                sections.append(PaperSection(
                    title=current_title,
                    content=content,
                    level=level
                ))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append(PaperSection(title=current_title, content=content))

    return sections if sections else [PaperSection(title="Full Text", content=text)]


def _extract_metadata(text: str) -> Dict[str, str]:
    """Extract title, authors, year, DOI from the first ~2000 chars of paper text."""
    head = text[:2000]
    meta = {}

    # Year
    year_m = re.search(r'\b(19|20)\d{2}\b', head)
    if year_m:
        meta["year"] = year_m.group()

    # DOI
    doi_m = re.search(r'(?:doi\.org/|DOI:\s*)(10\.\d{4,}/\S+)', head, re.IGNORECASE)
    if doi_m:
        meta["doi"] = doi_m.group(1).rstrip(".,;")

    # arXiv
    ax_m = re.search(r'arXiv[:\s]+(\d{4}\.\d{4,5})', head, re.IGNORECASE)
    if ax_m:
        meta["arxiv_id"] = ax_m.group(1)

    return meta


def read_pdf(path: str) -> Paper:
    """
    Read a local PDF file and return a structured Paper object.

    Parameters
    ----------
    path : str  Absolute or relative path to the PDF.

    Returns
    -------
    Paper
    """
    path = str(Path(path).resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    text = _read_pdf_text(path)
    if not text or len(text.strip()) < 50:
        raise ValueError(f"无法从 PDF 提取文本（可能是扫描版）: {path}")

    # Clean up hyphenated line breaks common in two-column PDFs
    text = re.sub(r'-\n(\S)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    sections = _split_sections(text)
    meta = _extract_metadata(text)

    # Extract abstract
    abstract = ""
    for sec in sections:
        if "abstract" in sec.title.lower() or "摘要" in sec.title:
            abstract = sec.content[:1500]
            break
    if not abstract:
        # Take first 800 chars of body as abstract proxy
        abstract = text[:800]

    title = Path(path).stem  # Fallback: use filename as title

    return Paper(
        title=title,
        year=meta.get("year", ""),
        doi=meta.get("doi", ""),
        arxiv_id=meta.get("arxiv_id", ""),
        abstract=abstract,
        sections=sections,
        full_text=text,
        source=path,
    )


# ---------------------------------------------------------------------------
# arXiv fetcher
# ---------------------------------------------------------------------------

def _arxiv_id_from_url(url_or_id: str) -> str:
    """Extract clean arXiv ID from URL or bare ID string."""
    m = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', url_or_id)
    return m.group(1) if m else url_or_id.strip()


def fetch_arxiv(arxiv_id_or_url: str) -> Paper:
    """
    Download a paper from arXiv (abstract page + PDF).

    Parameters
    ----------
    arxiv_id_or_url : str
        arXiv ID (e.g., "2104.12345") or full URL.

    Returns
    -------
    Paper
    """
    arxiv_id = _arxiv_id_from_url(arxiv_id_or_url)

    # 1. Fetch abstract/metadata via arXiv API
    api_url = f"https://export.arxiv.org/abs/{arxiv_id}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "SAGE-SeismoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise ConnectionError(f"无法访问 arXiv ({api_url}): {e}")

    # Parse metadata from HTML
    title_m = re.search(r'<h1[^>]*class="title[^"]*"[^>]*>(.*?)</h1>', html, re.DOTALL)
    title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else arxiv_id

    abstract_m = re.search(r'<blockquote[^>]*class="abstract[^"]*"[^>]*>(.*?)</blockquote>', html, re.DOTALL)
    abstract = re.sub(r'<[^>]+>', '', abstract_m.group(1)).strip() if abstract_m else ""
    abstract = re.sub(r'^Abstract:\s*', '', abstract)

    authors_m = re.findall(r'<a href="/search/[^"]+">([^<]+)</a>', html)
    authors = [a.strip() for a in authors_m[:10]]

    year_m = re.search(r'Submitted.*?(\d{4})', html)
    year = year_m.group(1) if year_m else ""

    # 2. Download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    import tempfile
    tmp_pdf = tempfile.mktemp(suffix=".pdf")
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "SAGE-SeismoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(tmp_pdf, 'wb') as f:
                f.write(resp.read())
        text = _read_pdf_text(tmp_pdf)
        sections = _split_sections(text)
    except Exception:
        # Fall back to abstract only
        text = abstract
        sections = [PaperSection(title="Abstract", content=abstract)]
    finally:
        try:
            os.unlink(tmp_pdf)
        except Exception:
            pass

    return Paper(
        title=title,
        authors=authors,
        year=year,
        arxiv_id=arxiv_id,
        abstract=abstract,
        sections=sections,
        full_text=text,
        source=f"arXiv:{arxiv_id}",
    )


# ---------------------------------------------------------------------------
# DOI / URL fetcher
# ---------------------------------------------------------------------------

def fetch_doi(doi_or_url: str) -> Paper:
    """
    Attempt to fetch a paper by DOI.
    Tries: Unpaywall → Sci-Hub (fallback) → metadata-only from CrossRef.

    Parameters
    ----------
    doi_or_url : str
        DOI (e.g., "10.1029/2022JB024987") or https://doi.org/... URL.

    Returns
    -------
    Paper
    """
    doi = doi_or_url
    if doi.startswith("http"):
        m = re.search(r'10\.\d{4,}/\S+', doi)
        doi = m.group(0).rstrip(".,;") if m else doi

    # --- Metadata via CrossRef ---
    cr_url = f"https://api.crossref.org/works/{doi}"
    title, authors, year, journal = "", [], "", ""
    try:
        req = urllib.request.Request(cr_url, headers={"User-Agent": "SAGE-SeismoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            cr = json.loads(resp.read().decode("utf-8"))
        msg = cr.get("message", {})
        titles = msg.get("title", [])
        title = titles[0] if titles else ""
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                   for a in msg.get("author", [])[:8]]
        published = msg.get("published", {}).get("date-parts", [[""]])[0]
        year = str(published[0]) if published else ""
        containers = msg.get("container-title", [])
        journal = containers[0] if containers else ""
    except Exception:
        pass

    # --- Try Unpaywall for open-access PDF ---
    email = "seismo@sage.tool"
    unp_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    pdf_url = None
    try:
        with urllib.request.urlopen(unp_url, timeout=10) as resp:
            unp = json.loads(resp.read().decode("utf-8"))
        loc = unp.get("best_oa_location") or {}
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        pass

    sections = []
    full_text = ""
    abstract = ""

    if pdf_url:
        import tempfile
        tmp = tempfile.mktemp(suffix=".pdf")
        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "SAGE-SeismoAgent/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(tmp, 'wb') as f:
                    f.write(resp.read())
            full_text = _read_pdf_text(tmp)
            sections = _split_sections(full_text)
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    if not sections:
        abstract = f"[仅元数据] {title}. {journal} ({year}). DOI: {doi}"
        sections = [PaperSection(title="Metadata", content=abstract)]

    return Paper(
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
        abstract=abstract,
        sections=sections,
        full_text=full_text,
        source=f"DOI:{doi}",
    )


# ---------------------------------------------------------------------------
# Plain text / abstract
# ---------------------------------------------------------------------------

def read_text(text: str, title: str = "Pasted Text") -> Paper:
    """
    Wrap a raw text string (pasted abstract or paper excerpt) as a Paper.

    Parameters
    ----------
    text : str
    title : str  Optional title label.

    Returns
    -------
    Paper
    """
    abstract = text[:600] if len(text) > 600 else text
    sections = _split_sections(text)
    meta = _extract_metadata(text)
    return Paper(
        title=title,
        year=meta.get("year", ""),
        doi=meta.get("doi", ""),
        arxiv_id=meta.get("arxiv_id", ""),
        abstract=abstract,
        sections=sections,
        full_text=text,
        source="pasted_text",
    )


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_paper(source: str, title: str = "") -> Paper:
    """
    Universal paper loader — auto-detects source type.

    Parameters
    ----------
    source : str
        One of:
        - Local PDF path  (/path/to/paper.pdf)
        - arXiv ID        (2104.12345 or arxiv:2104.12345)
        - arXiv URL       (https://arxiv.org/abs/2104.12345)
        - DOI             (10.1029/2022JB024987)
        - DOI URL         (https://doi.org/10.1029/...)
        - Raw text        (long string > 50 chars with no path-like pattern)
    title : str  Optional title override.

    Returns
    -------
    Paper
    """
    src = source.strip()

    # Local PDF
    if src.endswith(".pdf") or (os.path.exists(src) and src.endswith(".pdf")):
        paper = read_pdf(src)
        if title:
            paper.title = title
        return paper

    # arXiv
    if re.match(r'^\d{4}\.\d{4,}', src) or "arxiv.org" in src or src.lower().startswith("arxiv:"):
        return fetch_arxiv(src)

    # DOI
    if re.match(r'^10\.\d{4,}/', src) or "doi.org" in src:
        return fetch_doi(src)

    # URL that might be a PDF
    if src.startswith("http"):
        if src.endswith(".pdf"):
            import tempfile
            tmp = tempfile.mktemp(suffix=".pdf")
            try:
                urllib.request.urlretrieve(src, tmp)
                paper = read_pdf(tmp)
                if title:
                    paper.title = title
                return paper
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
        # Try DOI URL
        if "doi.org" in src:
            return fetch_doi(src)

    # Fallback: treat as plain text
    return read_text(src, title=title or "Text Input")


# ---------------------------------------------------------------------------
# Paper store (in-memory library)
# ---------------------------------------------------------------------------

class PaperStore:
    """In-memory collection of loaded papers."""

    def __init__(self):
        self._papers: Dict[str, Paper] = {}  # key → Paper

    def add(self, paper: Paper, key: Optional[str] = None) -> str:
        """Add a paper. Returns the key used."""
        if key is None:
            key = f"paper_{len(self._papers) + 1}"
        self._papers[key] = paper
        return key

    def get(self, key: str) -> Optional[Paper]:
        return self._papers.get(key)

    def list(self) -> List[Tuple[str, Paper]]:
        return list(self._papers.items())

    def __len__(self):
        return len(self._papers)

    def combined_context(self, max_chars_per_paper: int = 5000) -> str:
        """Return combined key content from all papers for LLM context."""
        parts = []
        for key, paper in self._papers.items():
            parts.append(
                f"=== [{key}] {paper.title} ({paper.year}) ===\n"
                + paper.get_key_content(max_chars_per_paper)
            )
        return "\n\n".join(parts)

    def search(self, query: str, max_chars: int = 2000) -> str:
        """
        Simple keyword search across all papers.
        Returns relevant excerpts (up to max_chars).
        """
        query_words = set(query.lower().split())
        results = []
        for key, paper in self._papers.items():
            for sec in paper.sections:
                text = sec.content.lower()
                hits = sum(1 for w in query_words if w in text)
                if hits >= max(1, len(query_words) // 2):
                    excerpt = sec.content[:500]
                    results.append((hits, f"[{key} / {sec.title}]\n{excerpt}"))
        results.sort(reverse=True)
        combined = "\n\n".join(r for _, r in results[:5])
        return combined[:max_chars] if combined else "（未找到相关段落）"
