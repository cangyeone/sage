"""Built-in repository intelligence for SAGE's coding agent.

This module is independent of external coding agents. It borrows the useful
ideas from mature repo-aware coding tools: build a compact code map, rank files
by requested identifiers, and give the generator a disciplined editing protocol.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class RepoSymbol:
    rel_fname: str
    line: int
    name: str
    kind: str
    parent: str = ""


@dataclass
class RepoIntelligenceContext:
    available: bool
    repo_map: str = ""
    ranked_files: List[str] = field(default_factory=list)
    symbols: List[RepoSymbol] = field(default_factory=list)
    error: str = ""


_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "code", "file", "class",
    "function", "method", "route", "api", "where", "what", "which", "help",
    "实现", "修复", "代码", "文件", "函数", "类", "接口", "路由", "位置", "测试",
}


def request_hints(request: str) -> tuple[set[str], set[str]]:
    """Extract filename and identifier hints from a natural-language request."""
    text = request or ""
    mentioned_fnames = {
        token.strip("`'\" ")
        for token in re.findall(
            r"[\w./-]+\.(?:py|js|ts|tsx|html|css|md|json|toml|yml|yaml)",
            text,
        )
    }
    mentioned_idents = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
        if token.lower() not in _STOP_WORDS
    }
    for quoted in re.findall(r"`([^`]{2,80})`|['\"]([A-Za-z_][\w.:-]{2,80})['\"]", text):
        token = quoted[0] or quoted[1]
        if token and re.match(r"[A-Za-z_][A-Za-z0-9_.:-]*$", token):
            mentioned_idents.add(token.split(".")[-1])
    return mentioned_fnames, mentioned_idents


def _safe_text(path: Path, max_bytes: int = 260_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _python_symbols(rel_fname: str, text: str) -> List[RepoSymbol]:
    symbols: List[RepoSymbol] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(RepoSymbol(rel_fname, node.lineno, node.name, "class"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = ""
                symbols.append(RepoSymbol(rel_fname, node.lineno, node.name, "def", parent))
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    symbols.append(
                        RepoSymbol(rel_fname, node.lineno, alias.asname or alias.name, "import")
                    )

    # Regex pass keeps decorators/routes and nested methods easy to see.
    route_decorator = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        route = re.search(r"^\s*@(?:bp|app)\.route\((.+)\)", line)
        if route:
            route_decorator = route.group(1).strip()
            symbols.append(RepoSymbol(rel_fname, lineno, route_decorator, "route"))
            continue
        match = re.search(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if match and route_decorator:
            symbols.append(RepoSymbol(rel_fname, lineno, match.group(1), "route-handler"))
            route_decorator = ""
    return _dedupe_symbols(symbols)


def _generic_symbols(rel_fname: str, text: str) -> List[RepoSymbol]:
    patterns = [
        (r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", "class"),
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", "function"),
        (r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", "var"),
        (r"^\s*([A-Za-z_][A-Za-z0-9_-]+)\s*:\s*function\b", "function"),
        (r"\bid=[\"']([A-Za-z_][A-Za-z0-9_-]+)[\"']", "html-id"),
    ]
    symbols: List[RepoSymbol] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for pat, kind in patterns:
            match = re.search(pat, line)
            if match:
                symbols.append(RepoSymbol(rel_fname, lineno, match.group(1), kind))
    return _dedupe_symbols(symbols)


def _dedupe_symbols(symbols: Iterable[RepoSymbol]) -> List[RepoSymbol]:
    seen = set()
    out: List[RepoSymbol] = []
    for symbol in symbols:
        key = (symbol.rel_fname, symbol.line, symbol.name, symbol.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(symbol)
    return sorted(out, key=lambda s: (s.rel_fname, s.line, s.kind, s.name))


def collect_symbols(project_root: str | Path, rel_files: Sequence[str]) -> List[RepoSymbol]:
    root = Path(project_root)
    symbols: List[RepoSymbol] = []
    for rel in rel_files:
        path = root / rel
        if not path.is_file():
            continue
        text = _safe_text(path)
        if not text:
            continue
        if path.suffix == ".py":
            symbols.extend(_python_symbols(rel, text))
        elif path.suffix in {".js", ".ts", ".tsx", ".jsx", ".html", ".css"}:
            symbols.extend(_generic_symbols(rel, text))
    return _dedupe_symbols(symbols)


def _score_file(
    rel: str,
    request: str,
    symbols: Sequence[RepoSymbol],
    mentioned_fnames: set[str],
    mentioned_idents: set[str],
) -> int:
    text = (request or "").lower()
    lower_rel = rel.lower()
    score = 0
    if Path(rel).name in mentioned_fnames or rel in mentioned_fnames:
        score += 80
    if Path(rel).name.lower() in text:
        score += 30
    for part in lower_rel.replace("/", " ").replace("_", " ").replace("-", " ").split():
        if len(part) >= 3 and part in text:
            score += 6
    for symbol in symbols:
        name = symbol.name
        if name in mentioned_idents or name.lower() in text:
            score += 25 if symbol.kind in {"class", "def", "function", "route-handler"} else 10
        elif any(hint.lower() in name.lower() for hint in mentioned_idents):
            score += 8
    if rel.startswith("tests/") and re.search(r"test|测试|pytest|unit", request or "", re.I):
        score += 15
    return score


def rank_files(
    project_root: str | Path,
    request: str,
    rel_files: Sequence[str],
    symbols: Sequence[RepoSymbol] | None = None,
) -> List[str]:
    symbols = list(symbols if symbols is not None else collect_symbols(project_root, rel_files))
    by_file: dict[str, List[RepoSymbol]] = {}
    for symbol in symbols:
        by_file.setdefault(symbol.rel_fname, []).append(symbol)
    mentioned_fnames, mentioned_idents = request_hints(request)
    ranked = sorted(
        rel_files,
        key=lambda rel: _score_file(
            rel,
            request,
            by_file.get(rel, []),
            mentioned_fnames,
            mentioned_idents,
        ),
        reverse=True,
    )
    return [
        rel for rel in ranked
        if _score_file(rel, request, by_file.get(rel, []), mentioned_fnames, mentioned_idents) > 0
    ]


def render_repo_map(
    symbols: Sequence[RepoSymbol],
    ranked_files: Sequence[str],
    *,
    max_chars: int = 14000,
) -> str:
    by_file: dict[str, List[RepoSymbol]] = {}
    for symbol in symbols:
        by_file.setdefault(symbol.rel_fname, []).append(symbol)

    ordered_files = list(ranked_files)
    for rel in sorted(by_file):
        if rel not in ordered_files:
            ordered_files.append(rel)

    lines: List[str] = []
    for rel in ordered_files:
        file_symbols = by_file.get(rel, [])
        if not file_symbols:
            continue
        lines.append(f"{rel}:")
        for symbol in file_symbols[:40]:
            name = symbol.name
            if symbol.parent:
                name = f"{symbol.parent}.{name}"
            lines.append(f"  L{symbol.line}: {symbol.kind} {name}")
        if len(file_symbols) > 40:
            lines.append(f"  ... {len(file_symbols) - 40} more symbols")
        if sum(len(line) + 1 for line in lines) > max_chars:
            lines.append("... [SAGE repo map truncated]")
            break
    return "\n".join(lines)


def build_repo_intelligence(
    project_root: str | Path,
    request: str,
    rel_files: Sequence[str],
    *,
    max_chars: int = 14000,
) -> RepoIntelligenceContext:
    try:
        symbols = collect_symbols(project_root, rel_files)
        ranked_files = rank_files(project_root, request, rel_files, symbols)
        repo_map = render_repo_map(symbols, ranked_files, max_chars=max_chars)
        return RepoIntelligenceContext(
            available=True,
            repo_map=repo_map,
            ranked_files=ranked_files,
            symbols=symbols,
        )
    except Exception as exc:
        return RepoIntelligenceContext(available=False, error=str(exc))


SAGE_EDITING_GUIDE = """## SAGE Built-in Editing Discipline
- Treat the SAGE Repo Map as the first codebase map: use it to locate related classes, functions, routes, and tests before editing.
- Prefer small, exact edits to complete rewrites. For each changed file, preserve unrelated code and imports.
- For each replacement, identify the unique old block, the intended new block, and why that block is the right place to edit.
- Edit all files needed for the feature in one coherent pass, including focused tests for changed behavior.
- Before edits, print `[SAGE_AGENT] located <path>: <reason>` for each selected implementation/test file so the run shows code search and bug localization.
- After edits, print `[SAGE_CHANGED] <path>` for every changed file.
- After edits, run targeted validation (`py_compile`, changed/related pytest files, or the relevant API smoke test) and print `[SAGE_TEST]` lines.
- Python behavior changes must add/update focused tests, or at least locate and run existing focused tests that cover the changed function/API.
"""
