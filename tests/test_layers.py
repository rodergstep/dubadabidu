# SPDX-License-Identifier: AGPL-3.0-or-later
"""The layer contract (ARCHITECTURE_PLAN.md §2.3), enforced on the real graph.

    cli > workflows > {lab, infra} > review > stages > qc > engines > core

A module may import its own layer or any layer below it — never above. lab
and infra are siblings that must not import each other. Nothing in the
product may import research/. Every import counts, including the ones inside
functions: a lazy import is still a dependency.

ALLOWED lists today's known violations; each commit that removes one deletes
its line, and the set must end empty.
"""
from __future__ import annotations

import ast
from pathlib import Path
from tests.paths import REPO

RANK = {"core": 0, "engines": 1, "qc": 2, "stages": 3, "review": 4,
        "lab": 5, "infra": 5, "workflows": 6, "cli": 7}
SIBLINGS_APART = {frozenset({"lab", "infra"})}

LAYER_OF = {
    "dub": "cli",
    "pipeline.manifest": "core", "pipeline.text_norm": "core",
    "pipeline.logic": "core", "pipeline.device": "core",
    "pipeline.runlog": "core", "pipeline.config_schema": "core",
    "pipeline.audio": "core",
    "pipeline.tts_engine": "engines", "pipeline.asr": "engines",
    "qc.metrics": "qc", "qc.backcheck": "qc", "qc.evaluate": "qc",
    "qc.stress_words": "qc", "qc.selection": "qc",
    **{f"pipeline.{s}": "stages" for s in (
        "s1_extract", "s2_transcribe", "s3_translate", "s4_synthesize",
        "s5_fit", "s6_mix", "s7_subtitles", "s8_mux")},
    "qc.review_page": "review", "qc.verdicts": "review", "qc.refit": "review",
    "qc.batch_report": "review", "qc.blind": "review", "qc.compare": "review",
    "qc.listen": "review",
    "pipeline.prep": "lab", "pipeline.tune": "lab", "qc.bakeoff": "lab",
    "qc.tune_page": "lab", "pipeline.experiments": "lab",
    "pipeline.runpod_infra": "infra",
    "pipeline.autopilot": "workflows", "pipeline.course": "workflows",
}
RESEARCH_DIR = "research"
ALLOWED: set[tuple[str, str]] = set()


def modules() -> dict[str, Path]:
    out = {"dub": REPO / "dub.py"} if (REPO / "dub.py").exists() else {}
    for pkg in ("pipeline", "qc", RESEARCH_DIR):
        for p in sorted((REPO / pkg).glob("*.py")):
            if p.stem != "__init__":
                out[f"{pkg}.{p.stem}"] = p
    return out


def imports(name: str, path: Path, known: set[str]) -> set[str]:
    """Every internal module `name` imports, wherever the statement sits."""
    pkg = name.rsplit(".", 1)[0] if "." in name else ""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            cands = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                base = f"{parent}.{base}" if base else parent
            cands = [f"{base}.{a.name}" for a in node.names] + [base]
        else:
            continue
        out.update(c for c in cands if c in known and c != name)
    return out


def edges() -> set[tuple[str, str]]:
    mods = modules()
    known = set(mods)
    return {(a, b) for a, p in mods.items() for b in imports(a, p, known)}


def layer(m: str) -> str:
    if m.startswith(RESEARCH_DIR + "."):
        return "research"
    assert m in LAYER_OF, f"{m} has no layer: add it to LAYER_OF in tests/test_layers.py"
    return LAYER_OF[m]


def violations() -> set[tuple[str, str]]:
    bad = set()
    for a, b in edges():
        la, lb = layer(a), layer(b)
        if la == "research":
            continue
        if lb == "research" or RANK[lb] > RANK[la] \
                or (la != lb and frozenset({la, lb}) in SIBLINGS_APART):
            bad.add((a, b))
    return bad


def test_every_module_has_a_layer() -> None:
    for m in modules():
        layer(m)


def test_no_upward_or_research_imports() -> None:
    bad = violations()
    new = sorted(bad - ALLOWED)
    assert not new, ("layer violations (an import pointing up the stack, or "
                     "into research/):\n  " + "\n  ".join(f"{a} -> {b} "
                     f"({layer(a)} -> {layer(b)})" for a, b in new))


def test_allowlist_has_no_stale_entries() -> None:
    stale = sorted(ALLOWED - violations())
    assert not stale, f"fixed — remove from ALLOWED: {stale}"
