#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inventory every raw read of the merged config, with the default it assumes.

    python scripts/scan_config_reads.py          # print the inventory
    python scripts/scan_config_reads.py --write  # refresh tests/config/reads_inventory.json

A read is `cfg[...]`, `.get(key)` or `.get(key, default)` on an expression the
scanner can resolve to a dotted config path: the whole config (`cfg`), an alias
assigned from one (`tts = cfg["tts"]`, `rp = {**DEFAULTS, **cfg["runpod"]}`),
or a parameter that by convention holds a subtree (PARAMS below). Module-level
DEFAULTS dicts merged under a block are reported as kind "defaults".

Rows are (path, kind, default) with no file or line, so moving code does not
change the inventory; only adding or removing a read does. The typed-config
migration shrinks it to nothing (ARCHITECTURE_PLAN.md §3.2).
"""
from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "tests" / "config" / "reads_inventory.json"
ROOTS = frozenset({"cfg", "config", "base_cfg"})
# parameter / local names that hold a config SUBTREE, per module basename
PARAMS: dict[str, dict[str, str]] = {
    "tts_engine.py": {"t": "tts", "tts_cfg": "tts", "tu": "tts"},
    "selection.py": {"t": "tts", "tts_cfg": "tts"},
    "manifest.py": {"tts_cfg": "tts"},
    "text_norm.py": {"cfg": "tts"},
    "asr.py": {"a": "asr"},
    "s2_transcribe.py": {"a": "asr"},
    "s3_translate.py": {"tcfg": "translation"},
    "s5_fit.py": {"tu": "tts"},
    "s6_mix.py": {"mix": "mix", "norm": "mix.segment_norm"},
    "s1_extract.py": {"sep": "separation", "sep_cfg": "separation"},
    "runpod_infra.py": {"rp": "runpod"},
    "prep.py": {"p": "prep"},
    "tune.py": {"tn": "tune"},
    "bakeoff.py": {"t": "tts", "tcfg": "bakeoff.tune", "bcfg": "bakeoff"},
    "dub.py": {"tn": "tune", "sep": "separation", "tcfg": "translation"},
    "backcheck.py": {},
    "evaluate.py": {"ecfg": "qc.eval"},
    "refit.py": {"rcfg": "qc.refit"},
}
Row = tuple[str, str, str]


def sources() -> list[Path]:
    root = [REPO / "dub.py"] if (REPO / "dub.py").exists() else []
    return root + sorted((REPO / "src" / "dubadabidu").rglob("*.py"))


def _chain(node: ast.AST, env: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        return "" if node.id in ROOTS else None
    if isinstance(node, ast.Subscript):
        base = _chain(node.value, env)
        k = node.slice
        if base is None or not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            return None
        return f"{base}.{k.value}" if base else k.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        base = _chain(node.func.value, env)
        k = node.args[0]
        if base is None or not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            return None
        return f"{base}.{k.value}" if base else k.value
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _chain(node.values[0], env)
    if isinstance(node, ast.Dict):
        # {**cfg, "tts": {...}} is still the whole config; {**DEFAULTS,
        # **cfg["runpod"]} is the runpod block: follow the first ** spread
        # that resolves (key None marks a spread)
        for k, v in zip(node.keys, node.values, strict=True):
            c = _chain(v, env) if k is None else None
            if c is not None:
                return c
    return None


def _default(node: ast.AST) -> str:
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, SyntaxError):
        return "dynamic:" + ast.unparse(node)


def _rows(path: Path) -> Iterator[Row]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    params = PARAMS.get(path.name, {})

    found: list[Row] = []

    def visit(node: ast.AST, env: dict[str, str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env = {**env}
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            c = _chain(node.value, env)
            if c:
                env[node.targets[0].id] = c
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            base = _chain(node.func.value, env)
            if base is not None:
                p = f"{base}.{node.args[0].value}" if base else node.args[0].value
                d = _default(node.args[1]) if len(node.args) > 1 else "<falsy>"
                found.append((p, "get", d))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            base = _chain(node.value, env)
            k = node.slice
            if base is not None and isinstance(k, ast.Constant) and isinstance(k.value, str):
                found.append((f"{base}.{k.value}" if base else k.value, "[]", ""))
        for child in ast.iter_child_nodes(node):
            visit(child, env)

    visit(tree, dict(params))
    yield from found


def module_defaults() -> list[Row]:
    """Module DEFAULTS dicts merged under a config block."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    from tests.config import locations
    out: list[Row] = []
    for block, table in locations.defaults_tables().items():
        out += [(f"{block}.{k}", "defaults", repr(v)) for k, v in table.items()]
    return out


def scan() -> list[Row]:
    rows: set[Row] = set(module_defaults())
    for p in sources():
        rows.update(_rows(p))
    return sorted(rows)


def main() -> None:
    rows = scan()
    if "--write" in sys.argv:
        INVENTORY.write_text(json.dumps([list(r) for r in rows], ensure_ascii=False,
                                        indent=0) + "\n", encoding="utf-8")
        print(f"{len(rows)} rows -> {INVENTORY.relative_to(REPO)}")
        return
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    main()
