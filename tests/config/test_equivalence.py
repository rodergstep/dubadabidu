# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proof obligations for moving the stages onto the typed config.

(a) round trip: for every shipped overlay combination, validating the merged
    dict and dumping it back (exclude_unset) gives the same JSON. pydantic must
    coerce nothing, because synth_hash JSON-encodes raw config values.
(b) inventory: scripts/scan_config_reads.py finds exactly the reads listed in
    reads_inventory.json. A new raw read cannot slip in unlisted; the typed
    migration shrinks the list to nothing.
(c) effective values: for every inventoried read and every combination, the
    value the code sees through the raw dict (with ITS default) equals what the
    typed model holds. Where they differ — or where a code default disagrees
    with the schema default — the row must be in drift.DRIFT with a decision.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from tests.config import locations
from tests.config.drift import DRIFT

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import scan_config_reads  # noqa: E402

ABSENT = object()
Raw = dict[str, object]


def _load(path: str) -> Raw:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def combos() -> dict[str, Raw]:
    base = _load(locations.config_path("config.yaml"))
    gpu = locations.deep_merge(base, _load(locations.config_path("config.gpu.yaml")))
    out: dict[str, Raw] = {"base": base, "base+gpu": gpu}
    for f in locations.overlay_files():
        name = Path(f).name
        if name not in ("config.yaml", "config.gpu.yaml"):
            out[f"base+gpu+{name}"] = locations.deep_merge(gpu, _load(f))
    out["base+source-ru"] = locations.deep_merge(
        base, _load(locations.config_path("config.source-ru.yaml")))
    edge = json.loads(json.dumps(base))
    edge["tts"]["engine"] = "edge"           # what `--engine edge` does
    out["base --engine edge"] = edge
    return out


@pytest.mark.parametrize("label", list(combos()))
def test_round_trip_is_exact(label: str) -> None:
    raw = combos()[label]
    dumped = locations.validate(raw).model_dump(exclude_unset=True)
    assert json.dumps(dumped, sort_keys=True, ensure_ascii=False) == \
        json.dumps(raw, sort_keys=True, ensure_ascii=False)


def test_inventory_is_current() -> None:
    want = {tuple(r) for r in json.loads(
        scan_config_reads.INVENTORY.read_text(encoding="utf-8"))}
    got = set(scan_config_reads.scan())
    assert got == want, (
        f"raw config reads changed — new: {sorted(got - want)[:8]}, "
        f"gone: {sorted(want - got)[:8]}. Refresh with "
        f"`python scripts/scan_config_reads.py --write` and, for a NEW read, "
        f"read it from the typed config instead.")


def _raw(raw: Raw, path: str, kind: str, default: str) -> object:
    node: object = raw
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            node = {}
    last = parts[-1]
    have = isinstance(node, dict) and last in node
    if kind == "[]":
        return node[last] if have else ABSENT
    if have:
        return node[last]
    if default == "<falsy>" or default.startswith("dynamic:"):
        return None
    return ast.literal_eval(default)


def _typed(model: BaseModel, path: str) -> object:
    node: object = model
    for p in path.split("."):
        if isinstance(node, BaseModel):
            node = getattr(node, p) if p in type(node).model_fields else ABSENT
        elif isinstance(node, dict):
            node = node.get(p, ABSENT)
        else:
            return ABSENT
        if node is None or node is ABSENT:
            return node
    # a subtree read compares as the dict the code would have been handed
    return node.model_dump(exclude_unset=True) if isinstance(node, BaseModel) else node


def _schema_default(path: str) -> object:
    model: type[BaseModel] | None = locations.Config
    parts = path.split(".")
    for i, p in enumerate(parts):
        if model is None or p not in model.model_fields:
            return ABSENT
        f = model.model_fields[p]
        if i == len(parts) - 1:
            if f.is_required():
                return ABSENT
            return f.default_factory() if f.default_factory else f.default
        model = next((a for a in getattr(f.annotation, "__args__", (f.annotation,))
                      if isinstance(a, type) and issubclass(a, BaseModel)), None)
    return ABSENT


def _same(kind: str, default: str, a: object, b: object) -> bool:
    if default == "<falsy>":
        return bool(None if a is ABSENT else a) == bool(None if b is ABSENT else b)
    if default.startswith("dynamic:") and a is None:
        return b is None
    return a == b and type(a) is type(b) or (a == b and {type(a), type(b)} <= {int, float})


def drift_rows() -> dict[str, list[str]]:
    rows = [tuple(r) for r in json.loads(
        scan_config_reads.INVENTORY.read_text(encoding="utf-8"))]
    models = {label: locations.validate(raw) for label, raw in combos().items()}
    out: dict[str, list[str]] = {}
    for path, kind, default in rows:
        if kind == "[]":
            continue
        key = f"{path} {kind} {default}"
        why: list[str] = []
        for label, raw in combos().items():
            a, b = _raw(raw, path, kind, default), _typed(models[label], path)
            if not _same(kind, default, a, b):
                why.append(f"{label}: code sees {a!r}, typed holds "
                           f"{'<absent>' if b is ABSENT else repr(b)}")
        sd = _schema_default(path)
        if sd is not ABSENT and not default.startswith("dynamic:"):
            cd = None if default == "<falsy>" else ast.literal_eval(default)
            if not _same(kind, default, cd, sd):
                why.append(f"latent: code default {cd!r}, schema default {sd!r}")
        if why:
            out[key] = why
    return out


def test_every_drift_is_in_the_ledger() -> None:
    found = drift_rows()
    new = sorted(set(found) - set(DRIFT))
    stale = sorted(set(DRIFT) - set(found))
    msg = []
    if new:
        msg.append("code/schema drift with no decision in tests/config/drift.py:\n"
                   + "\n".join(f"  {k}\n    " + "\n    ".join(found[k][:3]) for k in new))
    if stale:
        msg.append("drift.py lists rows that no longer drift (remove them):\n  "
                   + "\n  ".join(stale))
    assert not msg, "\n".join(msg)
