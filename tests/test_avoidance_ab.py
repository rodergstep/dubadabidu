"""The avoidance A/B — the ru-stress remediation that is still open.

Both are gates, not features. FINDINGS 2.1 closed detection and selection; what
is left is per-word control of the INPUT, and neither route has been measured.
These tests pin the mechanics so the pod run measures the axis it claims to.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



# --- avoidance A/B ------------------------------------------------------------

def test_affected_uses_whole_word_matching():
    """`цветами` contains `цвета` as a substring — the first pass of this
    analysis reported exactly that false positive."""
    from qc.avoidance_ab import affected
    man = {"utterances": [
        {"id": "u1", "tr": {"ru": {"text": "холодными цветами и тёплыми"}}},
        {"id": "u2", "tr": {"ru": {"text": "самые светлые цвета"}}}]}
    assert [u["id"] for u in affected(man, "ru", ["цвета"])] == ["u2"]


def test_the_text_variant_key_is_scoped_to_its_only_consumer():
    """`bakeoff.text_variant`, never `tts.text_variant`. A tts.* key would look
    like it changed what production synthesizes while only the bake-off read it
    — the shape of qc.eval.weights ranking nothing for weeks."""
    src = (ROOT / "qc" / "bakeoff.py").read_text(encoding="utf-8")
    assert 'get("bakeoff", {}) or {}).get("text_variant")' in src
    for f in ("pipeline/s4_synthesize.py", "pipeline/s5_fit.py",
              "pipeline/tts_engine.py"):
        assert "text_variant" not in (ROOT / f).read_text(encoding="utf-8"), \
            f"{f} reads text_variant — then the key has two consumers again"


def test_a_segment_without_an_alternative_falls_back_to_shipped():
    src = (ROOT / "qc" / "bakeoff.py").read_text(encoding="utf-8")
    block = src[src.index("bakeoff.text_variant selects"):]
    block = block[:block.index("\n\n")]
    assert 'or u["tr"][lang].get("fitted_text")' in block, \
        "an A/B must cover only the segments that actually differ"


# --- the experiment overlays --------------------------------------------------

@pytest.mark.parametrize("path,axis", [
    ("config.exp.ru-avoided.yaml", ("bakeoff", "text_variant")),
])
def test_each_overlay_isolates_exactly_one_axis(path, axis):
    """Its control is config.exp.ru-control.yaml. If an overlay moves anything
    besides its axis and the label, the A/B measures a mixture."""
    c = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    sec, key = axis
    assert c[sec][key], f"{path} does not set its own axis"
    assert c["tts"].get("variant_label"), "rows would merge without a label"
    allowed = {"variant_label", key}
    assert set(c["tts"]) <= allowed, f"{path} tts moves more than its axis"


def test_take_offset_selects_unheard_takes():
    """qwen+fast+k5 holds five takes per segment and the first round used 0-2.
    Without an offset a second page re-rates the same audio: more rows, no more
    evidence, and the pair is already in comparisons_<lang>.json."""
    import inspect
    from qc.compare import _groups, build
    assert "take_offset" in inspect.signature(_groups).parameters
    assert "take_offset" in inspect.signature(build).parameters
    src = inspect.getsource(_groups)
    assert "ts[take_offset:take_offset +" in src, \
        "the window must start at the offset, not at zero"
