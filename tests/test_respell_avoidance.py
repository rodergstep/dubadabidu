"""The two untested ru-stress remediations: reduction respelling and avoidance.

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

from pipeline import text_norm as TN  # noqa: E402
from pipeline.manifest import synth_hash  # noqa: E402


# --- respelling ---------------------------------------------------------------

def test_unstressed_o_and_e_reduce_but_the_stressed_vowel_survives():
    """Akanye/ikanye: the stressed vowel keeps full quality, so respelling
    leaves it as the only unreduced one. qc/stress_detect measured that this
    contrast is real in the audio (stressed о -> [o] 89% vs 26%)."""
    assert TN.respell_word_ru("молоко") == "малако"
    assert TN.respell_word_ru("натюрморте") == "натюрморти"
    assert TN.respell_word_ru("белилам") == "билилам"


def test_a_word_with_nothing_to_reduce_is_untouched():
    assert TN.respell_word_ru("хватит") == "хватит"


def test_an_unknown_stress_leaves_the_word_alone(monkeypatch):
    """Guessing would encode a WRONG stress, which is worse than encoding none
    — the whole route exists because the engine already guesses badly."""
    monkeypatch.setattr(TN, "accent_ru_plus", lambda t: None)
    assert TN.respell_word_ru("молоко") == "молоко"


def test_only_marked_words_are_respelled():
    """Targeted, not global. Respelling every word would hand the engine a page
    of misspelled Russian to test one hypothesis."""
    text = "я выдавливаю краску в натюрморте"
    out = TN.respell_ru(text, {"натюрморте"})
    assert "натюрморти" in out
    assert "выдавливаю" in out, "an unmarked word must not be touched"


def test_an_empty_lexicon_makes_respelling_a_no_op():
    assert TN.respell_ru("в натюрморте", set()) == "в натюрморте"


def test_respell_and_stress_marking_are_mutually_exclusive():
    """One marks the stressed vowel with a diacritic (refuted, 2.1b), the other
    respells the unstressed ones. Both at once tests neither."""
    both = {"ru_stress": True, "ru_respell": True}
    out = TN.normalize_for_tts("натюрморте", "ru", both)
    assert "́" in out, "ru_stress should win when both are set"
    assert "натюрморти" not in out


def test_respelling_salts_the_cache_key():
    """Without this an A/B would score the respelled config against
    un-respelled cached audio."""
    base = {"reference_wav": "r.wav", "engine": "qwen"}
    assert synth_hash("натюрморте", "ru", base) != \
        synth_hash("натюрморте", "ru", {**base, "ru_respell": True})


def test_growing_the_lexicon_invalidates_the_cache(tmp_path, monkeypatch):
    """The lexicon is an INPUT: adding a word changes what gets respelled, so a
    grown table must not silently reuse takes made from the smaller one."""
    monkeypatch.chdir(tmp_path)
    base = {"reference_wav": "r.wav", "engine": "qwen", "ru_respell": True}
    Path("stress_lexicon_ru.json").write_text(json.dumps(
        {"охра": {"marked": 1, "forms": ["охра"], "occurrences": []}}))
    one = synth_hash("охра", "ru", base)
    Path("stress_lexicon_ru.json").write_text(json.dumps(
        {"охра": {"marked": 1, "forms": ["охра"], "occurrences": []},
         "белилам": {"marked": 1, "forms": ["белилам"], "occurrences": []}}))
    assert synth_hash("охра", "ru", base) != one


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
    ("config.exp.ru-respell.yaml", ("tts", "ru_respell")),
    ("config.exp.ru-avoided.yaml", ("bakeoff", "text_variant")),
])
def test_each_overlay_isolates_exactly_one_axis(path, axis):
    """Its control is config.exp.ru-control.yaml. If an overlay moves anything
    besides its axis and the label, the A/B measures a mixture."""
    c = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    sec, key = axis
    assert c[sec][key], f"{path} does not set its own axis"
    assert c["tts"].get("variant_label"), "rows would merge without a label"
    allowed = {"variant_label", key, "ru_stress"}
    assert set(c["tts"]) <= allowed, f"{path} tts moves more than its axis"
