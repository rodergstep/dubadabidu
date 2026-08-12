"""s5's emergency-shorten rescue must not fire on a slot it cannot place into.

choose_placement returns "no" for slot <= 0 BEFORE it inspects a single
duration (logic.py: `if slot <= 0: return ..., "no", max_tempo`). s5 read that
"no" as "every variant overflowed" and ran the rescue anyway:

    budget = int(len(candidates[k]) * slot * max_tempo / durs[k] * 0.95)

with slot == 0 that is a budget of ZERO characters. So it billed a DeepSeek call
asking for rewrites of at most 0 chars, synthesized whatever came back — a full
best_of unit per variant, on a rented GPU — and fed it to a function that
returns "no" for slot <= 0 no matter what the durations are. Both costs, every
run, for audio that cannot be placed by construction.

Zero-slot segments are not exotic: the LAST utterance is capped at
hard_end=man["duration"] (retime_step), so any final dub that runs to the end of
the video lands here, on every video, in every language.

The overflow branch sits AFTER the rescue block rather than inside it, so
guarding the rescue must not change what gets FLAGGED — that is the second test
below, and it is the one that would have caught a guard placed one level too
high.
"""
from __future__ import annotations
import json
import struct
import wave
from pathlib import Path

import pytest

from pipeline import manifest as M
from pipeline import s5_fit as S5

CFG = {"work_dir": None,
       "fit": {"max_tempo": 1.12, "soft_tempo": 1.06, "drift_max_s": 1.5,
               "min_gap_s": 0.15},
       "tts": {"engine": "edge", "reference_wav": "", "sample_rate": 16000},
       "qc": {"eval": {"weights": {"sim": 0.0, "mos": 0.55, "f0": 0.30,
                                   "tempo": 0.15}}}}


def _wav(p: Path, seconds: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 1) * int(16000 * seconds))


@pytest.fixture
def zero_slot(tmp_path):
    """One utterance that ends exactly at the video duration.

    retime_step caps the last segment at hard_end, so limit == placed and
    slot == max(0, limit - placed - min_gap) == 0.
    """
    cfg = {**CFG, "work_dir": str(tmp_path / "work")}
    video = "lesson.mp4"
    man = {"video": video, "duration": 10.0, "stages": {},
           "utterances": [{"id": "u0001", "start": 10.0, "end": 10.0,
                           "text_uk": "джерело",
                           "tr": {"en": {"text": "a line with nowhere to go",
                                         "variants": ["shorter", "shortest"]}}}]}
    wd = M.video_workdir(cfg, video)
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return cfg, video, wd


@pytest.fixture
def stubs(monkeypatch):
    """Record every LLM call and every synthesis; make audio deterministically."""
    calls = {"shorten": [], "synth": []}

    def fake_shorten(cfg, lang, text_uk, text, max_chars, n=2):
        calls["shorten"].append(max_chars)
        return ["rescued variant"]

    def fake_synth(text, lang, out, t, **kw):
        calls["synth"].append(text)
        _wav(Path(out), 3.0)
        return 4.0

    def fake_atempo(src, dst, tempo, fit_cfg):
        _wav(Path(dst), 3.0 / max(tempo, 0.01))

    monkeypatch.setattr(S5, "llm_shorten", fake_shorten)
    monkeypatch.setattr(S5, "synth_best_of", fake_synth)
    monkeypatch.setattr(S5, "_atempo", fake_atempo)
    return calls


def test_zero_slot_does_not_pay_the_llm_or_the_gpu(zero_slot, stubs):
    cfg, video, _ = zero_slot
    S5.run(cfg, video, ["en"])
    assert stubs["shorten"] == [], (
        "s5 called the emergency shortener on a zero-length slot — the budget "
        "it would ask for is 0 characters and nothing it returns can place")
    assert "rescued variant" not in stubs["synth"], (
        "s5 synthesized a rescue variant that choose_placement rejects for "
        "slot <= 0 regardless of its duration — a paid best_of unit for nothing")


def test_zero_slot_is_still_flagged_as_overflow(zero_slot, stubs):
    """The guard must skip the RESCUE, not the flagging. The overflow branch is
    below the rescue block, not inside it; a guard placed one level too high
    would silently stop reporting unplaceable segments, and `report`,
    batch_report and the autopilot spec (overflow_max: 0) all read that flag."""
    cfg, video, _ = zero_slot
    S5.run(cfg, video, ["en"])
    man = M.load(cfg, video)
    tr = man["utterances"][0]["tr"]["en"]
    assert tr["fit"] == "overflow", f"expected overflow, got {tr['fit']!r}"
    assert tr["tempo"] == cfg["fit"]["max_tempo"]


def test_a_real_slot_still_gets_the_rescue(zero_slot, stubs, monkeypatch):
    """The complement: where the rescue CAN help, it must still run. Otherwise
    this guard trades a wasted call for a lost one."""
    cfg, video, wd = zero_slot
    man = M.load(cfg, video)
    # a real 4 s slot, with every candidate far too long to fit or stretch
    man["utterances"][0]["start"] = 0.0
    man["utterances"][0]["end"] = 4.0
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    def long_synth(text, lang, out, t, **kw):
        stubs["synth"].append(text)
        _wav(Path(out), 1.0 if text == "rescued variant" else 30.0)
        return 4.0

    monkeypatch.setattr(S5, "synth_best_of", long_synth)
    S5.run(cfg, video, ["en"])
    assert stubs["shorten"], "the rescue is gone for slots where it works"
    assert stubs["shorten"][0] > 0, "a real slot must yield a positive budget"
    assert M.load(cfg, video)["utterances"][0]["tr"]["en"]["fit"] != "overflow"
