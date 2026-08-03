"""s4's fit-variant pre-synthesis (s4_synthesize.VARIANT_MARGIN).

s4 pre-makes the shorter variants s5 would otherwise synthesize itself, because
s5 runs on the laptop where the GPU engine is not installed (course.py phase C
is local and free by design). It used to make ALL of them for every segment
that tripped the gate — up to translation.n_short_variants extra best_of units
per segment, i.e. up to 3x s4's GPU bill on those segments, for audio nothing
reads: s5 walks the candidates and takes the first that places.

These tests pin both halves of the contract — stop once a variant clears the
bar, but keep the VARIANT_MARGIN headroom that stops s5 needing a GPU.
"""
import json
import struct
import wave
from pathlib import Path

import pytest

from pipeline import s4_synthesize as S4
from pipeline import manifest as M

CFG = {"work_dir": None,           # filled per-test
       "fit": {"soft_tempo": 1.06},
       "tts": {"engine": "edge", "reference_wav": "", "sample_rate": 16000}}


def _wav(p: Path, seconds: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 1) * int(16000 * seconds))


@pytest.fixture
def bed(tmp_path):
    """A one-utterance video whose primary overruns a 10 s slot."""
    cfg = {**CFG, "work_dir": str(tmp_path / "work")}
    video = "lesson.mp4"
    man = {"video": video, "duration": 10.0, "stages": {},
           "utterances": [{"id": "u0001", "start": 0.0, "end": 10.0,
                           "text_uk": "джерело",
                           "tr": {"en": {"text": "the long primary line",
                                         "variants": ["shorter one",
                                                      "shortest"]}}}]}
    wd = M.video_workdir(cfg, video)
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return cfg, video


def _synth_stub(monkeypatch, durations: dict[str, float], default: float):
    """Give each candidate TEXT a chosen synth duration; record the order."""
    made: list[str] = []

    def fake(text, lang, out, t, meta=None, verify_cfg=None, verify_text=None,
             target_dur=None):
        made.append(text)
        _wav(Path(out), durations.get(text, default))
        return 4.0

    monkeypatch.setattr(S4, "synth_best_of", fake)
    return made


def test_stops_at_the_first_variant_that_clears_the_bar(bed, monkeypatch):
    cfg, video = bed
    # slot 10s, soft 1.06, margin 0.85 -> a variant is enough at <= 9.01s
    made = _synth_stub(monkeypatch, {"the long primary line": 12.0,
                                     "shorter one": 8.0,
                                     "shortest": 4.0}, 12.0)
    S4.run(cfg, video, ["en"])
    assert made == ["the long primary line", "shorter one"], \
        "'shortest' can never be reached by s5 once 'shorter one' fits"


def test_keeps_going_while_variants_are_still_too_long(bed, monkeypatch):
    cfg, video = bed
    made = _synth_stub(monkeypatch, {"the long primary line": 12.0,
                                     "shorter one": 11.0,
                                     "shortest": 4.0}, 12.0)
    S4.run(cfg, video, ["en"])
    assert made == ["the long primary line", "shorter one", "shortest"]


def test_the_margin_is_kept_not_traded_away(bed, monkeypatch):
    """A variant that fits the SOURCE slot but not with VARIANT_MARGIN must not
    end the loop: s5 measures against the retimed slot, which can be tighter,
    and a missing variant is fatal on a laptop with no GPU engine."""
    cfg, video = bed
    made = _synth_stub(monkeypatch, {"the long primary line": 12.0,
                                     "shorter one": 10.4,   # < slot*soft, > *margin
                                     "shortest": 4.0}, 12.0)
    S4.run(cfg, video, ["en"])
    assert made[-1] == "shortest", "the safety margin was traded for one take"


def test_no_variants_when_the_primary_comfortably_fits(bed, monkeypatch):
    cfg, video = bed
    made = _synth_stub(monkeypatch, {"the long primary line": 6.0}, 6.0)
    S4.run(cfg, video, ["en"])
    assert made == ["the long primary line"]


def test_cached_variants_are_not_re_synthesized_but_still_end_the_loop(
        bed, monkeypatch):
    """A resumed run must reach the same stopping point without paying again."""
    cfg, video = bed
    durations = {"the long primary line": 12.0, "shorter one": 8.0,
                 "shortest": 4.0}
    _synth_stub(monkeypatch, durations, 12.0)
    S4.run(cfg, video, ["en"])
    made2 = _synth_stub(monkeypatch, durations, 12.0)
    S4.run(cfg, video, ["en"])
    assert made2 == [], "nothing should re-synthesize on a resumed run"
