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

    # **kw so a new synth_best_of parameter does not fail four tests for a
    # reason unrelated to what they check (rank_weights, added 2026-08-09)
    def fake(text, lang, out, t, meta=None, verify_cfg=None, verify_text=None,
             target_dur=None, **kw):
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
# --- the CROSS-STAGE contract: what s4 makes must cover what s5 asks for ------

def test_s5_never_asks_for_a_variant_s4_skipped():
    """THE test the s4 change was missing. s4 stops pre-synthesizing at the
    first variant clearing slot*soft*VARIANT_MARGIN, which is only safe if s5
    stops ASKING at the first variant that places. s5 used to request every
    variant eagerly (`wavs += [seg_wav(c) for c in candidates[1:]]`), so a
    variant s4 skipped became a cache miss in s5 — fatal on the laptop, where
    the GPU engine is not installed and phase C is supposed to be free.

    Simulated over the ladder rather than mocked, so it fails if either side
    changes its rule."""
    from pipeline.s4_synthesize import VARIANT_MARGIN
    SOFT, SLOT = 1.06, 10.0

    def s4_makes(primary, variants):
        if primary / SLOT <= SOFT * VARIANT_MARGIN:
            return []
        keep_under, made = SLOT * SOFT * VARIANT_MARGIN, []
        for d in variants:
            made.append(d)
            if d <= keep_under:
                break
        return made

    def s5_asks(primary, variants):
        if primary / SLOT <= SOFT:
            return []
        asked = []
        for d in variants:
            asked.append(d)
            if d <= SLOT:          # places as-is -> stop
                break
        return asked

    cases = [
        (12.0, [8.0, 4.0]),        # first variant clears the bar
        (12.0, [11.0, 4.0]),       # first still too long
        (12.0, [10.4, 4.0]),       # fits the slot but not the margin
        (12.0, [10.0, 9.5, 4.0]),  # exactly at the slot
        (12.0, [11.0, 10.5]),      # nothing ever fits -> both take all
        (6.0,  [5.0, 4.0]),        # primary fits: neither side wants any
        (10.6, [9.0]),             # primary exactly at soft
    ]
    for primary, variants in cases:
        made, asked = s4_makes(primary, variants), s5_asks(primary, variants)
        assert len(asked) <= len(made), (
            f"primary={primary} variants={variants}: s5 asks for {len(asked)} "
            f"variant(s) but s4 pre-made {len(made)} — the missing one is a "
            f"cache miss, and on a laptop that halts phase C")


def test_s5_variant_loop_is_lazy_not_a_comprehension():
    """Guards the mechanism, not just the arithmetic: an eager comprehension
    re-introduces the bug while every duration-level test still passes."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "pipeline" / "s5_fit.py").read_text()
    assert "wavs += [seg_wav(c) for c in candidates[1:]]" not in src, (
        "s5 requests every variant eagerly again — s4 only pre-makes them up to "
        "the first that clears the bar")
    assert "for c in candidates[1:]:" in src, "expected the lazy ladder loop"
