"""Crash-safety of the best-of take cache (tts_engine.synth_best_of).

Regression guard: `out` is the content-hash cache key — s4 treats its existence
as "this segment has an accepted take" (fresh = not out.exists()). Take 0 used
to be written straight there, so a run killed between take 0 and the ranking
left an UNRANKED, ungated take cached as the winner, with no tr.takes record.
Nothing re-derived it; only --force cleared it. That matters because the GPU
plan runs on preemptible spot pods with a budget deadline and a pod-side
self-destruct watchdog, all of which kill mid-segment.
"""
import struct
import wave
from pathlib import Path

import pytest

from pipeline import tts_engine as T

TCFG = {"engine": "edge", "best_of": 3, "rank_takes": True,
        "best_of_early_accept": False, "retake_mos_below": 0.0,
        "min_f0st": 0.0, "f0_reroll_max": 0, "reference_wav": ""}


def _wav(p: Path, marker: int) -> None:
    """A real 0.1s wav so soundfile.info works; `marker` makes takes distinct."""
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", marker) * 1600)


@pytest.fixture
def stub_metrics(monkeypatch):
    """Stub the model-backed metrics; this suite is about file lifecycle."""
    import qc.metrics
    monkeypatch.setattr(qc.metrics, "mos_min_window", lambda p: 4.0)
    monkeypatch.setattr(T, "_f0_delivered", lambda p: 3.0)


def _fake_synthesizer(monkeypatch, fail_on: int | None = None,
                      scores: dict | None = None):
    """Replace real TTS with wav-writing. fail_on=N raises on the Nth call."""
    calls = []

    def fake(text, lang, out, t, retries=2):
        calls.append(Path(out))
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("pod preempted mid-best-of")
        _wav(Path(out), len(calls))

    monkeypatch.setattr(T, "synthesize", fake)
    if scores is not None:
        import qc.metrics
        monkeypatch.setattr(qc.metrics, "mos_min_window",
                            lambda p: scores.get(Path(p).name, 4.0))
    return calls


def test_crash_mid_best_of_caches_nothing(tmp_path, monkeypatch, stub_metrics):
    """THE fix: two takes succeeded, the third died — `out` must not exist, so
    the next run re-rolls instead of serving an unranked take forever."""
    out = tmp_path / "u0001_abcd1234.wav"
    _fake_synthesizer(monkeypatch, fail_on=3)

    with pytest.raises(RuntimeError):
        T.synth_best_of("hello", "en", out, TCFG)

    assert not out.exists()


def test_successful_run_publishes_exactly_one_file(tmp_path, monkeypatch,
                                                   stub_metrics):
    out = tmp_path / "u0001_abcd1234.wav"
    _fake_synthesizer(monkeypatch)

    T.synth_best_of("hello", "en", out, TCFG)

    assert out.exists()
    assert list(tmp_path.iterdir()) == [out]     # no _take* left behind


def test_winner_content_is_published(tmp_path, monkeypatch, stub_metrics):
    """The file at `out` must be the take that actually won the ranking."""
    out = tmp_path / "u0001_abcd1234.wav"
    # take index 1 (2nd call, marker 2) scores best
    _fake_synthesizer(monkeypatch, scores={
        "u0001_abcd1234_take0.wav": 3.0,
        "u0001_abcd1234_take1.wav": 4.9,
        "u0001_abcd1234_take2.wav": 3.5})
    meta: list = []

    T.synth_best_of("hello", "en", out, TCFG, meta=meta)

    with wave.open(str(out), "rb") as w:
        marker = struct.unpack("<h", w.readframes(1))[0]
    assert marker == 2                                   # the 2nd take's audio
    assert [m["picked"] for m in meta] == [False, True, False]


def test_leftovers_from_a_crashed_attempt_are_swept(tmp_path, monkeypatch,
                                                    stub_metrics):
    """A previous crash (or a larger best_of) leaves _take files; the next
    attempt must not accumulate them on the pod's disk."""
    out = tmp_path / "u0001_abcd1234.wav"
    for k in range(5):                                   # a prior best_of: 5 run
        _wav(tmp_path / f"u0001_abcd1234_take{k}.wav", 99)
    _wav(tmp_path / "u0001_abcd1234_reroll1.wav", 99)
    _fake_synthesizer(monkeypatch)

    T.synth_best_of("hello", "en", out, TCFG)            # best_of: 3 now

    assert list(tmp_path.iterdir()) == [out]


def test_sweep_is_scoped_to_this_segment(tmp_path, monkeypatch, stub_metrics):
    """Other segments' in-flight takes must survive the sweep — and so must the
    load-bearing s5/s6 artifacts that share this segment's stem prefix."""
    out = tmp_path / "u0001_abcd1234.wav"
    neighbour = tmp_path / "u0002_beef5678_take0.wav"
    _wav(neighbour, 7)
    keeper = tmp_path / "u0001_ffff9999.wav"             # same id, other hash
    _wav(keeper, 8)
    fitted = tmp_path / "u0001_abcd1234_fit.wav"         # s5 stretch output
    _wav(fitted, 9)
    placed = tmp_path / "u0001_placed.wav"               # s6 mix input
    _wav(placed, 10)
    _fake_synthesizer(monkeypatch)

    T.synth_best_of("hello", "en", out, TCFG)

    assert neighbour.exists() and keeper.exists()
    assert fitted.exists() and placed.exists()
    assert out.exists()


def test_single_take_still_publishes_atomically(tmp_path, monkeypatch,
                                                stub_metrics):
    out = tmp_path / "u0001_abcd1234.wav"
    _fake_synthesizer(monkeypatch)

    T.synth_best_of("hello", "en", out, {**TCFG, "best_of": 1})

    assert out.exists()
    assert list(tmp_path.iterdir()) == [out]


def test_crash_on_the_very_first_take_caches_nothing(tmp_path, monkeypatch,
                                                     stub_metrics):
    out = tmp_path / "u0001_abcd1234.wav"
    _fake_synthesizer(monkeypatch, fail_on=1)

    with pytest.raises(RuntimeError):
        T.synth_best_of("hello", "en", out, TCFG)

    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
