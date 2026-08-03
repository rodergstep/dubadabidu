"""s6 timeline assembly.

The loop this replaced was `track = track.overlay(seg, position=pos)`. pydub
never mutates: each overlay copies the WHOLE timeline, so the loop was O(n^2) —
a 1 h lesson at 24 kHz mono is ~173 MB per copy and ~400 utterances moved tens
of GB to produce one track. These tests pin the two things that matter:
the output is identical to what the old loop produced, and the cost is linear.
"""
import time

from pydub import AudioSegment

from pipeline.s6_mix import _assemble

RATE = 24000


def _tone(ms: int, level: int = 8000, rate: int = RATE) -> AudioSegment:
    import numpy as np
    n = int(ms * rate / 1000)
    t = np.arange(n) / rate
    data = (level * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    return AudioSegment(data.tobytes(), frame_rate=rate, sample_width=2,
                        channels=1)


def _overlay_reference(placements, total_ms, rate):
    """Exactly the old implementation."""
    need_ms = max([total_ms] + [pos + len(seg) for pos, seg in placements])
    track = AudioSegment.silent(duration=need_ms, frame_rate=rate)
    for pos, seg in placements:
        track = track.overlay(seg, position=pos)
    return track


def test_matches_the_overlay_loop_it_replaced():
    placements = [(0, _tone(300)), (500, _tone(400)), (1200, _tone(250))]
    got = _assemble(placements, 2000, RATE)
    want = _overlay_reference(placements, 2000, RATE)
    assert len(got) == len(want)
    assert got.raw_data == want.raw_data


def test_timeline_grows_for_a_segment_that_runs_past_the_video():
    """s5's soft anchor can place the last dub late; pydub's overlay would
    TRUNCATE it against a track sized to the source duration."""
    placements = [(900, _tone(500))]
    got = _assemble(placements, 1000, RATE)
    assert len(got) == 1400
    assert got.raw_data == _overlay_reference(placements, 1000, RATE).raw_data


def test_resamples_a_segment_recorded_at_another_rate():
    """pydub's overlay ran _sync; dropping that would splice raw samples at the
    wrong rate and pitch-shift the segment."""
    placements = [(100, _tone(200, rate=16000))]
    got = _assemble(placements, 500, RATE)
    assert got.frame_rate == RATE
    assert got.raw_data == _overlay_reference(placements, 500, RATE).raw_data


def test_overlapping_segments_saturate_like_audioop_add():
    """s5 makes overlap impossible, but the mix must not depend on that."""
    loud = _tone(200, level=30000)
    placements = [(0, loud), (50, loud)]
    got = _assemble(placements, 300, RATE)
    assert got.raw_data == _overlay_reference(placements, 300, RATE).raw_data


def test_cost_is_linear_not_quadratic():
    """400 segments on a long timeline is the real shape (a 1 h lesson). The
    quadratic version is ~n times slower at 4x the segment count; linear is
    ~4x. Generous bound so this measures the algorithm, not the machine."""
    seg = _tone(100)
    small = [(i * 200, seg) for i in range(100)]
    large = [(i * 200, seg) for i in range(400)]

    def _timed(pl):
        total = pl[-1][0] + 1000
        t = time.perf_counter()
        _assemble(pl, total, RATE)
        return time.perf_counter() - t

    _assemble(small, 100000, RATE)          # warm numpy
    ratio = _timed(large) / max(_timed(small), 1e-4)
    assert ratio < 12, f"4x the segments took {ratio:.1f}x the time — quadratic"
