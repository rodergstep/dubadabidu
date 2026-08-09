"""One take per variant on the comparison page."""
from pathlib import Path
import soundfile as sf
import numpy as np
from qc import compare


def _mk(d: Path, n: int, secs: float = 6.0):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(d / f"u0001_t{i}.wav", np.zeros(int(16000 * secs)), 16000)


def test_default_is_one_clip_per_variant(tmp_path):
    """4 clips for a 2-arm test is what the listener rejected: more clips per
    decision is harder, not more informative."""
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 2)
    _mk(bo / "seg" / "B" / "ru", 2)
    g = compare._groups(bo, "ru", ["A", "B"], 5.0)
    assert len(g) == 1
    assert len(g[0]["takes"]) == 2, "one take per variant, not every take"
    assert {p.parent.parent.name for p in g[0]["takes"]} == {"A", "B"}


def test_the_same_take_index_is_taken_from_both_arms(tmp_path):
    """Positional rule, identical for both arms — picking each variant's
    'best' would use the take-selection objective, which correlates +0.022
    with this listener and would bias the comparison."""
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 3)
    _mk(bo / "seg" / "B" / "ru", 3)
    g = compare._groups(bo, "ru", ["A", "B"], 5.0)
    assert {p.name for p in g[0]["takes"]} == {"u0001_t0.wav"}


def test_more_takes_can_still_be_requested(tmp_path):
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 2)
    _mk(bo / "seg" / "B" / "ru", 2)
    g = compare._groups(bo, "ru", ["A", "B"], 5.0, takes_per_variant=2)
    assert len(g[0]["takes"]) == 4
