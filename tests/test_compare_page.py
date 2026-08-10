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


def test_localstorage_key_is_unique_per_build(tmp_path):
    """One key per language meant a new page loaded the PREVIOUS page's answers
    and the download merged both. A 15x2 ru page came back on 2026-08-09
    carrying 64 marks for clips it never contained."""
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 1)
    _mk(bo / "seg" / "B" / "ru", 1)
    one = compare.build(tmp_path, "ru", ["A", "B"], embed=False).read_text()
    _mk(bo / "seg" / "C" / "ru", 1)
    two = compare.build(tmp_path, "ru", ["A", "B", "C"], embed=False).read_text()

    import re
    k1 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", one))
    k2 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", two))
    assert len(k1) == 1 and len(k2) == 1, "get and set must use the SAME key"
    assert k1 != k2, (
        f"both builds store under {k1} — a differently-shaped page would load "
        f"stale answers and silently merge them into the download")


def test_the_axis_under_test_is_printed_on_the_page(tmp_path):
    """The listener judges what the PAGE says, not what chat said. A page built
    to test monotony was rated for stress because the axis lived only in the
    conversation."""
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 1)
    _mk(bo / "seg" / "B" / "ru", 1)
    html = compare.build(tmp_path, "ru", ["A", "B"], embed=False,
                         axis="monotony — which is LESS flat").read_text()
    assert "monotony — which is LESS flat" in html
    assert "__AXIS__" not in html


def test_truth_file_records_the_axis_and_build(tmp_path):
    """Un-blinding months later must not have to guess what was being judged."""
    import json
    bo = tmp_path / "bakeoff"
    _mk(bo / "seg" / "A" / "ru", 1)
    _mk(bo / "seg" / "B" / "ru", 1)
    compare.build(tmp_path, "ru", ["A", "B"], embed=False, axis="stress only")
    truth = json.loads((bo / "compare_ru_truth.json").read_text())
    assert truth["_axis"] == "stress only" and truth["_build"]


def test_skip_groups_continues_where_the_last_page_stopped(tmp_path):
    """Rating rounds are capped to stay short, so the rest must be reachable
    without re-asking what was already answered."""
    bo = tmp_path / "bakeoff"
    for v in ("A", "B"):
        d = bo / "seg" / v / "ru"
        d.mkdir(parents=True)
        for i, secs in enumerate([9.0, 8.0, 7.0, 6.0]):
            sf.write(d / f"u{i:04d}_t0.wav", np.zeros(int(16000 * secs)), 16000)
    first = compare._groups(bo, "ru", ["A", "B"], 5.0)[:2]
    rest = compare._groups(bo, "ru", ["A", "B"], 5.0)[2:]
    assert [g["seg"] for g in first] + [g["seg"] for g in rest] == \
        [g["seg"] for g in compare._groups(bo, "ru", ["A", "B"], 5.0)]
    assert not ({g["seg"] for g in first} & {g["seg"] for g in rest})


def test_same_shape_different_content_gets_a_different_key(tmp_path):
    """THE case the first version of this test missed. Clip keys are g00c0,
    g00c1, ... so every 12x2 page hashed identically regardless of the audio in
    it — and a page built from different variants loaded the previous page's
    answers. Shipped 2026-08-09, caught by the listener on the next page."""
    import re
    bo = tmp_path / "bakeoff"
    for v in ("A", "B", "C"):
        _mk(bo / "seg" / v / "ru", 1)
    one = compare.build(tmp_path, "ru", ["A", "B"], embed=False).read_text()
    two = compare.build(tmp_path, "ru", ["A", "C"], embed=False).read_text()
    k1 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", one))
    k2 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", two))
    assert k1 != k2, (
        f"same shape, different variants, same storage key {k1} — the second "
        f"page will load the first page's ratings")


def test_a_different_axis_alone_gets_a_different_key(tmp_path):
    """Same clips, different QUESTION, is a different experiment: monotony
    answers must not prefill a stress page."""
    import re
    bo = tmp_path / "bakeoff"
    for v in ("A", "B"):
        _mk(bo / "seg" / v / "ru", 1)
    one = compare.build(tmp_path, "ru", ["A", "B"], embed=False,
                        axis="monotony").read_text()
    two = compare.build(tmp_path, "ru", ["A", "B"], embed=False,
                        axis="stress only").read_text()
    k1 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", one))
    k2 = set(re.findall(r"localStorage\.\w+Item\('([^']+)'", two))
    assert k1 != k2
