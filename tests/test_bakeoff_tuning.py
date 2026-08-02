"""Bake-off tuning parity (IMPROVEMENT_PLAN Phase C: tune-lite on BOTH engines).

Regression guard: the bake-off used to hand every challenger its library
defaults while chatterbox arrived at a cfg_weight/exaggeration a real tune run
had selected. An untuned-vs-tuned comparison predictably returns "keep
incumbent" for all four challengers — and that result would have looked like
evidence rather than an artifact of the protocol.
"""
from pathlib import Path

import pytest

from pipeline.manifest import synth_hash
from qc import bakeoff as B


# --- grid expansion ---

def test_empty_grid_is_one_default_point():
    assert B._grid_points({}) == [{}]


def test_grid_is_a_full_cartesian_product():
    pts = B._grid_points({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(pts) == 6
    assert {"a": 1, "b": "x"} in pts and {"a": 2, "b": "z"} in pts


def test_empty_axis_is_ignored_not_collapsing_the_grid():
    """A key present with no values must not wipe out the other axes."""
    assert B._grid_points({"a": [1, 2], "b": []}) == [{"a": 1}, {"a": 2}]


def test_grid_points_are_deterministic():
    g = {"a": [0.5, 1.0], "b": [0.9, 1.1]}
    assert B._grid_points(g) == B._grid_points(g)


# --- default grids: parity, and UA-reference safety ---

def test_every_bakeoff_engine_has_a_grid():
    for engine in ["qwen", "edge"]:
        assert engine in B.ENGINE_GRIDS


def test_no_default_grid_needs_a_target_language_ref_transcript():
    """A Ukrainian ref cannot be tokenized by qwen's full clone mode — sweeping
    it would only manufacture failures."""
    assert "qwen_x_vector_only" not in B.ENGINE_GRIDS["qwen"]


def test_grid_axes_all_affect_the_synth_cache_key():
    """A swept knob that isn't in synth_hash would let the cache serve one grid
    point's audio for another, silently flattening the whole sweep."""
    base = {"engine": "x", "reference_wav": "ref/a.wav", "cfg_weight": 0.0,
            "exaggeration": 0.55}
    for engine, grid in B.ENGINE_GRIDS.items():
        for axis, values in grid.items():
            if len(values) < 2:
                continue
            t = {**base, "engine": engine}
            a = synth_hash("hi", "en", {**t, axis: values[0]})
            b = synth_hash("hi", "en", {**t, axis: values[1]})
            assert a != b, f"{engine}.{axis} does not reach synth_hash"


# --- tune-lite selection ---

@pytest.fixture
def stub_synth(monkeypatch, tmp_path):
    """Fake synthesis + metrics; the scores key off the grid point so we can
    assert which point tune-lite picks."""
    import qc.metrics
    from pipeline import tts_engine

    def fake_synth(text, lang, out, t, retries=2):
        Path(out).write_bytes(b"wav")

    monkeypatch.setattr(tts_engine, "synthesize", fake_synth)
    monkeypatch.setattr(qc.metrics, "ecapa_embed", lambda p: p)
    # f0 joined the tune objective 2026-08-01; without a stub it would try to
    # read these placeholder bytes as audio. Constant and above any floor, so
    # tests that predate it still assert on sim/mos exactly as before.
    monkeypatch.setattr(qc.metrics, "f0_semitone_std", lambda p: 2.5)
    return monkeypatch


def _subset():
    return [{"id": "u0001", "tr": {"en": {"text": "hello"}}}]


def test_tune_lite_picks_the_best_grid_point(stub_synth, tmp_path):
    import qc.metrics
    # cfg 2.5 is the good one; everything else is mediocre
    stub_synth.setattr(qc.metrics, "cosine",
                       lambda a, b: 0.9 if "p2_" in str(b) else 0.5)
    stub_synth.setattr(qc.metrics, "mos_min_window",
                       lambda p: 4.5 if "p2_" in str(p) else 3.0)

    grid = {"qwen_cfg": [1.5, 2.0, 2.5]}
    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1, grid)

    assert unavail is None
    assert over == {"qwen_cfg": 2.5}      # p2 = third point
    assert len(trials) == 3


def test_tune_lite_reports_every_trial_for_audit(stub_synth, tmp_path):
    import qc.metrics
    stub_synth.setattr(qc.metrics, "cosine", lambda a, b: 0.7)
    stub_synth.setattr(qc.metrics, "mos_min_window", lambda p: 4.0)

    _, trials, _ = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"qwen_cfg": [1.5, 2.0]})

    assert [t["point"] for t in trials] == [{"qwen_cfg": 1.5},
                                            {"qwen_cfg": 2.0}]
    assert all("sim" in t and "mos" in t and "score" in t for t in trials)


def test_tune_lite_surfaces_an_uninstalled_engine(monkeypatch, tmp_path):
    from pipeline import tts_engine

    def missing(text, lang, out, t, retries=2):
        raise FileNotFoundError("qwen_tts not importable — git clone ...")

    monkeypatch.setattr(tts_engine, "synthesize", missing)
    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"qwen_cfg": [0.5, 1.0]})

    assert over == {} and trials == []
    assert unavail == "qwen_tts not importable"


def test_tune_score_weights_sim_and_mos_comparably(stub_synth, tmp_path):
    """No single axis may dominate by SCALE, and f0 is NOT one of the axes.
    f0st was weighted at 0.25 for one day (2026-08-01) until control runs put
    its run-to-run band at 0.24-0.44 on an unchanged config — a term that
    cannot resolve itself cannot rank anything, so it was removed."""
    import qc.metrics
    # p0: sim 1.0 / mos 1.0 (norm 0.0)   p1: sim 0.0 / mos 5.0 (norm 1.0)
    stub_synth.setattr(qc.metrics, "cosine",
                       lambda a, b: 1.0 if "p0_" in str(b) else 0.0)
    stub_synth.setattr(qc.metrics, "mos_min_window",
                       lambda p: 1.0 if "p0_" in str(p) else 5.0)

    _, trials, _ = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"qwen_cfg": [1.5, 2.0]})

    # sim is 0..1 while mos is 1..5, so mos MUST be normalized before
    # weighting or a one-point mos swing would outrank everything else. Equal
    # weights on normalized terms means these mirror-image extremes tie.
    assert trials[0]["score"] == trials[1]["score"] == 0.5


# --- report ---

def test_report_records_what_each_engine_ran_at():
    tuning = {
        "chatterbox": {"winner": {}, "trials": [], "n_points": 1,
                       "skipped": "single-point grid"},
        "qwen": {"winner": {"qwen_cfg": 2.5},
                   "trials": [{"point": {"qwen_cfg": 2.5}, "sim": 0.8,
                               "mos": 4.4, "score": 0.825},
                              {"point": {"qwen_cfg": 1.5}, "sim": 0.6,
                               "mos": 4.0, "score": 0.675}],
                   "n_points": 2},
    }
    md = "\n".join(B._tuning_section(tuning, ["chatterbox", "qwen"]))
    assert "qwen_cfg=2.5" in md                  # the winner is stated
    assert "single-point grid" in md             # and why the incumbent skipped
    assert "grid points that lost" in md         # losers are auditable


def test_fmt_point_labels_the_default_case():
    assert B._fmt_point({}) == "(defaults)"
    assert B._fmt_point({"b": 2, "a": 1}) == "a=1 b=2"     # sorted, stable


# --- subset sizing (tune-lite runs on small n) ---

def _us(n):
    return [{"id": f"u{i:04d}", "text_uk": "x" * (i + 1)} for i in range(n)]


def test_subset_of_one_takes_the_median_length_segment():
    """bakeoff.tune.subset_size: 1 is the natural cheapest setting; the
    even-spacing formula divides by n-1 and used to raise ZeroDivisionError."""
    from pipeline.tune import _subset
    got = _subset(_us(5), 1)
    assert [u["id"] for u in got] == ["u0002"]


def test_subset_zero_is_empty_not_a_crash():
    from pipeline.tune import _subset
    assert _subset(_us(5), 0) == []


def test_subset_spans_short_and_long():
    from pipeline.tune import _subset
    got = _subset(_us(9), 3)
    assert [u["id"] for u in got] == ["u0000", "u0004", "u0008"]


def test_subset_smaller_than_n_returns_everything():
    from pipeline.tune import _subset
    assert len(_subset(_us(2), 5)) == 2


# --- a bad grid point must not disqualify the engine ---

def test_one_failing_grid_point_is_skipped_not_fatal(monkeypatch, tmp_path):
    """A grid exists to EXPLORE. If one unsupported value disqualified the whole
    engine, the safe move would be never to widen a grid — which defeats the
    point. Only a total failure means the engine is unusable."""
    import qc.metrics
    from pipeline import tts_engine

    def fake(text, lang, out, t, retries=2):
        if t.get("qwen_cfg") == 9.9:          # unsupported value
            raise RuntimeError("engine rejected cfg_value=9.9")
        Path(out).write_bytes(b"wav")

    monkeypatch.setattr(tts_engine, "synthesize", fake)
    monkeypatch.setattr(qc.metrics, "ecapa_embed", lambda p: p)
    monkeypatch.setattr(qc.metrics, "cosine", lambda a, b: 0.7)
    monkeypatch.setattr(qc.metrics, "mos_min_window", lambda p: 4.0)
    monkeypatch.setattr(qc.metrics, "f0_semitone_std", lambda p: 2.5)

    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"qwen_cfg": [2.0, 9.9, 2.5]})

    assert unavail is None                    # engine survives
    assert len(trials) == 2                   # the bad point is simply absent
    assert 9.9 not in [t["point"]["qwen_cfg"] for t in trials]
    assert over["qwen_cfg"] in (2.0, 2.5)


def test_engine_is_unavailable_only_when_every_point_fails(monkeypatch, tmp_path):
    from pipeline import tts_engine

    def always_fail(text, lang, out, t, retries=2):
        raise RuntimeError("worker died: No space left on device")

    monkeypatch.setattr(tts_engine, "synthesize", always_fail)
    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"qwen_cfg": [0.5, 1.0]})

    assert over == {} and trials == []
    assert "No space left" in unavail


def test_removed_engines_stay_removed():
    """cosyvoice (never produced audio), voxcpm and indextts (both lost to
    qwen+fast on speed and cost) were cut 2026-07-31. A grid reappearing would
    resurrect an engine whose adapter no longer exists — the bake-off would then
    report it unavailable every run instead of failing loudly here."""
    for gone in ("cosyvoice", "voxcpm", "indextts", "chatterbox"):
        assert gone not in B.ENGINE_GRIDS


# --- f0st in the tune objective (added 2026-08-01) ---

def test_tune_disqualifies_points_under_the_monotony_floor(monkeypatch, tmp_path):
    """The sim+mos objective picked the most MONOTONE reference we own, because
    a flat clip reads as clean and on-voice. min_f0st must veto that outright."""
    from pipeline import tts_engine
    from qc import metrics as X

    # point 0: flat but pristine.  point 1: livelier, slightly worse mos/sim.
    vals = {"p0": (0.90, 4.5, 1.0), "p1": (0.80, 4.0, 3.0)}

    def fake_synth(text, lang, out, t, retries=2):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")

    monkeypatch.setattr(tts_engine, "synthesize", fake_synth)
    key = lambda w: "p0" if "p0_" in w.name else "p1"
    monkeypatch.setattr(X, "ecapa_embed", lambda w: w)
    monkeypatch.setattr(X, "cosine", lambda a, w: vals[key(w)][0])
    monkeypatch.setattr(X, "mos_min_window", lambda w: vals[key(w)][1])
    monkeypatch.setattr(X, "f0_semitone_std", lambda w: vals[key(w)][2])

    grid = {"z": [0, 1]}
    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1, grid,
        min_f0st=2.2)
    assert unavail is None
    # without the floor p0 wins on sim+mos; with it, p0 is disqualified
    assert over == {"z": 1}
    assert [r["under_floor"] for r in trials] == [True, False]


def test_tune_relaxes_the_floor_when_nothing_clears_it(monkeypatch, tmp_path):
    """Returning nothing would mark a working engine unavailable; ranking the
    unreachable set is the same `filtered or pool` shape synth_best_of uses."""
    from pipeline import tts_engine
    from qc import metrics as X
    monkeypatch.setattr(tts_engine, "synthesize",
                        lambda text, lang, out, t, retries=2:
                        (out.parent.mkdir(parents=True, exist_ok=True),
                         out.write_bytes(b"x")))
    monkeypatch.setattr(X, "ecapa_embed", lambda w: w)
    monkeypatch.setattr(X, "cosine", lambda a, w: 0.8)
    monkeypatch.setattr(X, "mos_min_window", lambda w: 4.0)
    monkeypatch.setattr(X, "f0_semitone_std", lambda w: 1.0)   # all under floor
    over, trials, unavail = B._tune_engine(
        "qwen", {"engine": "qwen"}, _subset(), "en", None, tmp_path, 1,
        {"z": [0, 1]}, min_f0st=2.2)
    assert unavail is None and over in ({"z": 0}, {"z": 1})
    assert all(r["under_floor"] for r in trials)


def test_bakeoff_synthesises_takes_serially():
    """Takes were pooled while each went to its own worker PROCESS. With the
    venvs merged (2026-08-02) synthesis runs in-process against ONE model held
    in module globals — not thread-safe, so two concurrent takes would drive a
    single model. Scoring stays overlapped; that is where the measured 40% was."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "qc" / "bakeoff.py").read_text()
    block = src[src.index("def _synth_one("):src.index("for w, _d in done:")]
    assert "ThreadPoolExecutor" not in block, \
        "synthesis must not be pooled while the model is a module global"
    assert "synth_workers" not in src, "the pool setting should be gone too"
