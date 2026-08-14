from pipeline.manifest import synth_hash
from qc.bakeoff import beats_incumbent


def _tts(**over):
    base = {"engine": "chatterbox", "reference_wav": "ref/x.wav",
            "cfg_weight": 0.0, "exaggeration": 0.55}
    return {**base, **over}


# --- adoption gate ---

def test_gate_requires_both_sim_and_mos():
    inc = {"sim": 0.74, "mos": 4.5}
    assert beats_incumbent({"sim": 0.76, "mos": 4.6}, inc)
    assert not beats_incumbent({"sim": 0.80, "mos": 4.4}, inc)   # mos loses
    assert not beats_incumbent({"sim": 0.70, "mos": 4.9}, inc)   # sim loses


def test_gate_margin():
    inc = {"sim": 0.74, "mos": 4.5}
    tie = {"sim": 0.74, "mos": 4.5}
    assert beats_incumbent(tie, inc)                     # tie passes at eps=0
    assert not beats_incumbent(tie, inc, sim_eps=0.01)   # margin demanded


def test_wer_veto_disqualifies_sim_mos_winner():
    inc = {"sim": 0.74, "mos": 4.5, "wer": 0.05}
    # wins sim+mos but hallucinates (WER far worse) -> vetoed
    assert not beats_incumbent({"sim": 0.80, "mos": 4.7, "wer": 0.20}, inc)
    # wins sim+mos and holds intelligibility within tolerance -> adopts
    assert beats_incumbent({"sim": 0.80, "mos": 4.7, "wer": 0.06}, inc)


def test_wer_veto_skipped_when_absent():
    inc = {"sim": 0.74, "mos": 4.5}          # legacy/partial scorecard, no wer
    assert beats_incumbent({"sim": 0.76, "mos": 4.6}, inc)  # gates on sim+mos only


# --- cache keying: new engine params must change the hash, chatterbox unchanged ---

def test_chatterbox_hash_stable():
    a = synth_hash("hello", "en", _tts())
    b = synth_hash("hello", "en", _tts())
    assert a == b


def _bakeoff_cfg(work_dir):
    return {"work_dir": str(work_dir),
            "tts": {"engine": "chatterbox", "reference_wav": "ref/x.wav",
                    "cfg_weight": 0.0, "exaggeration": 0.55},
            "bakeoff": {"engines": ["chatterbox"], "takes": 1,
                        "subset_size": 2, "tune": {"enabled": False}},
            "qc": {"wer_flag_threshold": 0.15}, "fit": {"max_tempo": 1.12}}


def test_missing_voice_anchor_fails_with_an_actionable_message(tmp_path,
                                                              monkeypatch):
    """The anchor IS the cross-engine metric, so there is no fallback to degrade
    to. Without a guard this was torch's opaque 'stack expects a non-empty
    TensorList' -- raised AFTER the engine installs, i.e. at the most expensive
    point of a billed pod run."""
    import json
    import pytest
    # bakeoff.run imports torch before it reaches the guard under test. CI
    # deliberately omits the heavy ML stack, so skip rather than fail there —
    # this test is about the reference-slice check, not about torch.
    pytest.importorskip("torch")
    import qc.evaluate
    from qc import bakeoff

    wd = tmp_path / "vid"
    wd.mkdir()
    (wd / "manifest.json").write_text(json.dumps({
        "video": "vid.mp4", "duration": 60.0, "stages": {},
        "utterances": [{"id": f"u{i:04d}", "start": i, "end": i + 0.5,
                        "text_uk": "коротко", "tr": {"en": {"text": "short"}}}
                       for i in range(3)]}))
    # every sampled utterance is under 1s -> _ua_slices yields nothing
    monkeypatch.setattr(qc.evaluate, "_ua_slices", lambda *a, **k: iter(()))

    with pytest.raises(SystemExit, match="no usable reference slices"):
        bakeoff.run(_bakeoff_cfg(tmp_path), "vid", ["en"])


# --- adoption gate: it must never look like it ran when it didn't ---

def test_incumbent_is_configurable():
    """Hardcoding it is how the gate died: engines became [qwen] while INCUMBENT
    stayed 'chatterbox', so beats_incumbent() never had a baseline."""
    from qc.bakeoff import incumbent_of, INCUMBENT
    assert incumbent_of({}) == INCUMBENT              # back-compatible default
    assert incumbent_of({"incumbent": "qwen"}) == "qwen"


def test_missing_baseline_is_reported_as_advisory_not_as_a_pass():
    """The old wording ('no incumbent baseline') sat in a column of verdicts and
    read like a neutral note, so a run where the gate never executed looked the
    same as one where it passed."""
    from qc.bakeoff import _verdict
    ch = {"sim": 0.7, "mos": 2.6, "wer": 0.004}
    v = _verdict("challenger", ch, None, "qwen")
    assert "ADVISORY" in v and "did NOT run" in v
    assert "ADOPT" not in v


def test_verdict_still_adopts_when_a_baseline_exists():
    from qc.bakeoff import _verdict
    inc = {"sim": 0.68, "mos": 2.48, "wer": 0.005}
    better = {"sim": 0.70, "mos": 2.60, "wer": 0.004}
    worse = {"sim": 0.60, "mos": 2.30, "wer": 0.004}
    assert _verdict("c", better, inc, "qwen") == "ADOPT"
    assert _verdict("c", worse, inc, "qwen") == "keep incumbent"
    assert _verdict("qwen", inc, inc, "qwen") == "incumbent (qwen)"


def test_variant_label_separates_an_otherwise_identical_run():
    """A control run must NOT merge with the row it is being compared against —
    that is the whole experiment. Identical config + a label = distinct key."""
    from qc.bakeoff import variant_key
    t = {"engine": "qwen", "qwen_fast": True}
    assert variant_key("qwen", t, "en") == "qwen+fast"
    assert variant_key("qwen", dict(t, variant_label="control"), "en") \
        == "qwen+fast+control"


def test_variant_label_is_not_in_synth_hash():
    """It changes no synthesis input, so salting the cache key with it would
    force pointless re-synthesis everywhere else in the pipeline. Fresh takes
    for the control come from the per-variant seg/ directory instead."""
    from pipeline.manifest import synth_hash
    base = {"engine": "qwen", "reference_wav": "r.wav",
            "cfg_weight": 0.0, "exaggeration": 0.5}
    assert synth_hash("hi", "en", base) == \
        synth_hash("hi", "en", dict(base, variant_label="control"))


def test_a_bakeoff_that_measured_nothing_exits_nonzero():
    """Every engine unavailable used to exit 0, so remote_run recorded ok=True
    and the ledger row looked like a clean run — 11.6 min of billed bootstrap
    producing zero rows, indistinguishable from a good result until someone
    opened the scorecard (2026-08-13)."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "qc" / "bakeoff.py").read_text(
        encoding="utf-8")
    assert "measured NOTHING" in src, "a zero-measurement run still exits 0"
    assert 'all("unavailable" in v for v in per_engine.values())' in src, (
        "the check must look at THIS run's engines, not the merged scorecard — "
        "engines from previous runs are not evidence this pod did anything")


def test_verdict_names_the_actual_failure():
    """"n/a (not installed)" was printed for every cause and sent a diagnosis
    the wrong way: the install had printed "qwen ok" and what really failed was
    the weights fetch."""
    from qc.bakeoff import _verdict
    hub = _verdict("qwen", {"unavailable": "An error happened while trying to "
                            "locate the file on the Hub"}, None)
    imp = _verdict("qwen", {"unavailable": "qwen_tts not importable"}, None)
    assert "weights" in hub and "not installed" not in hub, hub
    assert "not installed" in imp, imp
