from pipeline.manifest import synth_hash
from pipeline.tts_engine import _cosyvoice_mode
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


# --- cosyvoice mode resolution ---

def test_cosyvoice_default_cross_lingual():
    assert _cosyvoice_mode({}) == "cross_lingual"


def test_cosyvoice_auto_picks_instruct_then_zeroshot():
    assert _cosyvoice_mode({"cosyvoice_mode": "auto",
                            "instruct_text": "warmly"}) == "instruct"
    assert _cosyvoice_mode({"cosyvoice_mode": "auto",
                            "reference_text": "hi"}) == "zero_shot"
    assert _cosyvoice_mode({"cosyvoice_mode": "auto"}) == "cross_lingual"


def test_cosyvoice_explicit_mode_wins():
    assert _cosyvoice_mode({"cosyvoice_mode": "zero_shot"}) == "zero_shot"


# --- cache keying: new engine params must change the hash, chatterbox unchanged ---

def test_chatterbox_hash_stable():
    a = synth_hash("hello", "en", _tts())
    b = synth_hash("hello", "en", _tts())
    assert a == b


def test_cosyvoice_mode_changes_hash():
    cl = synth_hash("hi", "en", _tts(engine="cosyvoice",
                                     cosyvoice_mode="cross_lingual"))
    zs = synth_hash("hi", "en", _tts(engine="cosyvoice",
                                     cosyvoice_mode="zero_shot"))
    assert cl != zs


def test_cosyvoice_instruct_changes_hash():
    plain = synth_hash("hi", "en", _tts(engine="cosyvoice"))
    inst = synth_hash("hi", "en", _tts(engine="cosyvoice",
                                       instruct_text="sadly"))
    assert plain != inst


def test_indextts_emotion_changes_hash():
    plain = synth_hash("hi", "en", _tts(engine="indextts"))
    emo = synth_hash("hi", "en", _tts(engine="indextts",
                                      emotion_wav="work/x/qc_ua/u1.wav"))
    assert plain != emo


def test_indextts_duration_changes_hash():
    plain = synth_hash("hi", "en", _tts(engine="indextts"))
    dur = synth_hash("hi", "en", _tts(engine="indextts",
                                      indextts_duration_ratio=1.15))
    assert plain != dur


# --- real-voice anchor is mandatory ---

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
