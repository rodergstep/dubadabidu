from pipeline.manifest import synth_hash

T = {"reference_wav": "ref/r.wav", "cfg_weight": 0.0,
     "exaggeration": 0.55, "engine": "chatterbox"}


def test_hash_stable():
    assert synth_hash("Hello", "en", T) == synth_hash("Hello", "en", T)


def test_hash_changes_with_text_lang_params():
    h = synth_hash("Hello", "en", T)
    assert synth_hash("Hello!", "en", T) != h
    assert synth_hash("Hello", "de", T) != h
    assert synth_hash("Hello", "en", {**T, "exaggeration": 0.7}) != h
