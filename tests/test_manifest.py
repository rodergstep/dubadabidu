# SPDX-License-Identifier: AGPL-3.0-or-later
from dubadabidu.core.manifest import synth_hash
from tests.cfgs import toy_tts

T = {"reference_wav": "ref/r.wav", "cfg_weight": 0.0,
     "exaggeration": 0.55, "engine": "chatterbox"}


def test_hash_stable():
    assert synth_hash("Hello", "en", toy_tts(T)) == synth_hash("Hello", "en", toy_tts(T))


def test_hash_changes_with_text_lang_params():
    h = synth_hash("Hello", "en", toy_tts(T))
    assert synth_hash("Hello!", "en", toy_tts(T)) != h
    assert synth_hash("Hello", "de", toy_tts(T)) != h
    assert synth_hash("Hello", "en", toy_tts({**T, "exaggeration": 0.7})) != h
