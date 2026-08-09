"""Detect WRONG lexical stress in synthesized Russian, from the audio.

WHY THIS CAN WORK AT ALL, and why the first attempt could not.

An earlier detector measured acoustic prominence — energy x F0 x duration over
syllable nuclei guessed from a smoothed sonority envelope. It disagreed with
ITSELF on 29% of words when the same take was time-stretched (qc/stress_var.py),
because it had to infer where the vowels were and then rank a continuous
quantity. It measured its own instrument.

This reads a CATEGORICAL symbol instead. Russian reduces unstressed vowels:
/o/ and /a/ collapse to [ɐ]/[ə] when unstressed while the stressed vowel keeps
full quality. So in Russian the stressed syllable is not merely louder — it is a
DIFFERENT PHONEME. A multilingual phoneme recogniser that distinguishes `a o e
i u ɨ` from `ə ɐ ɪ ʊ` therefore reports stress position directly, and
facebook/wav2vec2-xlsr-53-espeak-cv-ft has all of both sets in its 392-token
vocabulary (checked 2026-08-09; it has NO ˈ/ˌ stress tokens, which is what sent
me looking at the vowels).

That is also why wrong stress sounds so wrong to a Russian listener rather than
merely odd, and why back-transcription WER cannot see it: Whisper is robust to
stress and still emits the right word (measured: error takes 0.090, clean 0.089,
AUC 0.647 — qc/stress_wer.py).

THE ORACLE is RUAccent, which was never the broken part. Its marks destroyed
qwen's audio when fed IN (97% unusable), but as a source of truth for which
vowel SHOULD carry stress it is accurate and free.

    expected  = RUAccent(word)                -> index of the stressed vowel
    observed  = phonemes(word audio)          -> index of the UNREDUCED vowel
    mismatch  = a stress error on that word

Validate before wiring it to anything: `python -m qc.stress_detect --validate`
scores it against the 28 human-labelled takes and must clearly beat AUC 0.647,
which is where WER — the metric we already have — already sits.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("dubadabidu.qc.stress_detect")

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SR = 16000

# Russian orthographic vowels, in the order they can appear in a word.
VOWELS_CYR = "аеёиоуыэюя"

# Full-quality vowels: in Russian these occur (essentially) only under stress.
FULL = set("aoeiuɨɛɔæɑ")
# Reduced vowels: unstressed positions. ɪ/ʊ also appear stressed in other
# languages, which is why they are weighted rather than treated as proof.
REDUCED = set("əɐɪʊɵɤ")

_model = None
_proc = None


def _load():
    """Feature extractor + CTC model + raw vocab.

    Deliberately NOT AutoProcessor. That pulls Wav2Vec2PhonemeCTCTokenizer,
    which imports `phonemizer` at construction — a dependency needed only for
    the TEXT->phoneme direction we never use, and one that additionally requires
    the espeak-ng system library. We only decode CTC ids, so the vocab file is
    enough and the pod stays free of a system package.
    """
    global _model, _proc
    if _model is None:
        import json
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (AutoFeatureExtractor,
                                  Wav2Vec2ForCTC)  # type: ignore
        log.info("loading %s (first run downloads ~1.2 GB)", MODEL)
        fe = AutoFeatureExtractor.from_pretrained(MODEL)
        vocab = json.loads(Path(
            hf_hub_download(MODEL, "vocab.json")).read_text(encoding="utf-8"))
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL)
        _model.eval()
        if torch.backends.mps.is_available():
            _model = _model.to("mps")
        _proc = (fe, {i: tok for tok, i in vocab.items()},
                 vocab.get("<pad>", 0))
    return _model, _proc


def phonemes(wave) -> str:
    """IPA phoneme string for one audio segment (space-separated tokens)."""
    import torch
    m, (fe, id2tok, blank) = _load()
    inp = fe(wave, sampling_rate=SR, return_tensors="pt")
    dev = next(m.parameters()).device
    with torch.no_grad():
        logits = m(inp.input_values.to(dev)).logits
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    # standard CTC collapse: drop repeats, then drop blanks
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            tok = id2tok.get(i, "")
            if tok and not tok.startswith("<"):
                out.append(tok)
        prev = i
    return " ".join(out)


def _vowel_seq(ipa: str) -> list[str]:
    """Vowel nuclei in order. Diphthongs/long vowels collapse to their head so
    the count matches the orthographic syllable count."""
    out = []
    for tok in ipa.split():
        # a token may be a diphthong ("aɪ") or long vowel ("iː")
        head = tok[0]
        if head in FULL or head in REDUCED:
            out.append(head)
    return out


def expected_stress_index(word: str, accentizer=None) -> int | None:
    """Which vowel of `word` should carry the stress, 0-based. None if unknown
    or the word is monosyllabic (nothing to get wrong)."""
    vowels = [c for c in word.lower() if c in VOWELS_CYR]
    if len(vowels) < 2:
        return None
    # ё is ALWAYS stressed in Russian and needs no model
    if "ё" in word.lower():
        return vowels.index("ё")
    if accentizer is None:
        return None
    marked = accentizer.process_all(word)
    # RUAccent marks with "+" before the stressed vowel
    seen = 0
    for i, ch in enumerate(marked):
        if ch == "+" and i + 1 < len(marked):
            nxt = marked[i + 1].lower()
            if nxt in VOWELS_CYR:
                return seen
        if ch.lower() in VOWELS_CYR:
            seen += 1
    return None


def observed_stress_on_o(wave) -> bool:
    """Does this audio realise a FULL [o]?

    THE PREMISE THAT FAILED, kept because the failure is the finding. The plan
    was to read stress off vowel REDUCTION: unstressed /o a/ collapse to [ɐ ə],
    so the unreduced vowel is the stressed one. Measured over 320 vowels from 8
    real takes, this model emits **99% full vowels and 1% reduced** — it
    transcribes Russian broadly, not narrowly, and there is no reduction signal
    to read.

    What DID survive is the о/а contrast (akanye) specifically: unstressed
    orthographic о is realised [a], so a full [o] in the output marks a STRESSED
    о. Measured on 66 words with ±120 ms slice padding:
        stressed-о words   -> [o] present 89%   (sensitivity)
        unstressed-о words -> [o] present 26%   (false positives)
        separation +0.63
    Narrow but real. It only speaks about words containing о — that is the
    coverage price of the model's broad transcription.
    """
    return "o" in _vowel_seq(phonemes(wave))


# Whisper word boundaries clip the edges off short Russian words, and a
# truncated slice loses exactly the vowel being judged. Measured: 'o' found on a
# stressed о in 50% of words at no padding, 68% at ±60 ms, 86% at ±120 ms — the
# same localisation weakness that made the earlier prominence detector useless.
SLICE_PAD_S = 0.12


def word_mismatches(wav: Path, words: list[tuple[str, float, float]],
                    accentizer) -> list[dict]:
    """Per-word stress verdicts for one take. `words` is (text, start_s, end_s).

    Only words CONTAINING о are judged: they are the ones akanye makes audible.
    Everything else is reported as unresolved rather than guessed at.
    """
    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    out = []
    for text, t0, t1 in words:
        w = re.sub(r"[^а-яёА-ЯЁ]", "", text).lower()
        exp = expected_stress_index(w, accentizer)
        if exp is None or "о" not in w:
            continue
        seg = y[max(0, int((t0 - SLICE_PAD_S) * SR)):int((t1 + SLICE_PAD_S) * SR)]
        if len(seg) < int(0.08 * SR):
            continue
        vowels = [c for c in w if c in VOWELS_CYR]
        expect_o = vowels[exp] == "о"
        heard_o = observed_stress_on_o(seg)
        out.append({"word": w, "expect_o": expect_o, "heard_o": heard_o,
                    "resolved": True, "mismatch": expect_o != heard_o})
    return out


# --- validation gate -------------------------------------------------------

def validate(labels_json: str, truth_json: str, wd: str) -> int:
    """Score the detector against human-labelled takes.

    The bar is AUC 0.647 — where back-transcription WER already sits, i.e. what
    we get for free today. Anything at or below that is not worth wiring in, and
    saying so here is the whole point of this function: the previous detector
    was built and used before it was ever checked against a label.
    """
    import json
    from statistics import mean
    import mlx_whisper
    from ruaccent import RUAccent

    truth = json.loads(Path(truth_json).read_text(encoding="utf-8"))
    truth.pop("_axis", None)
    truth.pop("_build", None)
    rated = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    bad_keys = {k for v in rated.values() for k in v.get("bad", [])}
    # only groups the listener actually judged carry a label
    judged = {g for g, v in rated.items() if v.get("bad") or v.get("best")}

    acc = RUAccent()
    acc.load(omograph_model_size="turbo", use_dictionary=True)

    rows = []
    for key, meta in sorted(truth.items()):
        if key.split("c")[0] not in judged:
            continue
        wav = Path(wd) / "bakeoff" / meta["path"]
        if not wav.exists():
            continue
        r = mlx_whisper.transcribe(
            str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="ru", word_timestamps=True, verbose=False)
        words = [(w["word"], float(w["start"]), float(w["end"]))
                 for seg in r.get("segments", []) for w in seg.get("words", [])]
        ms = word_mismatches(wav, words, acc)
        n_res = sum(m["resolved"] for m in ms)
        n_bad = sum(m["mismatch"] for m in ms)
        rows.append({"key": key, "label": "ERROR" if key in bad_keys else "clean",
                     "n_words": len(ms), "resolved": n_res, "mismatches": n_bad,
                     "rate": n_bad / n_res if n_res else 0.0,
                     "detail": [m for m in ms if m["mismatch"]]})
        print(f"  {key} {rows[-1]['label']:6} words={len(ms):2} "
              f"resolved={n_res:2} mismatch={n_bad:2}", flush=True)

    bad = [r for r in rows if r["label"] == "ERROR"]
    good = [r for r in rows if r["label"] == "clean"]
    if not bad or not good:
        print("not enough labelled takes")
        return 1
    print("\n" + "=" * 64)
    cov = mean(r["resolved"] / r["n_words"] for r in rows if r["n_words"])
    print(f"syllable-count agreement (coverage): {cov:.0%} of words")
    for feat in ("mismatches", "rate"):
        b = [r[feat] for r in bad]
        g = [r[feat] for r in good]
        wins = sum((x > y) + 0.5 * (x == y) for x in b for y in g)
        auc = wins / (len(b) * len(g))
        print(f"{feat:11} error-take mean {mean(b):.3f} | clean {mean(g):.3f} "
              f"| AUC {auc:.3f}")
    best = max(
        (sum((x > y) + 0.5 * (x == y) for x in [r[f] for r in bad]
             for y in [r[f] for r in good]) / (len(bad) * len(good)))
        for f in ("mismatches", "rate"))
    print(f"\nbar to beat (WER today): 0.647   ->  "
          f"{'PASSES' if best > 0.72 else 'FAILS — do not wire this in'}")
    return 0 if best > 0.72 else 1


if __name__ == "__main__":
    import glob
    import sys
    logging.basicConfig(level=logging.INFO)
    wd = glob.glob("work/Organising*")[0]
    sys.exit(validate(
        "/Users/diadumenoss/Downloads/compare_ru_stressing.json",
        f"{wd}/bakeoff/compare_ru_truth.json", wd))
