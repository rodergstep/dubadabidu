"""Word-level stress marking — the label every remediation route needs.

WHY THIS EXISTS. FINDINGS closes automated Russian stress detection: four
detectors were built and validated and all four came in at or under the AUC
0.647 that back-transcription WER already gives free (§2.1f-2.1h), and
consensus selection is ANTI-correlated on exactly the segments that matter
(§2.1i, §2.1j). The structural reason is §2.1j: the errors are partly
SYSTEMATIC, so for some words qwen is reliably wrong, the majority placement is
wrong, and no selection rule can reach them.

But systematic means DETERMINISTIC, and deterministic means ENUMERABLE. Every
attempt so far tried to solve stress in general — a detector that works on any
word, or conditioning that works on any word. A painting course has a bounded
lexicon (work/<video>/terms_<lang>.json already enumerates part of it), so the
words qwen reliably mis-stresses are a finite set. A word that is RELIABLY
wrong needs fixing once, not detecting every take.

That reframing escapes both traps that killed the detectors:
  - no oracle, so no inherited oracle error. §2.1h's real finding is that every
    oracle-based detector inherits RUAccent's error rate (it marks `перед` on
    the wrong slot) and that rate is comparable to the defect being measured.
    A human-populated table has no oracle.
  - no AUC bar, because nothing is being detected.

AND THE HUMAN DETECTOR ALREADY WORKS — it is the only one that ever did. The
listener produced 28 and then 16 labelled takes. The review page rates
SEGMENTS, so "which word" was discarded every single time. This module keeps
it: the marginal cost is one click on audio the reviewer is already playing.

What the table feeds (none of which needs a detector):
  1. lexical avoidance — a known-bad word list in the s3 prompt, exactly like
     the glossary; most pigment/technique terms have synonyms. Needs no new
     capability anywhere.
  2. ё — Russian ё is ALWAYS stressed, so writing it where text uses е is both
     correct orthography and an unambiguous cue qwen saw in training.
  3. reduction respelling — write unstressed о as а, е as и, in ordinary
     Cyrillic. NOT the refuted route: §2.1b/2.1bb refuted U+0301 specifically
     because qwen was not trained on it and reads it as content to pronounce.
     Ordinary Cyrillic it was trained on. UNTESTED, and it must go through the
     same gate the four detectors got before anything is wired in.

Ukrainian is included in STRESS_LANGS for later, but note the ru levers do NOT
transfer: Ukrainian has no ё, and no akanye (unstressed о stays [o]), so both
2 and 3 above are gone and only 1 carries over. It is also not a supported
Qwen3-TTS language at all (THIRD_PARTY.md), so uk is an engine problem before
it is a stress problem.
"""
from __future__ import annotations
import re
import unicodedata

# Languages whose stress is lexical, mobile and unmarked in writing — i.e.
# where a TTS must GUESS and can be wrong. A linguistic fact, not a preference,
# so it is a constant rather than a config knob.
STRESS_LANGS = ("ru", "uk")

# A word is a run of letters, allowing an internal hyphen or apostrophe so that
# compounds stay ONE token: `хром-кобальт`, `сине-зелёная`, `clair-obscur`,
# `п'ять`. Digits are excluded — they are expanded to words at the synthesis
# boundary (pipeline.text_norm), so a digit in the manifest is not what the
# voice said and marking it would label the wrong thing.
_WORD = re.compile(r"[^\W\d_]+(?:['’‐-―\-][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """The words of `text`, in order. Index into this list IS the stable id.

    Shared by the page and the ingest ON PURPOSE. The two used to hold separate
    copies of the export-key expression and drifted; this file exists so the
    same mistake cannot be made with tokenization, where a one-token offset
    would silently attribute a stress error to the wrong word.
    """
    return [m.group() for m in _WORD.finditer(text or "")]


def spans(text: str) -> list[tuple[str, int]]:
    """[(fragment, word_index_or_-1)] covering `text` with no gaps.

    Lets the page render clickable words while keeping punctuation and spacing
    exactly as written — the reviewer has to recognise the sentence they just
    heard.
    """
    out: list[tuple[str, int]] = []
    pos = 0
    for i, m in enumerate(_WORD.finditer(text or "")):
        if m.start() > pos:
            out.append((text[pos:m.start()], -1))
        out.append((m.group(), i))
        pos = m.end()
    if text and pos < len(text):
        out.append((text[pos:], -1))
    return out


def lexicon_key(word: str) -> str:
    """Normalised table key for one marked word.

    Case-folded (stress is a property of the lexeme, not of sentence position)
    and stripped of combining marks, so a stress-marked form can never open a
    second entry for a word already in the table.

    NOT lemmatised, deliberately. `молоко́` and `молока́` are different forms and
    a TTS can be right about one and wrong about the other, so the table is
    per-surface-form — which is also the form every remediation route needs to
    match against.
    """
    w = unicodedata.normalize("NFD", (word or "").strip().casefold())
    return unicodedata.normalize(
        "NFC", "".join(c for c in w if not unicodedata.combining(c)))


def verify(text: str, marks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split `marks` into (agreeing, stale) against the CURRENT text.

    A mark is {"i": word index, "w": surface form}. Both are exported so this
    check is possible at all: if `text` was hand-edited between review and
    ingest, the index now points at a different word, and silently recording
    the new one would poison the table with a word the listener never heard.
    """
    toks = tokenize(text)
    ok, stale = [], []
    for m in marks:
        i = m.get("i")
        w = m.get("w", "")
        if isinstance(i, int) and 0 <= i < len(toks) and toks[i] == w:
            ok.append({"i": i, "w": w})
        else:
            stale.append(m)
    return ok, stale
