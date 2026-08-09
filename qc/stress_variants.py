"""Generate stress-variant pronunciations for Russian, by permuting vowels only.

THE TRICK THAT MAKES THIS CHEAP. Detecting stress by forced alignment means
offering the aligner one pronunciation per possible stress position and seeing
which fits the audio best. Building those from scratch would need a full Russian
G2P — real work, and a second thing to get wrong.

It is not necessary. The MFA Russian dictionary (v3.1.0, CC BY 4.0) already
gives a NARROW phone string per word, in which stress is encoded as vowel
quality: the stressed vowel is full (`a o e i u ɨ`), the rest are reduced
(`ə ɐ ɪ ʊ`). So a variant with the stress somewhere else is the same string with
the vowels re-qualified — consonants, palatalisation and gemination all stay
exactly as the dictionary has them. Only the vowel slots move.

Reduction follows the standard Moscow norm, which is what the dictionary
encodes:
  - stressed        -> full quality
  - immediately pre-tonic, or word-initial -> [ɐ] for /a o/
  - elsewhere       -> [ə] for /a o/
  - /e i/ unstressed -> [ɪ];  /u/ unstressed -> [ʊ];  /ɨ/ stays [ɨ]

Promotion is ambiguous in one direction — [ɐ] could underlie /a/ or /o/, [ɪ]
could underlie /e/ or /i/ — so BOTH candidates are emitted and the aligner is
left to choose. That is the right place for the ambiguity to live: it is an
acoustic question, and the acoustic model is what answers it.
"""
from __future__ import annotations

# Full (stressed) vowels in the MFA Russian phone set.
FULL = {"a", "o", "e", "i", "u", "ɨ", "ɛ", "æ", "ʉ", "ɵ"}
# Reduced (unstressed) vowels.
REDUCED = {"ə", "ɐ", "ɪ", "ʊ"}
VOWELS = FULL | REDUCED

# What a reduced vowel may underlyingly be. Ambiguous on purpose — see module
# docstring; the aligner resolves it.
PROMOTE = {"ə": ("a", "o"), "ɐ": ("a", "o"), "ɪ": ("i", "e"), "ʊ": ("u",)}
# What a full vowel becomes when it loses the stress. /a o/ depend on position
# (pre-tonic vs elsewhere), handled by the caller.
DEMOTE_FAR = {"a": "ə", "o": "ə", "e": "ɪ", "i": "ɪ", "u": "ʊ",
              "ɨ": "ɨ", "ɛ": "ɪ", "æ": "ɪ", "ʉ": "ʊ", "ɵ": "ə"}
DEMOTE_NEAR = {**DEMOTE_FAR, "a": "ɐ", "o": "ɐ", "ɵ": "ɐ"}


def vowel_slots(phones: list[str]) -> list[int]:
    """Indices of the vowel phones, in order — one per syllable."""
    return [i for i, p in enumerate(phones) if p in VOWELS]


def variants(phones: list[str]) -> dict[int, list[list[str]]]:
    """{stress_slot: [pronunciation, ...]} for every possible stress position.

    Slot numbering is over VOWELS (i.e. syllables), not over phones, so it lines
    up with `expected_stress_index` from the orthography.
    """
    slots = vowel_slots(phones)
    if len(slots) < 2:
        return {}
    out: dict[int, list[list[str]]] = {}
    for k, _ in enumerate(slots):
        cands: list[list[str]] = [[]]
        for si, pi in enumerate(slots):
            v = phones[pi]
            if si == k:
                # stressed: full quality. Promote if currently reduced, and keep
                # both readings when the underlying vowel is ambiguous.
                opts = [v] if v in FULL else list(PROMOTE.get(v, (v,)))
            else:
                # unstressed: reduce. Position matters only for /a o/ —
                # immediately pre-tonic and word-initial keep [ɐ], the rest [ə].
                near = (si == k - 1) or si == 0
                tbl = DEMOTE_NEAR if near else DEMOTE_FAR
                opts = [tbl.get(v, v)] if v in FULL else [v]
            cands = [c + [o] for c in cands for o in opts]
        # splice each vowel choice back into the full phone string
        outs = []
        for choice in cands:
            ph = list(phones)
            for si, pi in enumerate(slots):
                ph[pi] = choice[si]
            outs.append(ph)
        out[k] = outs
    return out


# [ɨ] is NOT diagnostic of stress. Russian ы has no distinct reduced counterpart
# in this phone set, so it is transcribed [ɨ] stressed or not — measured on real
# alignments, counting it as "full" produced spurious two-full-vowel words and
# threw away жёлтые / холодные / светлые as unreadable when the OTHER vowel gave
# a clean answer. It still occupies a syllable slot; it just cannot carry the
# verdict.
NON_DIAGNOSTIC = {"ɨ"}


def observed_slot(phones: list[str]) -> int | None:
    """Which syllable an ALIGNED phone string says was stressed.

    The full vowel is the stressed one. None when the reading is unusable — no
    full vowel, or more than one — which must not be silently resolved into a
    confident answer, because a guess here becomes a false stress error.
    """
    slots = vowel_slots(phones)
    if len(slots) < 2:
        return None
    full_at = [i for i, pi in enumerate(slots)
               if phones[pi] in FULL and phones[pi] not in NON_DIAGNOSTIC]
    return full_at[0] if len(full_at) == 1 else None
