"""Pick the take whose stress agrees with the other takes — no oracle, no human.

THE ONE FACT THIS RESTS ON. Russian stress errors from qwen are STOCHASTIC per
take: 8 of 14 sentences had exactly one bad take and ZERO had two (FINDINGS
2.1f). At a ~2.4%-per-word error rate, two takes independently choosing the SAME
wrong syllable is rare. So across K takes the modal stress placement for a word
is almost certainly right, and the take that agrees with the mode most often is
almost certainly the clean one.

WHY THIS ESCAPES WHAT BLOCKED EVERYTHING ELSE.
  - No oracle. RUAccent never enters. §2.1h measured it erring on real words
    (`перед` marked on the wrong syllable) and unable to resolve words that are
    genuinely ambiguous out of context (`цвета`). That capped every previous
    detector; here there is nothing for it to cap.
  - Detector bias cancels. If MFA systematically misreads a word it misreads it
    the same way in every take, so the take-vs-mode comparison is unaffected.
    That is why pairwise disagreement detection hit 6/6 sensitivity while
    absolute accuracy sat at chance.
  - No human. It is a selection rule, the shape `best_of` already has.

WHAT WOULD MAKE IT FAIL, and it is not ruled out: errors independent per take is
an assumption. If qwen holds a strong wrong prior for a particular word, all K
takes agree and consensus ships the error confidently. The 0-of-8 observation
was made with two takes, not five.

NOT WIRED IN until it beats a blind A/B against the status quo (a take picked
without regard to stress). Three detectors have failed that discipline's gate
already; this one gets the same treatment.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

from qc.stress_variants import observed_slot

log = logging.getLogger("dubadabidu.qc.stress_consensus")

# A word needs this many readable takes before its mode means anything. Two
# takes cannot form a majority — they can only disagree.
MIN_VOTES = 3


def slots_for(textgrid: Path) -> dict[str, int]:
    """{word: stress slot} for one take, unambiguous readings only.

    A word appearing twice in a sentence is dropped rather than guessed at: the
    two occurrences are different audio and pairing them by name would compare
    the wrong things.
    """
    from qc.stress_mfa import parse_textgrid
    seen: dict[str, list[int]] = {}
    for w, phones in parse_textgrid(textgrid):
        s = observed_slot(phones)
        if s is not None:
            seen.setdefault(w, []).append(s)
    return {w: v[0] for w, v in seen.items() if len(v) == 1}


def consensus(per_take: dict[str, dict[str, int]]) -> dict[str, int]:
    """{word: modal stress slot} across takes of the SAME sentence.

    A word is only decided when it has MIN_VOTES readings AND the mode is a
    strict plurality — an even split is the model genuinely disagreeing with
    itself, and calling it would invent a verdict.
    """
    votes: dict[str, Counter] = {}
    for slots in per_take.values():
        for w, s in slots.items():
            votes.setdefault(w, Counter())[s] += 1
    out = {}
    for w, c in votes.items():
        if sum(c.values()) < MIN_VOTES:
            continue
        top = c.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[w] = top[0][0]
    return out


def rank_takes(per_take: dict[str, dict[str, int]]) -> list[tuple[str, int, int]]:
    """[(take, deviations, judged)] sorted best first.

    `deviations` counts words where this take disagrees with the consensus;
    `judged` is how many words it could be scored on, which the caller needs —
    zero deviations out of one word is not the same evidence as zero out of ten.
    """
    cons = consensus(per_take)
    rows = []
    for take, slots in per_take.items():
        judged = [w for w in slots if w in cons]
        dev = sum(1 for w in judged if slots[w] != cons[w])
        rows.append((take, dev, len(judged)))
    # fewest deviations, then most words judged, then take order for determinism
    rows.sort(key=lambda r: (r[1], -r[2], r[0]))
    return rows


def pick(per_take: dict[str, dict[str, int]]) -> str | None:
    r = rank_takes(per_take)
    return r[0][0] if r else None


def status_quo(takes: list[str], seed: str) -> str:
    """A take chosen WITHOUT regard to stress — the honest comparison.

    Today's selector ranks on mos/f0/sim/pace, all of which are blind to stress
    (measured: it picks the stress-error take 5 of 8, i.e. a coin flip), so a
    seeded random draw is a fair stand-in and does not require re-running s4.
    Deterministic so the A/B page can be rebuilt without reshuffling.
    """
    return random.Random(seed).choice(sorted(takes))
