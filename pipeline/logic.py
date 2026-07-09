"""Pure functions: whisper-segment merging (s2) and fit decision (s5).
Kept dependency-free so unit tests run without GPU/models."""
from __future__ import annotations

SENT_END = (".", "!", "?", "…")


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge for config overlays (config.gpu.yaml over
    config.yaml). Overlay scalars/lists replace; overlay dicts merge."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def split_at_pauses(words: list[dict], max_gap: float,
                    min_words: int = 2, sent_gap: float = 0.25) -> list[dict]:
    """words: [{'start','end','word'}] -> [{'start','end','text'}] split where
    the inter-word gap exceeds max_gap, or exceeds sent_gap right after a
    sentence-ending word (translation-safe boundary, so a lower bar).

    Whisper emits slow, pause-rich narration as one long segment; a dub spoken
    fluently into that slot leaves seconds of dead air and loses the speaker's
    rhythm. Splitting at real pauses lets each part keep its own timestamp.
    Splits shorter than min_words are glued to the previous part (a lone word
    after a pause is usually a hesitation, not a sentence).
    """
    parts, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > (
                sent_gap if cur[-1]["word"].rstrip().endswith(SENT_END)
                else max_gap):
            if len(cur) >= min_words or not parts:
                parts.append(cur)
            else:
                parts[-1].extend(cur)
            cur = []
        cur.append(w)
    if cur:
        if len(cur) >= min_words or not parts:
            parts.append(cur)
        else:
            parts[-1].extend(cur)
    return [{"start": p[0]["start"], "end": p[-1]["end"],
             "text": "".join(w["word"] for w in p).strip()} for p in parts]


def merge_segments(segs: list[dict], max_chars: int, max_seconds: float,
                   gap_break: float = 0.35) -> list[dict]:
    """segs: [{'start','end','text'}] -> merged sentence-level utterances."""
    merged, cur = [], None
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": t}
            continue
        too_long = (len(cur["text"]) + len(t) + 1 > max_chars
                    or s["end"] - cur["start"] > max_seconds)
        sentence_done = cur["text"].endswith(SENT_END)
        if too_long or (sentence_done and s["start"] - cur["end"] > gap_break):
            merged.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": t}
        else:
            cur["end"] = s["end"]
            cur["text"] += " " + t
    if cur:
        merged.append(cur)
    return merged


def choose_placement(durations: list[float], slot: float, max_tempo: float,
                     soft_tempo: float) -> tuple[int, str, float]:
    """Pick the candidate (primary first, then shorter variants) that fits with
    the least audible time-stretching. -> (index, verdict, tempo).

    Ladder: primary needing <= soft_tempo (as-is or mild stretch)
            -> first variant fitting as-is
            -> candidate needing the least stretch, if <= max_tempo
            -> overflow (last candidate at max_tempo).
    Old first-fit behavior stretched the primary up to max_tempo even when a
    variant fit untouched; atempo above ~1.06 is audibly phasey on speech.
    """
    if slot <= 0:
        return len(durations) - 1, "no", max_tempo
    tempos = [d / slot for d in durations]
    if tempos[0] <= soft_tempo:
        return 0, ("as_is" if tempos[0] <= 1.0 else "stretch"), max(tempos[0], 1.0)
    for i, t in enumerate(tempos[1:], 1):
        if t <= 1.0:
            return i, "as_is", 1.0
    best = min(range(len(tempos)), key=lambda i: tempos[i])
    if tempos[best] <= max_tempo:
        return best, "stretch", tempos[best]
    return len(durations) - 1, "no", max_tempo


def decide_fit(duration: float, slot: float, max_tempo: float) -> tuple[str, float]:
    """-> (verdict, tempo). verdict: 'as_is' | 'stretch' | 'no'."""
    if slot <= 0:
        return "no", max_tempo
    tempo = duration / slot
    if tempo <= 1.0:
        return "as_is", 1.0
    if tempo <= max_tempo:
        return "stretch", tempo
    return "no", tempo
