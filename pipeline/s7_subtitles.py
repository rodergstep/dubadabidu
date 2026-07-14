"""s7: per-language SRT (plus Ukrainian) from the manifest, lines wrapped <=42
chars, with a reading-speed floor so cues timed to fast dub segments don't flash
by faster than they can be read."""
from __future__ import annotations
import datetime as dt
import logging
import srt
from . import manifest as M

log = logging.getLogger("dubadabidu.s7")

MAX_CPS = 17.0        # chars/second reading-speed ceiling (BBC/Netflix ~17)
MIN_CUE_S = 1.0       # a cue never shows for less than this
CUE_GAP_S = 0.08      # kept between a stretched cue and the next one


def _wrap(text: str, width: int = 42, max_lines: int = 2) -> str:
    """Greedy word-wrap to <=width. Previously fell back to the RAW unwrapped
    string when the text exceeded max_lines — rendering a long cue as one
    overflowing line. Now it always returns the wrapped form; if it needs more
    than max_lines (a too-long segment) it still wraps rather than overflow, and
    logs so the upstream merge can be tuned."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        log.debug("cue exceeds %d lines (%d): %r", max_lines, len(lines), text[:60])
    return "\n".join(lines)


def _read_floor(start: float, end: float, text: str,
                next_start: float | None) -> float:
    """Extend a cue's end so it stays on screen long enough to read (MAX_CPS
    and MIN_CUE_S), but never past the next cue (minus a small gap) and never
    shorter than it already is."""
    n = len(text.replace("\n", " "))
    need = max(MIN_CUE_S, n / MAX_CPS)
    desired = max(end, start + need)
    if next_start is not None:
        desired = min(desired, next_start - CUE_GAP_S)
    return max(end, desired)


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    us = man["utterances"]
    for lang in langs + [cfg["source_language"]]:
        subs = []
        for i, u in enumerate(us, 1):
            # fitted_text = what the dub actually says (may be a shorter variant);
            # subtitles must not contradict the audio. Timing follows the
            # PLACED dub (s5 soft-anchor timeline), not the source utterance —
            # otherwise subs vanish while the voice is still speaking.
            if lang == cfg["source_language"]:
                text, start, end = u["text_uk"], u["start"], u["end"]
                nxt = us[i]["start"] if i < len(us) else None
            else:
                tr = u["tr"][lang]
                text = tr.get("fitted_text", tr["text"])
                start = tr.get("placed_start", u["start"])
                end = tr.get("placed_end", u["end"])
                nxt = (us[i]["tr"][lang].get("placed_start", us[i]["start"])
                       if i < len(us) else None)
            end = _read_floor(start, end, text, nxt)
            subs.append(srt.Subtitle(
                index=i,
                start=dt.timedelta(seconds=start),
                end=dt.timedelta(seconds=end),
                content=_wrap(text)))
        out = wd / f"subs_{lang}.srt"
        out.write_text(srt.compose(subs), encoding="utf-8")
        print(f"[s7] {out}")
