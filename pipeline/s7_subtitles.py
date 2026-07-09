"""s7: per-language SRT (plus Ukrainian) from the manifest, lines wrapped <=42 chars."""
from __future__ import annotations
import datetime as dt
import srt
from . import manifest as M


def _wrap(text: str, width: int = 42, max_lines: int = 2) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:max_lines]) if len(lines) <= max_lines else text


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    for lang in langs + [cfg["source_language"]]:
        subs = []
        for i, u in enumerate(man["utterances"], 1):
            # fitted_text = what the dub actually says (may be a shorter variant);
            # subtitles must not contradict the audio
            text = u["text_uk"] if lang == cfg["source_language"] \
                else u["tr"][lang].get("fitted_text", u["tr"][lang]["text"])
            subs.append(srt.Subtitle(
                index=i,
                start=dt.timedelta(seconds=u["start"]),
                end=dt.timedelta(seconds=u["end"]),
                content=_wrap(text)))
        out = wd / f"subs_{lang}.srt"
        out.write_text(srt.compose(subs), encoding="utf-8")
        print(f"[s7] {out}")
