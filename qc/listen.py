"""Side-by-side listening page for engine VARIANTS, with every take.

    python -m qc.listen <video> [lang] [variant ...]

bakeoff_<lang>.html shows ONE take per variant per segment. That is fine for a
glance and misleading for a decision: take-to-take mos± runs 0.24-0.49 here, so
a single take is a coin flip and two variants can trade places purely on which
roll got rendered. Every take goes on this page instead, grouped by segment so
the same words are compared across variants.

Metrics come from results_<lang>.json, with the measured run-to-run noise floor
printed beside them — a difference smaller than that band is not a difference.
"""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path

# Measured 2026-08-01 by running an identical config twice (qwen+fast vs
# qwen+fast+control). f0st is the outlier: it swung 0.438 with NOTHING changed,
# so it cannot rank anything at this sample size — shown, but struck through.
NOISE = {"sim": 0.010, "mos": 0.007, "wer": 0.005, "f0": 0.438}


def build(wd: Path, lang: str, variants: list[str]) -> Path:
    bo = wd / "bakeoff"
    res = json.loads((bo / f"results_{lang}.json").read_text())
    engines = res.get("engines", {})

    seg_takes: dict[str, dict[str, list[Path]]] = {}
    for v in variants:
        d = bo / "seg" / v / lang
        for w in sorted(d.glob("*_t*.wav")) if d.is_dir() else []:
            sid = w.stem.split("_t")[0]
            seg_takes.setdefault(sid, {}).setdefault(v, []).append(w)
    if not seg_takes:
        raise SystemExit(f"no audio for {variants} under {bo}/seg")

    h = ["<meta charset='utf-8'>", f"<title>listen — {lang}</title>", "<style>",
         "body{font:15px/1.6 system-ui,sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem}",
         "table{border-collapse:collapse;width:100%;margin:.5rem 0 2rem}",
         "td,th{border-bottom:1px solid #e5e5e5;padding:.45rem .6rem;text-align:left;vertical-align:top}",
         "th{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:#666}",
         "audio{height:30px;display:block;margin:.15rem 0}",
         ".v{font-weight:600;white-space:nowrap}",
         ".m{color:#666;font-size:.82rem;font-weight:400}",
         "s{opacity:.45}",
         ".note{background:#fff8e1;border-left:4px solid #f9a825;padding:.7rem 1rem;margin:1rem 0}",
         "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}",
         "td,th{border-color:#333}.m{color:#aaa}.note{background:#2a2410}}",
         "</style>", f"<h1>listen — {lang}</h1>",
         "<div class='note'><b>Measured run-to-run noise floor</b> (identical "
         "config run twice): sim &plusmn;0.010 &middot; mos &plusmn;0.007 "
         "&middot; wer &plusmn;0.005 &middot; <b>f0st &plusmn;0.438</b>. "
         "Anything smaller than these is not a difference. f0st cannot rank "
         "anything at this sample size, so it is struck through below.</div>",
         "<table><tr><th>variant</th><th>metrics</th></tr>"]
    for v in variants:
        e = engines.get(v, {})
        f0 = e.get("f0")
        h.append(
            f"<tr><td class='v'>{html.escape(v)}</td><td class='m'>"
            f"sim {e.get('sim','-')} &middot; mos {e.get('mos','-')} &middot; "
            f"wer {e.get('wer','-')} &middot; <s>f0st {f0 if f0 else '-'}</s>"
            f" &middot; mos&plusmn; {e.get('mos_sd','-')} &middot; "
            f"{e.get('s_take','-')} s/take</td></tr>")
    h.append("</table>")

    for sid in sorted(seg_takes):
        h.append(f"<h2>{sid}</h2><table><tr><th>variant</th><th>takes</th></tr>")
        for v in variants:
            ws = seg_takes[sid].get(v) or []
            if not ws:
                continue
            players = "".join(
                f"<audio controls preload=none src='{w.relative_to(bo)}'></audio>"
                for w in ws)
            h.append(f"<tr><td class='v'>{html.escape(v)}</td>"
                     f"<td>{players}</td></tr>")
        h.append("</table>")

    out = bo / f"listen_{lang}.html"
    out.write_text("\n".join(h), encoding="utf-8")
    return out


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "sketch60"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    vs = sys.argv[3:] or ["qwen+fast", "qwen+fast+0.6B", "qwen+fast+control"]
    print(build(Path("work") / video, lang, vs))
