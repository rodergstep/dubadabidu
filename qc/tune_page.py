"""Listening page for a tune-lite sweep — one row per grid point, all takes.

The bake-off HTML compares ENGINES; a sweep compares configurations of one
engine, and its audio (bakeoff/tune/<variant>/<lang>/p<i>_<uid>_t<k>.wav) had
no page at all. Scores decided which reference won while the recordings that
justified it were never listenable, which is backwards for a project whose
standing rule is that the ear arbitrates.

Point indices map to grid points by qc.bakeoff._grid_points' own ordering
(sorted keys, cartesian product) — reconstructed here rather than stored, so
the page cannot disagree with what the sweep actually ran.

    python -m qc.tune_page <video> [lang] [variant]
"""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qc.bakeoff import _grid_points, _fmt_point  # noqa: E402


def build(wd: Path, lang: str, variant: str) -> Path:
    bo = wd / "bakeoff"
    tune_dir = bo / "tune" / variant / lang
    if not tune_dir.is_dir():
        raise SystemExit(f"no tune audio at {tune_dir}")

    results = json.loads((bo / f"results_{lang}.json").read_text())
    tn = (results.get("tuning") or {}).get(variant) or {}
    trials = {json.dumps(t["point"], sort_keys=True): t
              for t in tn.get("trials", [])}
    winner = json.dumps(tn.get("winner", {}), sort_keys=True)

    cfg_grid = json.loads((bo / "tune_grid.json").read_text()) \
        if (bo / "tune_grid.json").exists() else None
    if cfg_grid is None:                      # reconstruct from the trial points
        axes: dict[str, list] = {}
        for t in tn.get("trials", []):
            for k, v in t["point"].items():
                axes.setdefault(k, [])
                if v not in axes[k]:
                    axes[k].append(v)
        cfg_grid = axes
    points = _grid_points(cfg_grid)

    rows = []
    for i, pt in enumerate(points):
        key = json.dumps(pt, sort_keys=True)
        tr = trials.get(key)
        wavs = sorted(tune_dir.glob(f"p{i}_*.wav"))
        if not wavs:
            continue
        rows.append({"i": i, "point": pt, "trial": tr, "wavs": wavs,
                     "win": key == winner})
    # best first; unscored points last
    rows.sort(key=lambda r: -(r["trial"]["score"] if r["trial"] else -1))

    out = bo / f"tune_{lang}.html"
    h = [
        "<meta charset='utf-8'>",
        f"<title>tune sweep — {variant} / {lang}</title>",
        "<style>",
        "body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem}",
        "h2{margin:2rem 0 .3rem;font-size:1.05rem}",
        ".m{color:#666;font-size:.85rem;margin:0 0 .5rem}",
        ".win{background:#e8f5e9;border-left:4px solid #2e7d32;padding:.6rem .8rem}",
        ".low{color:#b26500}",
        "audio{height:32px;vertical-align:middle}",
        "table{border-collapse:collapse;margin:.5rem 0 1.5rem;width:100%}",
        "td,th{border-bottom:1px solid #eee;padding:.35rem .5rem;text-align:left;font-size:.9rem}",
        "code{background:#f5f5f5;padding:.1rem .3rem;border-radius:3px}",
        "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}",
        ".win{background:#14301a}code{background:#222}td,th{border-color:#333}",
        ".m{color:#aaa}}",
        "</style>",
        f"<h1>tune sweep — {html.escape(variant)} / {lang}</h1>",
        "<p class='m'>One block per grid point, best score first. Scores rank "
        "on 0.4*mos + 0.35*sim&rarr;real + 0.25*f0st with points under "
        "tts.min_f0st disqualified — but the scores are means over a small "
        "subset and the top few usually sit inside take-to-take noise. "
        "<b>Where they tie, your ear decides.</b></p>",
    ]
    for r in rows:
        t = r["trial"]
        cls = " class='win'" if r["win"] else ""
        tag = " &larr; sweep winner" if r["win"] else ""
        h.append(f"<div{cls}><h2>{html.escape(_fmt_point(r['point']))}{tag}</h2>")
        if t:
            low = " low" if t.get("under_floor") else ""
            floor = " (UNDER min_f0st)" if t.get("under_floor") else ""
            h.append(f"<p class='m{low}'>score {t['score']} &middot; "
                     f"sim&rarr;real {t['sim']} &middot; mos {t['mos']} "
                     f"&middot; f0st {t.get('f0st', '-')}{floor}</p>")
        else:
            h.append("<p class='m'>not scored in the stored trials</p>")
        by_seg: dict[str, list[Path]] = {}
        for w in r["wavs"]:
            by_seg.setdefault(w.stem.split("_")[1], []).append(w)
        h.append("<table><tr><th>segment</th><th>takes</th></tr>")
        for seg, ws in sorted(by_seg.items()):
            players = " ".join(
                f"<audio controls preload=none src='{w.relative_to(bo)}'></audio>"
                for w in sorted(ws))
            h.append(f"<tr><td><code>{seg}</code></td><td>{players}</td></tr>")
        h.append("</table></div>")

    out.write_text("\n".join(h), encoding="utf-8")
    return out


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "sketch60"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    variant = sys.argv[3] if len(sys.argv) > 3 else "qwen+fast"
    print(build(Path("work") / video, lang, variant))
