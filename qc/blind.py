"""Blind rating page — the input the take-ranking weights have never had.

    python -m qc.blind build sketch60 en [N] [variant ...]  # make the page
    python -m qc.blind ingest sketch60 en <export.json>    # -> ratings_<lang>.json

WHY BLIND. The listening comparisons so far were labelled, and on 2026-08-01 the
ear ranked `qwen+fast+control` first — a row that is byte-for-byte the same
CONFIG as `qwen+fast`, differing only in the random roll. Two identical
configurations came out ranked, which means labelled comparisons at this sample
size measure the dice (and any expectation the label creates), not the setting.
Shuffling and hiding the variant is the cheapest fix; it costs nothing and makes
the verdict worth storing.

WHY STORE IT. `qc/refit.py` needs ~30 (rating, metrics) pairs to re-fit the
take-ranking weights and has had ZERO since the project began, so every listen
has been discarded. The weights it would train are the ones deciding which take
ships — with take-to-take mos± at 0.24-0.49, that selection matters more than
the engine choice does.

The page is self-contained: no network, ratings held in localStorage so a
refresh doesn't lose them, and a Download button that writes the export file.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path


def _shuffled(items: list, seed: str) -> list:
    """Deterministic shuffle. A fixed order per (video, lang) means the page can
    be rebuilt or reopened without re-randomising a half-finished pass — and it
    keeps the mapping reproducible when ingest re-derives it."""
    return [x for _, x in sorted(
        (hashlib.sha1(f"{seed}{i}".encode()).hexdigest(), x)
        for i, x in enumerate(items))]


def _clips(bo: Path, lang: str, variants: list[str],
           per_group: int = 0, anchor=None) -> list[dict]:
    """Takes to rate. per_group>0 keeps only that many per (variant, segment),
    picked by the production take-ranking so the WORST roll is dropped first.

    Why not simply "keep the best": a rating set truncated to good audio is
    range-restricted, and Spearman against a truncated range is attenuated —
    refit would be learning from a sample that never shows it what bad sounds
    like. Keeping the top AND the median halves the listening while preserving
    most of the spread; only the clear duds go.

    Groups are balanced on purpose too. The raw directories hold 6 takes of one
    variant and 3 of the others, so an unbalanced sample would weight the
    ratings toward whichever variant happened to be run with more takes."""
    from pipeline.tts_engine import _take_rank
    from qc import metrics as X
    out = []
    for v in variants:
        d = bo / "seg" / v / lang
        by_seg: dict[str, list[Path]] = {}
        for w in sorted(d.glob("*_t*.wav")) if d.is_dir() else []:
            by_seg.setdefault(w.stem.split("_t")[0], []).append(w)
        for seg, ws in by_seg.items():
            if per_group and len(ws) > per_group:
                scored = []
                for w in ws:
                    m = {"mos_min": X.mos_min_window(str(w)),
                         "f0st": X.f0_semitone_std(str(w))}
                    if anchor is not None:
                        m["sim"] = X.cosine(anchor, X.ecapa_embed(str(w)))
                    scored.append((_take_rank(m), w))
                scored.sort(key=lambda x: -x[0])
                # best, then evenly spaced down the ranking (median-ish), so the
                # kept set still spans quality rather than clustering at the top
                idx = [round(i * (len(scored) - 1) / max(1, per_group - 1))
                       for i in range(per_group)]
                ws = [scored[i][1] for i in sorted(set(idx))[:per_group]]
            for w in ws:
                out.append({"variant": v, "seg": seg,
                            "path": str(w.relative_to(bo))})
    return out


def build(wd: Path, lang: str, variants: list[str],
          per_group: int = 2) -> Path:
    bo = wd / "bakeoff"
    from qc import metrics as X
    refs = sorted((wd / "qc_ua").glob("*.wav"))
    anchor = (sum(X.ecapa_embed(str(p)) for p in refs) / len(refs)
              if refs else None)
    clips = _shuffled(_clips(bo, lang, variants, per_group, anchor),
                      f"{wd.name}/{lang}")
    if not clips:
        raise SystemExit(f"no takes found for {variants}")
    # ANONYMISE THE FILES, not just the labels. Serving seg/<variant>/... would
    # put the variant in the audio src, visible on hover, right-click or view
    # source — a blind test that any curious click defeats is not blind. Takes
    # are copied to blind/<lang>/cNNN.wav so the page carries no provenance.
    import shutil
    anon = bo / "blind" / lang
    if anon.exists():
        shutil.rmtree(anon)
    anon.mkdir(parents=True)
    for i, c in enumerate(clips):
        c["key"] = f"c{i:03d}"
        dst = anon / f"{c['key']}.wav"
        shutil.copyfile(bo / c["path"], dst)
        c["src"] = str(dst.relative_to(bo))

    payload = json.dumps([{"key": c["key"], "src": c["src"]} for c in clips])
    out = bo / f"blind_{lang}.html"
    out.write_text(f"""<meta charset='utf-8'>
<title>blind rating — {lang}</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;margin:2rem auto;max-width:760px;padding:0 1rem}}
.card{{border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem 0}}
audio{{width:100%;margin:.6rem 0}}
button{{font:inherit;padding:.5rem .9rem;margin:.2rem;border-radius:8px;
  border:1px solid #bbb;background:#fafafa;cursor:pointer}}
button.sel{{background:#2e7d32;color:#fff;border-color:#2e7d32}}
#bar{{position:sticky;top:0;background:#fff;padding:.8rem 0;border-bottom:1px solid #eee}}
.muted{{color:#666;font-size:.9rem}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#eee}}
 .card{{border-color:#333}}button{{background:#222;color:#eee;border-color:#444}}
 #bar{{background:#111;border-color:#333}}.muted{{color:#aaa}}}}
</style>
<div id='bar'><b>blind rating — {lang}</b>
  <span class='muted' id='prog'></span>
  <button onclick='dl()'>Download ratings</button></div>
<p class='muted'>Which do you want to <b>ship</b>? 1 = unusable, 3 = acceptable,
5 = excellent. Variants are shuffled and hidden on purpose — a labelled pass on
2026-08-01 ranked two identical configurations apart, so labels were measuring
expectation. Ratings save as you go; reloading keeps them.</p>
<div id='list'></div>
<script>
const CLIPS = {payload};
const S = JSON.parse(localStorage.getItem('blind_{lang}') || '{{}}');
function draw() {{
  document.getElementById('list').innerHTML = CLIPS.map(c => `
    <div class='card'><audio controls preload=none src='${{c.src}}'></audio><div>
    ${{[1,2,3,4,5].map(n => `<button id='b_${{c.key}}_${{n}}'
      class='${{S[c.key]===n?'sel':''}}' onclick='rate("${{c.key}}",${{n}})'
      >${{n}}</button>`).join('')}}</div></div>`).join('');
  const done = Object.keys(S).length;
  document.getElementById('prog').textContent =
    ` ${{done}}/${{CLIPS.length}} rated`;
}}
function rate(k, n) {{
  S[k] = n; localStorage.setItem('blind_{lang}', JSON.stringify(S)); draw();
}}
function dl() {{
  const blob = new Blob([JSON.stringify(S, null, 1)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'blind_{lang}.json'; a.click();
}}
draw();
</script>""", encoding="utf-8")
    (bo / f"blind_{lang}_map.json").write_text(
        json.dumps(clips, indent=1), encoding="utf-8")
    return out


def ingest(wd: Path, lang: str, export: Path, cfg: dict) -> Path:
    """Rated keys -> ratings_<lang>.json rows carrying the metrics refit needs.

    Metrics are computed HERE rather than read from a manifest: these are
    bake-off takes, which never entered s4's manifest rows. qc_sim_cal is the
    per-reference calibrated similarity the production composite uses."""
    from qc import metrics as X
    bo = wd / "bakeoff"
    cmap = {c["key"]: c
            for c in json.loads((bo / f"blind_{lang}_map.json").read_text())}
    ratings = json.loads(export.read_text())

    refs = sorted((wd / "qc_ua").glob("*.wav"))
    anchor = sum(X.ecapa_embed(str(p)) for p in refs) / len(refs)
    band = cfg.get("qc", {}).get("sim_band", {})
    floor, ceiling = float(band.get("floor", 0.0)), float(band.get("ceiling", 1.0))

    rows = []
    for key, stars in ratings.items():
        c = cmap.get(key)
        if not c:
            continue
        w = bo / c["path"]
        if not w.exists():
            continue
        raw = X.cosine(anchor, X.ecapa_embed(str(w)))
        rows.append({
            "video": wd.name, "lang": lang,
            "id": f"{c['seg']}:{c['variant']}",
            "rating": int(stars),
            "qc_sim2": round(raw, 4),
            "qc_sim_cal": round(X.calibrate_sim(raw, floor, ceiling), 4),
            "qc_mos": round(X.mos_min_window(str(w)), 4),
            "qc_mos_min": round(X.mos_min_window(str(w)), 4),
            "qc_f0st": round(X.f0_semitone_std(str(w)), 4),
            "tempo": 1.0,     # bake-off takes are unstretched by construction
            # kept so a later analysis can ask whether the blind ear actually
            # separated the variants — the question that motivated the page
            "variant": c["variant"],
        })
    out = Path(f"ratings_{lang}.json")
    prev = json.loads(out.read_text()) if out.exists() else []
    if not isinstance(prev, list):
        raise SystemExit(f"{out} is not the flywheel format — see qc/refit.py")
    seen = {(r.get("video"), r.get("id")) for r in prev}
    fresh = [r for r in rows if (r["video"], r["id"]) not in seen]
    out.write_text(json.dumps(prev + fresh, indent=1), encoding="utf-8")
    print(f"[blind] {len(rows)} rated, {len(fresh)} new -> {out} "
          f"({len(prev) + len(fresh)} total; refit wants ~30)")
    return out


if __name__ == "__main__":
    import yaml
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    video = sys.argv[2] if len(sys.argv) > 2 else "sketch60"
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    wd = Path("work") / video
    if mode == "build":
        args = sys.argv[4:]
        per = 2
        if args and args[0].isdigit():        # build <video> <lang> <N> [variants]
            per, args = int(args[0]), args[1:]
        vs = args or ["qwen+fast", "qwen+fast+control"]
        print(build(wd, lang, vs, per))
    else:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        ingest(wd, lang, Path(sys.argv[4]), cfg)
