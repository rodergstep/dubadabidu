"""Within-sentence comparison page — pick the best of N versions of ONE line.

WHY THIS EXISTS, and why it replaces absolute rating for take selection.

qc/blind.py asks for an absolute 1-5 on each clip, with every clip a DIFFERENT
sentence. The listener's objection (2026-08-03) was that this is very hard: "if
it will be one sentence in different variation, it will be easier to choose the
best one... especially when track up to five seconds". He is describing a
measurement problem, not a UI preference.

An absolute score on differing content confounds two variables — how good the
take is, and how hard that sentence was. A 1.4 s fragment and a 14.9 s sentence
are not on the same scale, and half the clips in the last set were under 4 s.
That variance lands in the ratings as noise, and it is the most likely reason
refit's cross-validated rho fell to +0.119 on 112 en ratings while the in-sample
fit reached +0.293: the model was being asked to predict content difficulty.

Holding the sentence constant removes that variable. Comparative judgement is
also simply more sensitive than absolute — the same reason the bake-off's own
noise-floor work compares runs rather than scoring them.

WHAT IT PRODUCES. Per segment: a winner, and optionally clips marked unusable.
That is an ordering within a group, which converts to (rating, metrics) pairs
for refit WITHOUT the content confound — the winner scores above its siblings by
construction, and every sibling shares a sentence.

Blind by the same rule as qc/blind: files are copied to anonymous names, because
a variant visible in a src attribute is not a blind test.
"""
from __future__ import annotations
import base64
import hashlib
import json
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("dubadabidu.qc.compare")

# Below this a take is too short to judge — the listener's own threshold. Short
# clips are not merely harder, they are where absolute rating went wrong.
MIN_SECONDS = 5.0


def _dur(p: Path) -> float:
    import soundfile as sf
    try:
        i = sf.info(str(p))
        return i.frames / i.samplerate
    except Exception:
        return 0.0


def _groups(bo: Path, lang: str, variants: list[str],
            min_seconds: float, takes_per_variant: int = 1) -> list[dict]:
    """One group per segment: `takes_per_variant` take(s) of each, shuffled.

    ONE take per variant by default, and that default matters. Every take of
    every variant produced 4 clips per sentence for a 2-arm test, and the
    listener's objection to the 40-track page was precisely this: more clips per
    decision is harder, not more informative. The question a comparison page
    asks is "which VARIANT ships", so a second take of the same variant adds
    take-to-take noise to a variant-level judgement — the listener ends up
    resolving a 4-way with two near-duplicates instead of an A/B.

    Take 0 for every variant, never the highest-ranked one. The rank comes from
    the take-selection objective, which correlates +0.022 with this listener
    (FINDINGS 2.1) — picking each variant's "best" by a metric he does not agree
    with would quietly bias the comparison. The same positional rule applied to
    both arms cannot favour either.
    """
    by_seg: dict[str, list[Path]] = {}
    for v in variants:
        d = bo / "seg" / v / lang
        takes = sorted(d.glob("*_t*.wav")) if d.is_dir() else []
        per_seg: dict[str, list[Path]] = {}
        for w in takes:
            per_seg.setdefault(w.stem.split("_t")[0], []).append(w)
        for seg, ts in per_seg.items():
            by_seg.setdefault(seg, []).extend(ts[:max(1, takes_per_variant)])
    out = []
    for seg, takes in by_seg.items():
        dur = sum(_dur(t) for t in takes) / max(len(takes), 1)
        if dur < min_seconds or len(takes) < 2:
            continue
        # deterministic shuffle: re-running must not reshuffle a half-done page
        rnd = random.Random(hashlib.sha1(f"{lang}/{seg}".encode()).hexdigest())
        takes = takes[:]
        rnd.shuffle(takes)
        out.append({"seg": seg, "dur": round(dur, 1), "takes": takes})
    # longest first — easiest to judge, and the listener stops when tired
    out.sort(key=lambda g: -g["dur"])
    return out


def build(wd: Path, lang: str, variants: list[str],
          min_seconds: float = MIN_SECONDS, embed: bool = True,
          takes_per_variant: int = 1, max_groups: int | None = None,
          axis: str = "overall quality — the one you would ship",
          skip_groups: int = 0) -> Path:
    """`axis` is printed ON the page, and it is not decoration.

    2026-08-09: a page was built to settle whether ICL's flatter delivery was
    acceptable. The page said "click best on the one you would ship" and the
    monotony instruction was given in chat. The listener rated STRESS — "I rate
    the stressing, not the quality of the track... sometimes stress was okay but
    the whole quality of the voice was worse". Perfectly reasonable, and it
    answered a different question than the one the run was paid for. The axis
    has to travel WITH the page, because that is the only thing in front of the
    listener while they judge.
    """
    bo = wd / "bakeoff"
    groups = _groups(bo, lang, variants, min_seconds, takes_per_variant)
    # skip_groups continues an earlier page instead of re-asking what was already
    # answered. Groups are sorted longest-first and that order is deterministic,
    # so skip_groups=N picks up exactly where a max_groups=N page stopped. The
    # localStorage key is derived from the clip layout, so the continuation gets
    # its own store and cannot inherit the first page's answers.
    if skip_groups:
        groups = groups[skip_groups:]
    if max_groups:
        # groups are longest-first, so a cap keeps the most judgeable ones
        dropped = len(groups) - max_groups
        groups = groups[:max_groups]
        if dropped > 0:
            log.info("capped at %d group(s); %d shorter one(s) dropped",
                     max_groups, dropped)
    if not groups:
        raise SystemExit(
            f"no segment of {variants} in {lang} has >= {min_seconds}s of audio "
            f"and 2+ takes — raise bakeoff.subset_size or lower min_seconds")
    anon = bo / "compare" / lang
    if anon.exists():
        shutil.rmtree(anon)
    anon.mkdir(parents=True)

    tmp = Path(tempfile.mkdtemp())
    payload, truth = [], {}
    for gi, g in enumerate(groups):
        items = []
        for ti, src in enumerate(g["takes"]):
            key = f"g{gi:02d}c{ti}"
            truth[key] = {"seg": g["seg"], "variant": src.parent.parent.name,
                          "take": src.name, "path": str(src.relative_to(bo))}
            if embed:
                m4a = tmp / f"{key}.m4a"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-i", str(src), "-c:a", "aac", "-b:a", "128k",
                                str(m4a)], check=True)
                url = ("data:audio/mp4;base64,"
                       + base64.b64encode(m4a.read_bytes()).decode())
            else:
                shutil.copyfile(src, anon / f"{key}.wav")
                url = f"compare/{lang}/{key}.wav"
            items.append({"key": key, "src": url})
        payload.append({"g": f"g{gi:02d}", "dur": g["dur"], "items": items})

    # the un-blinding map stays OUT of the page
    (bo / f"compare_{lang}_truth.json").write_text(
        json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")

    # localStorage is namespaced by the BUILD, not just the language. It was
    # keyed "cmp_<lang>", one key shared by every ru page ever built, so a new
    # page loaded the previous page's answers and the download merged both. On
    # 2026-08-09 a 15x2 page came back carrying 64 marks for clips that did not
    # exist on it — the stress page's ratings, byte-identical. Silent, and it
    # would have contaminated the tally had the keys happened to line up.
    # Hash the TRUTH map (segment + variant + take + axis), not the clip keys.
    # The keys are g00c0, g00c1, ... — identical for ANY page with the same
    # shape, so a 12x2 page always produced the same id no matter what audio was
    # in it, and the previous 12x2 page's answers loaded straight into the next
    # one. That is the bug this whole mechanism was added to prevent, shipped
    # with a hash that could not see content. The first test only covered pages
    # of different SHAPE, which is exactly the case that already worked.
    build_id = hashlib.sha1(
        json.dumps({"axis": axis, "truth": truth},
                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    out = bo / f"compare_{lang}.html"
    out.write_text(_HTML.replace("__LANG__", lang)
                   .replace("__BUILD__", "_" + build_id)
                   .replace("__AXIS__", axis)
                   .replace("__DATA__", json.dumps(payload)), encoding="utf-8")
    (bo / f"compare_{lang}_truth.json").write_text(
        json.dumps({"_build": build_id, "_axis": axis, **truth},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%d group(s), %d clips -> %s", len(payload),
             sum(len(g["items"]) for g in payload), out)
    return out


_HTML = """<meta charset='utf-8'>
<title>pick the best — __LANG__</title>
<style>
 :root{color-scheme:light dark}
 body{font:16px/1.6 system-ui,sans-serif;margin:0 auto;max-width:820px;padding:1rem}
 #bar{position:sticky;top:0;padding:.7rem 0;backdrop-filter:blur(8px);
   border-bottom:1px solid color-mix(in srgb,currentColor 20%,transparent);z-index:9}
 .g{border:1px solid color-mix(in srgb,currentColor 22%,transparent);
   border-radius:12px;padding:1rem;margin:1.1rem 0}
 .row{display:flex;align-items:center;gap:.6rem;margin:.45rem 0;flex-wrap:wrap}
 audio{flex:1 1 320px;height:34px;min-width:240px}
 button{font:inherit;padding:.35rem .8rem;border-radius:8px;cursor:pointer;
   border:1px solid color-mix(in srgb,currentColor 35%,transparent);
   background:transparent;color:inherit}
 button.best{background:#1a7f37;color:#fff;border-color:#1a7f37}
 button.bad{background:#c9341a;color:#fff;border-color:#c9341a}
 .muted{opacity:.7;font-size:.9rem}
 h3{margin:.1rem 0 .5rem;font-size:1rem}
</style>
<div id='bar'><b>pick the best — __LANG__</b> <span class='muted' id='prog'></span>
 <button onclick='dl()'>Download</button></div>
<p class='muted'><b>Judge: __AXIS__.</b> Each block is the SAME sentence in
several versions, shuffled and unlabelled. Click <b>best</b> on the one that wins
<i>on that axis</i>, even if it loses on others. Click <b>unusable</b> on any that
are broken. Skip a block if you cannot tell them apart — that is a real answer and
better than a guess.</p>
<div id='list'></div>
<script>
const G = __DATA__;
const S = JSON.parse(localStorage.getItem('cmp___LANG____BUILD__') || '{}');
function draw(){
  document.getElementById('list').innerHTML = G.map(g => `
    <div class='g'><h3>${g.g} &middot; ${g.dur}s</h3>
    ${g.items.map(it => `<div class='row'>
      <audio controls preload='none' src='${it.src}'></audio>
      <button id='b_${it.key}' class='${(S[g.g]||{}).best===it.key?'best':''}'
        onclick='best("${g.g}","${it.key}")'>best</button>
      <button id='x_${it.key}' class='${((S[g.g]||{}).bad||[]).includes(it.key)?'bad':''}'
        onclick='bad("${g.g}","${it.key}")'>unusable</button>
    </div>`).join('')}</div>`).join('');
  document.getElementById('prog').textContent =
    ` ${Object.keys(S).filter(k=>S[k].best).length}/${G.length} decided`;
}
function best(g,k){ S[g]=S[g]||{}; S[g].best = S[g].best===k?null:k; save(); }
function bad(g,k){ S[g]=S[g]||{}; const b=new Set(S[g].bad||[]);
  b.has(k)?b.delete(k):b.add(k); S[g].bad=[...b]; save(); }
function save(){ localStorage.setItem('cmp___LANG____BUILD__', JSON.stringify(S)); draw(); }
function dl(){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(S,null,1)],
    {type:'application/json'}));
  a.download='compare___LANG__.json'; a.click();
}
draw();
</script>
"""


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 4:
        raise SystemExit("usage: python -m qc.compare <video-stem> <lang> "
                         "<variant> [variant ...]")
    wd = Path("work") / sys.argv[1]
    print(build(wd, sys.argv[2], sys.argv[3:]))
