"""Review page: work/<video>/review_<lang>.html — self-contained, offline.

Per segment: dubbed audio + the real UA slice side by side, metric badges,
and a 1..5 naturalness rating (persisted in localStorage, exportable as JSON).
The exported ratings calibrate composite-score weights against your ear.
Segments are sorted worst-first by qc_score when evaluate has run.
"""
from __future__ import annotations
import hashlib
import html
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402

CSS = """
body{font:14px/1.5 -apple-system,sans-serif;margin:2rem auto;max-width:960px;
     background:#111;color:#ddd}
h1{font-size:1.2rem} .cal{color:#8ab}
.seg{border:1px solid #333;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0}
.seg.flagged{border-color:#a53}
.row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
.badge{background:#222;border:1px solid #444;border-radius:4px;padding:0 .4rem}
.badge.bad{border-color:#a53;color:#fa8}
audio{height:2rem;max-width:260px} .txt{color:#aaa;margin:.3rem 0}
.stars button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;
  padding:.1rem .5rem;cursor:pointer} .stars button.on{background:#365;color:#fff}
#export{position:fixed;top:1rem;right:1rem;background:#365;color:#fff;
  border:none;border-radius:6px;padding:.4rem .8rem;cursor:pointer}
"""

JS = """
const KEY='dubadabidu_ratings_'+document.body.dataset.key;
const ratings=JSON.parse(localStorage.getItem(KEY)||'{}');
document.querySelectorAll('.stars').forEach(w=>{
  const id=w.dataset.id;
  w.querySelectorAll('button').forEach(b=>{
    if(ratings[id]==+b.textContent)b.classList.add('on');
    b.onclick=()=>{ratings[id]=+b.textContent;
      localStorage.setItem(KEY,JSON.stringify(ratings));
      w.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));};
  });
});
document.getElementById('export').onclick=()=>{
  const blob=new Blob([JSON.stringify({key:document.body.dataset.key,
    ratings},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='ratings_'+document.body.dataset.key+'.json';a.click();};
"""


def _ua_slice(wd: Path, u: dict) -> Path:
    """Cut the real-voice slice for side-by-side listening (cached)."""
    import soundfile as sf
    out_dir = wd / "qc_ua"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{u['id']}.wav"
    if not out.exists():
        info = sf.info(str(wd / "vocals.wav"))
        start = int(u["start"] * info.samplerate)
        stop = min(int(u["end"] * info.samplerate), info.frames)
        data, sr = sf.read(str(wd / "vocals.wav"), start=start, stop=stop)
        sf.write(str(out), data, sr)
    return out


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    flag_score = cfg["qc"].get("eval", {}).get("score_flag", 0.55)

    for lang in langs:
        cal = man.get("qc_calibration", {}).get(lang)
        us = sorted(man["utterances"],
                    key=lambda u: u["tr"][lang].get("qc_score", 1.0))
        segs = []
        for u in us:
            tr = u["tr"][lang]
            ua = _ua_slice(wd, u).relative_to(wd)
            score = tr.get("qc_score")
            flagged = score is not None and score < flag_score
            badges = "".join(
                f'<span class="badge{" bad" if bad else ""}">{k} {v}</span>'
                for k, v, bad in [
                    ("score", score, flagged),
                    ("sim", tr.get("qc_sim_cal"), (tr.get("qc_sim_cal") or 1) < 0.5),
                    ("mos", tr.get("qc_mos"), (tr.get("qc_mos") or 5) < 3.5),
                    ("tempo", tr.get("tempo"), (tr.get("tempo") or 1) > 1.06),
                    ("fit", tr.get("fit"), tr.get("fit") == "overflow"),
                ] if v is not None)
            stars = "".join(f"<button>{n}</button>" for n in range(1, 6))
            segs.append(f"""
<div class="seg{' flagged' if flagged else ''}">
 <div class="row"><b>{u['id']}</b> <span>{u['start']:.1f}–{u['end']:.1f}s</span>
   {badges}
   <span class="stars" data-id="{u['id']}">rate: {stars}</span></div>
 <div class="txt">dub: {html.escape(tr.get('fitted_text', tr.get('text', '')))}</div>
 <div class="txt">uk:&nbsp; {html.escape(u['text_uk'])}</div>
 <div class="row">
   <label>dub <audio controls preload="none" src="{tr.get('placed', tr['fitted'])}"></audio></label>
   <label>you <audio controls preload="none" src="{ua}"></audio></label>
 </div>
</div>""")
        cal_line = (f'floor {cal["floor"]} / ceiling {cal["ceiling"]} (ref {cal["ref"]})'
                    if cal else "run `dubadabidu evaluate` for metrics")
        # ratings key includes a segmentation hash: re-segmenting the video must
        # not resurrect ratings that belonged to different utterance boundaries
        seg_hash = hashlib.sha1(
            ",".join(u["id"] + str(u["start"]) for u in us).encode()).hexdigest()[:6]
        page = (f"<!doctype html><meta charset='utf-8'>"
                f"<title>review {Path(video).stem} {lang}</title>"
                f"<style>{CSS}</style><body data-key="
                f"{json.dumps(f'{Path(video).stem}_{lang}_{seg_hash}')}>"
                f"<button id='export'>Export ratings JSON</button>"
                f"<h1>{Path(video).stem} — {lang} ({len(us)} segments, worst first)</h1>"
                f"<div class='cal'>calibration: {cal_line}</div>"
                f"{''.join(segs)}<script>{JS}</script>")
        out = wd / f"review_{lang}.html"
        out.write_text(page, encoding="utf-8")
        print(f"[review] open {out}")
