"""Generate the AVOIDED translation of every segment the stress lexicon touches.

    python -m qc.avoidance_ab <video> <lang> [--apply]

WHY THIS IS A SEPARATE STEP. `translation.avoid_mis_stressed` asks s3 to prefer
a synonym for words a native listener marked as mis-stressed (FINDINGS 2.1k),
and it shipped UNMEASURED: nobody has checked whether the listener prefers a
correctly-stressed synonym to a mis-stressed exact word. That is a question for
the ear, and answering it needs BOTH texts for the same segment so they can be
synthesized and compared.

s3 cannot provide both. It skips segments that already have text, and re-running
it with --force would destroy the shipped translation to produce the
alternative. So this writes the alternative ALONGSIDE, into

    tr[lang]["text_variants"]["avoided"]

leaving tr[lang]["text"] exactly as it is.

WHY THE KEY IS SCOPED TO THE BAKE-OFF. `bakeoff.text_variant` — not
`tts.text_variant`. A tts.* key would look like it changed what production
synthesizes while only the bake-off read it, which is the failure this project
keeps finding: qc.eval.weights ranked nothing for weeks, tts.engine routed
nothing, `tempo` means two things. Naming it after its only consumer makes the
scope legible.

COST: one LLM call per batch of affected segments. No GPU — synthesis happens
later, on the pod, from the manifest.
"""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402
from pipeline import s3_translate as S3  # noqa: E402
from qc.stress_words import load_avoid_list  # noqa: E402

log = logging.getLogger("dubadabidu.qc.avoidance_ab")

VARIANT = "avoided"


def affected(man: dict, lang: str, words: list[str]) -> list[dict]:
    """Segments whose shipped text contains a marked word (whole-word match).

    Whole-word on purpose: `цветами` contains `цвета` as a substring, and a
    naive check reported it as a hit in the first pass of this analysis.
    """
    import re
    if not words:
        return []
    pat = re.compile(r"(?<![^\W\d_])(?:" +
                     "|".join(re.escape(w) for w in words) +
                     r")(?![^\W\d_])", re.UNICODE | re.IGNORECASE)
    out = []
    for u in man["utterances"]:
        tr = u["tr"].get(lang) or {}
        text = tr.get("fitted_text") or tr.get("text") or ""
        if text and pat.search(text):
            out.append(u)
    return out


def run(cfg: dict, video: str, lang: str, apply: bool = False) -> int:
    words = [w for w, _ in load_avoid_list(lang)]
    if not words:
        raise SystemExit(f"stress_lexicon_{lang}.json is empty — mark words on "
                         f"a review page first, then ingest with `verdicts`.")
    man = M.load(cfg, video)
    segs = affected(man, lang, words)
    print(f"[avoidance] {len(words)} marked word(s); {len(segs)} of "
          f"{len(man['utterances'])} segments contain one")
    if not segs:
        return 0

    tcfg = cfg["translation"]
    key = os.environ.get(tcfg["api_key_env"])
    if not key:
        raise SystemExit(f"set {tcfg['api_key_env']} — this step re-translates.")
    from openai import OpenAI
    client = OpenAI(base_url=tcfg["base_url"], api_key=key)

    # The SAME prompt s3 builds, so the alternative is what production would
    # actually have produced — not a differently-worded request that happens to
    # avoid the word. Glossary and terms included; the avoid block is the axis.
    gloss = S3._glossary(lang)
    terms_p = M.video_workdir(cfg, video) / f"terms_{lang}.json"
    terms_s = ""
    if terms_p.exists():
        terms = json.loads(terms_p.read_text(encoding="utf-8"))["terms"]
        terms_s = ("Video-specific terminology (Ukrainian -> target, mandatory):\n"
                   + "\n".join(f"  {t['uk']} -> {t['tr']}" for t in terms))
    avoid = S3._avoid_block(cfg, lang, gloss + terms_s)
    if not avoid:
        raise SystemExit("translation.avoid_mis_stressed is off — turn it on, "
                         "or there is no alternative to generate.")
    shared = "\n\n".join(x for x in [gloss, terms_s, avoid] if x)
    sysmsg = (S3._prompt(lang, tcfg["isometric_tolerance"],
                         tcfg["n_short_variants"]) + "\n\n" + shared)
    payload = [{"id": u["id"], "uk": u["text_uk"], "chars": len(u["text_uk"]),
                "seconds": round(u["end"] - u["start"], 1)} for u in segs]
    data = S3._chat(client, tcfg, sysmsg,
                    json.dumps({"segments": payload}, ensure_ascii=False))
    by_id = {s["id"]: s for s in data.get("segments", [])}

    changed = 0
    for u in segs:
        r = by_id.get(u["id"])
        if not r:
            log.warning("%s: no alternative returned", u["id"])
            continue
        tr = u["tr"][lang]
        old = tr.get("fitted_text") or tr["text"]
        new = r["text"].strip()
        same = new == old
        print(f"\n  {u['id']}{'  (UNCHANGED)' if same else ''}")
        print(f"    shipped: {old[:110]}")
        if not same:
            print(f"    avoided: {new[:110]}")
        if same:
            continue
        changed += 1
        if apply:
            tr.setdefault("text_variants", {})[VARIANT] = new
    if apply:
        M.save(cfg, video, man)
        print(f"\n[avoidance] wrote {changed} alternative(s) into "
              f"tr.{lang}.text_variants.{VARIANT}")
        print(f"[avoidance] synthesize them with `bakeoff.text_variant: "
              f"{VARIANT}` and compare by ear — no metric here can hear stress.")
    else:
        print(f"\n[avoidance] DRY RUN — {changed} segment(s) would change. "
              f"Re-run with --apply to write them.")
    return changed


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit("usage: python -m qc.avoidance_ab <video> <lang> "
                         "[--apply]")
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    sys.exit(0 if run(cfg, args[0], args[1], "--apply" in sys.argv) >= 0 else 1)
