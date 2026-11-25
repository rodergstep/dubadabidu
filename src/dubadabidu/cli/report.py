# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu report`: per-language fit / QC summary table."""
from __future__ import annotations

from dubadabidu.core import manifest as M


def report(cfg: dict, video: str) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    print(f"\n{video} — {len(man['utterances'])} utterances, "
          f"{man['duration']:.0f}s | stages: {man.get('stages', {})}")
    langs = sorted({l for u in man["utterances"] for l in u["tr"]})
    stale_note = []
    score_flag = cfg["qc"].get("eval", {}).get("score_flag", 0.55)
    adeq_flag = cfg["translation"].get("adequacy_flag", 3)
    hdr = f"{'lang':5} {'engine':>10} {'ok':>4} {'stretch':>8} {'shorten':>8} " \
          f"{'overflow':>9} {'overlap':>8} {'wer>thr':>8} {'score<f':>8} {'adeq<f':>7}"
    print(hdr); print("-" * len(hdr))
    for lang in langs:
        trs = [u["tr"].get(lang, {}) for u in man["utterances"]]
        st = M.stale_qc(wd, man, lang)
        if st["score"] or st["wer"]:
            stale_note.append(f"{lang} ({len(st['score'])} score, "
                              f"{len(st['wer'])} wer)")
        engine = "⚠EDGE" if M.edge_fallback_langs(man, [lang], cfg["tts"]) else \
            next((t["synth_engine"] for t in trs if t.get("synth_engine")), "-")
        fits = [t.get("fit") for t in trs]
        wer_bad = sum(1 for t in trs
                      if t.get("qc_wer", 0) > cfg["qc"]["wer_flag_threshold"])
        score_bad = sum(1 for t in trs if "qc_score" in t
                        and t["qc_score"] < score_flag)
        # translation faithfulness flags (s3 adequacy judge), if scored
        adeq_bad = sum(1 for t in trs if "adequacy" in t
                       and t["adequacy"].get("score", 5) < adeq_flag)
        # drift_exceeded superseded overrun_s when s5 went soft-anchor
        # (overlap in the mix is impossible now); old manifests keep overrun_s
        overlaps = sum(1 for t in trs
                       if t.get("overrun_s") or t.get("drift_exceeded"))
        print(f"{lang:5} {engine:>10} {fits.count('ok'):>4} {fits.count('stretched'):>8} "
              f"{fits.count('shortened'):>8} {fits.count('overflow'):>9} "
              f"{overlaps:>8} {wer_bad:>8} {score_bad:>8} {adeq_bad:>7}")
    if stale_note:
        print(f"\n  !! STALE QC — scores describe audio that s5/s6 has since "
              f"rewritten: {', '.join(stale_note)}."
              f"\n     The numbers above (and any review page or ratings row "
              f"built from them) are not current."
              f"\n     Re-score:  dubadabidu qc {video}")
