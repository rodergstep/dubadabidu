"""Do the metrics we ALREADY compute separate a stress-error take from a good one?

The listener marked 8 of 28 ru takes as containing at least one stress error,
and crucially never marked both takes of a pair — so a correct take exists for
every segment and the only missing piece is knowing which one it is.

Before building a forced-alignment stress detector, check whether something in
the existing pipeline already discriminates. WER is the plausible candidate: a
mis-stressed Russian word may back-transcribe as a different word. mos/f0st are
included as controls — if THEY separate the classes, the listener is hearing
something other than stress and the whole framing is wrong.

Free: local Whisper over 28 clips already on disk.
"""
import json
import glob
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, "/Users/diadumenoss/Documents/projects/dubadabidu")
import yaml
from pipeline.logic import deep_merge
from qc.backcheck import segment_wer
from qc import metrics as X

ROOT = Path("/Users/diadumenoss/Documents/projects/dubadabidu")
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
cfg = deep_merge(cfg, yaml.safe_load(open(ROOT / "config.gpu.yaml",
                                          encoding="utf-8")))
cfg["qc"]["metrics_device"] = "cpu"          # local Mac; cuda is the pod profile

wd = Path(glob.glob(str(ROOT / "work/Organising*"))[0])
truth = json.load(open(wd / "bakeoff/compare_ru_truth.json", encoding="utf-8"))
truth.pop("_axis", None)
truth.pop("_build", None)
rated = json.load(open("/Users/diadumenoss/Downloads/compare_ru_stressing.json",
                       encoding="utf-8"))
man = json.load(open(wd / "manifest.json", encoding="utf-8"))
text_of = {u["id"]: (u["tr"].get("ru") or {}).get("text") for u in man["utterances"]}

bad_keys = {k for v in rated.values() for k in v.get("bad", [])}

rows = []
for key, meta in sorted(truth.items()):
    seg = meta["seg"]
    txt = text_of.get(seg)
    wav = wd / "bakeoff" / meta["path"]
    if not txt or not wav.exists():
        continue
    label = "STRESS-ERROR" if key in bad_keys else "clean"
    w = segment_wer(cfg, txt, wav, "ru")
    m = X.mos_min_window(str(wav))
    rows.append({"key": key, "seg": seg, "label": label, "wer": w, "mos": m})
    print(f"  {key} {seg} {label:12} wer={w:.3f} mos={m:.2f}", flush=True)

print("\n" + "=" * 62)
for feat in ("wer", "mos"):
    bad = [r[feat] for r in rows if r["label"] == "STRESS-ERROR"]
    good = [r[feat] for r in rows if r["label"] == "clean"]
    if not bad or not good:
        continue
    print(f"{feat:5} stress-error n={len(bad):2} mean {mean(bad):.3f} "
          f"sd {pstdev(bad):.3f}   |   clean n={len(good):2} "
          f"mean {mean(good):.3f} sd {pstdev(good):.3f}")
    # AUC = P(a random error take scores higher than a random clean one)
    wins = sum((a > b) + 0.5 * (a == b) for a in bad for b in good)
    auc = wins / (len(bad) * len(good))
    print(f"      AUC {auc:.3f}   (0.5 = no separation, 1.0 = perfect)")
