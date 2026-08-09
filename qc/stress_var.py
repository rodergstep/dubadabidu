"""Does qwen's Russian stress placement VARY between takes of the same text?

This is the premise the whole oracle-veto idea rests on. If the model always
puts the stress in the same (wrong) place, re-rolling cannot escape it and
best_of is the wrong lever. If it moves, a veto can pick the good roll.

DELIBERATELY NOT A CORRECTNESS TEST. Asking "is this stress right?" needs a
reliable oracle AND a reliable detector. Asking "did take 0 and take 1 put the
stress on the same syllable?" needs only that the detector be CONSISTENT — a
much weaker requirement, and one this script measures rather than assumes:
every comparison is run alongside a self-comparison of the same take against a
perturbed copy of itself, which is the detector's own noise floor. A
disagreement rate that does not clearly exceed that floor means the script has
measured nothing.

Input: the ru control arm already on disk (46 segments x 2 takes), so this
costs nothing and needs no pod.
"""
import glob
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/diadumenoss/Documents/projects/dubadabidu")

VOWELS = set("аеёиоуыэюя")
SR = 16000
MODEL = "mlx-community/whisper-large-v3-mlx"


def norm_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


def load16k(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def nuclei_stress(y: np.ndarray, n_vowels: int) -> int | None:
    """Index of the syllable nucleus carrying the most prominence, 0-based.

    Sonority envelope = RMS energy gated by voicing. Peaks in it are syllable
    nuclei; we keep the n_vowels strongest, restore time order, and score each
    by energy x pitch x duration — the three acoustic correlates of Russian
    lexical stress. Returns None when the word is too short to resolve.
    """
    import librosa
    if len(y) < int(0.06 * SR) or n_vowels < 2:
        return None
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR,
                                     frame_length=1024, hop_length=hop)
    except Exception:
        return None
    n = min(len(rms), len(f0))
    rms, f0, voiced = rms[:n], f0[:n], voiced[:n]
    if n < n_vowels * 3:
        return None
    son = rms * np.nan_to_num(voiced, nan=0.0)
    if son.max() <= 0:
        return None
    son = son / son.max()
    # smooth so one nucleus is one peak, not a burst of frames
    k = max(3, n // (n_vowels * 4) | 1)
    son = np.convolve(son, np.ones(k) / k, mode="same")
    peaks = librosa.util.peak_pick(
        son, pre_max=k, post_max=k, pre_avg=k, post_avg=k,
        delta=0.01, wait=max(1, n // (n_vowels * 3)))
    if len(peaks) < n_vowels:
        # fall back to slicing the word into n_vowels equal parts
        edges = np.linspace(0, n, n_vowels + 1).astype(int)
        peaks = np.array([int((a + b) / 2) for a, b in zip(edges, edges[1:])])
    else:
        peaks = np.sort(peaks[np.argsort(son[peaks])[-n_vowels:]])
    f0f = np.nan_to_num(f0, nan=0.0)
    scores = []
    for p in peaks:
        lo, hi = max(0, p - k), min(n, p + k + 1)
        scores.append(float(son[lo:hi].mean() * (1.0 + f0f[lo:hi].mean() / 200.0)))
    return int(np.argmax(scores))


def words_of(path: str, model) -> list[tuple[str, float, float]]:
    import mlx_whisper
    r = mlx_whisper.transcribe(path, path_or_hf_repo=model, language="ru",
                               word_timestamps=True, verbose=False)
    out = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            t = norm_word(w.get("word", ""))
            if t:
                out.append((t, float(w["start"]), float(w["end"])))
    return out


def stress_map(path: str, jitter: bool = False) -> dict[str, int]:
    """jitter=True builds the NULL: the same take, re-rendered.

    A faint-noise copy was the first attempt and it is too weak a control —
    it leaves every word boundary exactly where it was, so it tests only that
    the detector is deterministic, which it trivially is. Two real takes differ
    in TIMING, and alignment drift is the most likely way for the detector to
    flip without the stress having moved. So the null time-stretches by 5% and
    changes gain: the prosody is perturbed the way a different take perturbs
    it, while the stressed syllable by construction stays put.
    """
    import librosa
    y = load16k(path)
    if jitter:
        y = librosa.effects.time_stretch(y, rate=1.05) * 0.85
        tmp = Path("/tmp/_stressnull.wav")
        sf.write(tmp, y, SR)
        path = str(tmp)
    out = {}
    for w, t0, t1 in words_of(path, MODEL):
        nv = sum(c in VOWELS for c in w)
        if nv < 2:
            continue
        seg = y[int(t0 * SR):int(t1 * SR)]
        idx = nuclei_stress(seg, nv)
        if idx is not None:
            out.setdefault(w, idx)   # first occurrence only
    return out


def main() -> None:
    d = Path(glob.glob(
        "work/Organising*/bakeoff/seg/qwen+fast+control/ru")[0])
    segs = sorted({p.name.split("_t")[0] for p in d.glob("*_t*.wav")})
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(segs)
    segs = segs[:limit]

    cross_same = cross_diff = 0
    self_same = self_diff = 0
    examples = []
    for i, s in enumerate(segs, 1):
        a, b = d / f"{s}_t0.wav", d / f"{s}_t1.wav"
        if not (a.exists() and b.exists()):
            continue
        ma, mb = stress_map(str(a)), stress_map(str(b))
        mj = stress_map(str(a), jitter=True)          # detector noise floor
        for w in set(ma) & set(mb):
            if ma[w] == mb[w]:
                cross_same += 1
            else:
                cross_diff += 1
                if len(examples) < 12:
                    examples.append(f"{s} {w}: take0 syl{ma[w]} vs take1 syl{mb[w]}")
        for w in set(ma) & set(mj):
            self_same += 1 if ma[w] == mj[w] else 0
            self_diff += 0 if ma[w] == mj[w] else 1
        print(f"  [{i}/{len(segs)}] {s}", flush=True)

    ct, st = cross_same + cross_diff, self_same + self_diff
    print("\n" + "=" * 62)
    print(f"words compared across takes : {ct}")
    print(f"  stress on a DIFFERENT syllable : {cross_diff} "
          f"({cross_diff/max(ct,1)*100:.1f}%)")
    print(f"detector noise floor (same take, +noise): {st} words, "
          f"{self_diff} flips ({self_diff/max(st,1)*100:.1f}%)")
    if ct and st:
        print(f"\nsignal-to-floor ratio: "
              f"{(cross_diff/ct) / max(self_diff/st, 1e-9):.2f}x")
    print("\nexamples:")
    for e in examples:
        print("  " + e)


if __name__ == "__main__":
    main()
