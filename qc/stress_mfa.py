"""Stress-error detection by pronunciation-variant forced alignment (MFA).

HOW IT WORKS. The MFA Russian dictionary (v3.1.0, CC BY 4.0) transcribes
narrowly: stress is encoded as vowel QUALITY, with the stressed vowel full
(`a o e i u ɨ`) and the rest reduced (`ə ɐ ɪ ʊ`). It already ships homograph
pairs on exactly this principle —

    замок   z̪ ɐ m o k     (замо́к, a lock)
    замок   z̪ a m ə k     (за́мок, a castle)

So: offer the aligner EVERY stress placement for every word (qc.stress_variants
builds them by permuting vowels in the canonical string), let its acoustic model
choose the best fit, and read the winner's vowels back. The full vowel is where
the stress actually landed. Compare against RUAccent, which is a reliable oracle
for where it SHOULD land, and a disagreement is a stress error.

WHY THIS AND NOT THE TWO THAT FAILED. Acoustic prominence (qc/stress_var.py) had
to guess syllable boundaries and rank a continuous quantity; it self-flipped 29%.
The wav2vec2 phoneme route (qc/stress_detect.py) needed vowel reduction the
model does not transcribe — 99% full vowels — and reached AUC 0.562. This uses
an acoustic model TRAINED on the narrow phone set, and real forced alignment
rather than padded Whisper bounds, which is precisely what both lacked.

RUN THE GATE FIRST:  python -m qc.stress_mfa
It scores against the 28 human-labelled takes and must clearly beat AUC 0.647 —
what back-transcription WER already gives free. Below that, it does not ship.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from qc.stress_variants import observed_slot, variants

log = logging.getLogger("dubadabidu.qc.stress_mfa")

SCRATCH = Path(os.environ.get(
    "DUB_MFA_SCRATCH",
    "/private/tmp/claude-501/-Users-diadumenoss-Documents-projects-dubadabidu/"
    "3eb38d6a-75ba-49d9-9436-e859d42c98b6/scratchpad"))
MFA_BIN = SCRATCH / "mfaenv/bin/mfa"
MFA_DATA = SCRATCH / "mfa_data"
BASE_DICT = MFA_DATA / "pretrained_models/dictionary/russian_mfa.dict"
CYR = re.compile(r"[а-яёА-ЯЁ]+")


def _env() -> dict:
    """MFA needs its own bin dir on PATH: the openfst binaries (fstcompile &c)
    are conda packages, not python imports, and MFA shells out to them. Without
    this it dies with ThirdpartyError long after accepting its arguments."""
    return {**os.environ,
            "PATH": f"{SCRATCH / 'mfaenv/bin'}:{os.environ.get('PATH', '')}",
            "MAMBA_ROOT_PREFIX": str(SCRATCH / "mmroot"),
            "MFA_ROOT_DIR": str(MFA_DATA)}


def load_dict(path: Path = BASE_DICT) -> dict[str, list[list[str]]]:
    """word -> list of phone strings. MFA format is
    `word \t prob \t ... \t p1 p2 p3`, with five numeric columns."""
    out: dict[str, list[list[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word, phones = parts[0], parts[-1].split()
        if phones:
            out.setdefault(word.lower(), []).append(phones)
    return out


def build_variant_dict(words: set[str], base: dict, out_path: Path) -> dict:
    """Write a dictionary offering every stress placement for every word.

    Returns {word: {slot: [pronunciation strings]}} so the caller can map an
    aligned phone sequence back to a stress slot.
    """
    index: dict[str, dict] = {}
    lines = []
    for w in sorted(words):
        prons = base.get(w)
        if not prons:
            continue
        # the canonical entry is the first; extra dictionary entries are real
        # homograph readings and are kept as-is
        allv: dict[int, list[list[str]]] = {}
        for ph in prons:
            for slot, cands in variants(ph).items():
                allv.setdefault(slot, []).extend(cands)
        if not allv:
            continue
        index[w] = allv
        seen = set()
        for slot, cands in allv.items():
            for ph in cands:
                key = " ".join(ph)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{w}\t{key}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("variant dictionary: %d words, %d pronunciations -> %s",
             len(index), len(lines), out_path)
    return index


def parse_textgrid(path: Path) -> list[tuple[str, list[str]]]:
    """[(word, [phones])] from an MFA TextGrid, using interval times to assign
    each phone to the word it falls inside."""
    txt = path.read_text(encoding="utf-8")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    cur = None
    xmin = xmax = None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith('name = "'):
            cur = s.split('"')[1]
            tiers[cur] = []
        elif s.startswith("xmin = "):
            xmin = float(s.split("=")[1])
        elif s.startswith("xmax = "):
            xmax = float(s.split("=")[1])
        elif s.startswith("text = ") and cur is not None:
            t = s.split("=", 1)[1].strip().strip('"')
            if t and xmin is not None and xmax is not None:
                tiers[cur].append((xmin, xmax, t))
    wt = tiers.get("words") or []
    pt = tiers.get("phones") or []
    out = []
    for a, b, w in wt:
        if not CYR.fullmatch(w or ""):
            continue
        out.append((w.lower(), [p for (pa, pb, p) in pt
                                if pa >= a - 1e-6 and pb <= b + 1e-6]))
    return out


def align(pairs: list[tuple[Path, str]], workdir: Path) -> dict[str, list]:
    """pairs: [(wav, transcript)] -> {stem: [(word, phones)]}."""
    corpus = workdir / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    words: set[str] = set()
    for wav, text in pairs:
        shutil.copyfile(wav, corpus / f"{wav.stem}.wav")
        (corpus / f"{wav.stem}.lab").write_text(text, encoding="utf-8")
        words |= {w.lower() for w in CYR.findall(text)}

    base = load_dict()
    dpath = workdir / "variants.dict"
    build_variant_dict(words, base, dpath)

    out = workdir / "aligned"
    if out.exists():
        shutil.rmtree(out)
    # ABSOLUTE paths only. MFA's click validator rejects a relative corpus with
    # "Corpus directory does not exist" even when it plainly does, which reads
    # like a missing-file bug and is really a cwd-resolution one.
    # ONLY --clean. Adding --single_speaker/--beam/--retry_beam made MFA report
    # "Invalid value for 'CORPUS_DIRECTORY': Corpus directory does not exist" —
    # one of them swallows the following argument, so the corpus path was being
    # parsed as an option value and a beam number was being parsed as the
    # corpus. The error names the wrong thing entirely; the defaults align fine.
    cmd = [str(MFA_BIN), "align", "--clean",
           str(corpus.resolve()), str(dpath.resolve()), "russian_mfa",
           str(out.resolve())]
    r = subprocess.run(cmd, env=_env(), capture_output=True, text=True)
    if r.returncode != 0:
        log.error("mfa align failed:\n%s", (r.stderr or r.stdout)[-2500:])
        raise SystemExit("mfa align failed")
    res = {}
    for tg in out.rglob("*.TextGrid"):
        res[tg.stem] = parse_textgrid(tg)
    return res


# --- validation gate -------------------------------------------------------

def validate(labels_json: str, truth_json: str, wd: str,
             workdir: Path | None = None) -> int:
    """Score against the 28 human-labelled takes. Bar: beat AUC 0.647 (WER)."""
    from statistics import mean
    from ruaccent import RUAccent
    from qc.stress_detect import expected_stress_index, VOWELS_CYR

    workdir = workdir or Path("work/_mfa")
    workdir.mkdir(parents=True, exist_ok=True)
    truth = json.loads(Path(truth_json).read_text(encoding="utf-8"))
    truth.pop("_axis", None)
    truth.pop("_build", None)
    rated = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    bad_keys = {k for v in rated.values() for k in v.get("bad", [])}
    judged = {g for g, v in rated.items() if v.get("bad") or v.get("best")}

    man = json.loads((Path(wd) / "manifest.json").read_text(encoding="utf-8"))
    text_of = {u["id"]: (u["tr"].get("ru") or {}).get("text")
               for u in man["utterances"]}

    pairs, keyof = [], {}
    for key, meta in sorted(truth.items()):
        if key.split("c")[0] not in judged:
            continue
        wav = Path(wd) / "bakeoff" / meta["path"]
        txt = text_of.get(meta["seg"])
        if not (wav.exists() and txt):
            continue
        staged = workdir / f"{key}.wav"
        shutil.copyfile(wav, staged)
        pairs.append((staged, txt))
        keyof[key] = key
    log.info("aligning %d takes", len(pairs))
    aligned = align(pairs, workdir)

    acc = RUAccent()
    acc.load(omograph_model_size="turbo", use_dictionary=True)

    rows = []
    for key in keyof:
        words = aligned.get(key) or []
        n_res = n_bad = 0
        detail = []
        for w, phones in words:
            exp = expected_stress_index(w, acc)
            if exp is None:
                continue
            obs = observed_slot(phones)
            if obs is None:
                continue
            n_res += 1
            if obs != exp:
                n_bad += 1
                detail.append(f"{w}({exp}->{obs})")
        rows.append({"key": key,
                     "label": "ERROR" if key in bad_keys else "clean",
                     "resolved": n_res, "mismatches": n_bad,
                     "rate": n_bad / n_res if n_res else 0.0,
                     "detail": detail})
        print(f"  {key} {rows[-1]['label']:6} resolved={n_res:2} "
              f"mismatch={n_bad:2}  {' '.join(detail[:4])}", flush=True)

    bad = [r for r in rows if r["label"] == "ERROR"]
    good = [r for r in rows if r["label"] == "clean"]
    if not bad or not good:
        print("not enough labelled takes")
        return 1
    print("\n" + "=" * 64)
    print(f"words resolved per take: mean "
          f"{mean(r['resolved'] for r in rows):.1f}")
    best = 0.0
    for feat in ("mismatches", "rate"):
        b = [r[feat] for r in bad]
        g = [r[feat] for r in good]
        auc = sum((x > y) + 0.5 * (x == y) for x in b for y in g) / (len(b) * len(g))
        best = max(best, auc)
        print(f"{feat:11} error-take mean {mean(b):.3f} | clean {mean(g):.3f} "
              f"| AUC {auc:.3f}")
    print(f"\nbar to beat (WER today): 0.647   ->  "
          f"{'PASSES' if best > 0.72 else 'FAILS — do not wire this in'}")
    return 0 if best > 0.72 else 1


if __name__ == "__main__":
    import glob
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    w = glob.glob("work/Organising*")[0]
    sys.exit(validate(
        "/Users/diadumenoss/Downloads/compare_ru_stressing.json",
        f"{w}/bakeoff/compare_ru_truth.json", w))
