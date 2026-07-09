"""QC 1: back-transcription. Re-transcribe every fitted segment with faster-whisper
in the TARGET language and compute WER vs. the text we asked TTS to speak.
Catches hallucinations, repetitions, skipped sentences, garbled numbers — without
listening to hours of audio. Segments above the threshold are listed for review.

QC 2 (evaluate.py): ECAPA speaker similarity, MOS, prosody, composite score.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402
from pipeline.device import whisper_device  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def run(cfg: dict, video: str, langs: list[str]) -> None:
    from faster_whisper import WhisperModel
    from jiwer import wer

    dev, ctype = whisper_device(cfg["asr"].get("device", "auto"))
    model = WhisperModel(cfg["asr"]["model"], device=dev, compute_type=ctype)
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    thr = cfg["qc"]["wer_flag_threshold"]

    for lang in langs:
        flagged = []
        for u in man["utterances"]:
            tr = u["tr"][lang]
            # Score against the text TTS actually spoke: s5 may substitute a shorter
            # variant to fit the slot (fit=shortened), recorded as fitted_text. Falling
            # back to tr["text"] would wrongly flag every shortened segment.
            spoken = tr.get("fitted_text") or tr["text"]
            segs, _ = model.transcribe(str(wd / tr.get("placed", tr["fitted"])),
                                       language=lang)
            hyp = " ".join(s.text for s in segs)
            score = wer(_norm(spoken), _norm(hyp)) if spoken.strip() else 1.0
            tr["qc_wer"] = round(score, 3)
            if score > thr or tr.get("fit") == "overflow":
                flagged.append((u["id"], tr["qc_wer"], tr.get("fit")))
        M.save(cfg, video, man)
        print(f"[qc] {lang}: {len(flagged)} flagged (WER>{thr} or overflow)")
        for fid, w, fit in flagged:
            print(f"      {fid}  wer={w}  fit={fit}")
