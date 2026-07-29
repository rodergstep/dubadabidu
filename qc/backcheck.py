"""QC 1: back-transcription. Re-transcribe every fitted segment with faster-whisper
in the TARGET language and compute WER vs. the text we asked TTS to speak.
Catches hallucinations, repetitions, skipped sentences, garbled numbers — without
listening to hours of audio. Segments above the threshold are listed for review.

The Whisper model is a module singleton (the autopilot re-runs QC every re-roll
round; reloading large-v3 each time costs ~30s a round for nothing) and is also
shared with tts_engine's per-take WER veto.

WER normalization expands digits to words on BOTH sides (num2words): TTS speaks
"25" as "twenty-five" and Whisper often writes it back as "25" (or vice versa),
which used to flag perfectly correct measurement-heavy segments — false flags
that made the autopilot burn re-roll rounds on good takes.

QC 2 (evaluate.py): ECAPA speaker similarity, MOS, prosody, composite score.
"""
from __future__ import annotations
import logging, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402
from pipeline.device import whisper_device  # noqa: E402

log = logging.getLogger("dubadabidu.qc.backcheck")
_model = None
_model_key = None


def _whisper(cfg: dict):
    global _model, _model_key
    dev, ctype = whisper_device(cfg["asr"].get("device", "auto"))
    key = (cfg["asr"]["model"], dev, ctype)
    if _model is None or _model_key != key:
        from faster_whisper import WhisperModel
        log.info("loading %s on %s/%s ...", *key)
        _model = WhisperModel(key[0], device=dev, compute_type=ctype)
        _model_key = key
    return _model


def transcribe_wav(cfg: dict, path: str | Path, lang: str) -> str:
    """Back-transcribe one wav in the target language (shared model)."""
    segs, _ = _whisper(cfg).transcribe(str(path), language=lang)
    return " ".join(s.text for s in segs)


def _expand_numbers(s: str, lang: str) -> str:
    """Digits -> words so '25' and 'twenty-five' compare equal. Decimal/ratio
    separators become spaces first (each number part is expanded on its own).
    Falls back to stripping digit runs if num2words is missing or the language
    is unsupported — still better than counting every numeral as an error."""
    s = re.sub(r"(?<=\d)[.,:/](?=\d)", " ", s)
    try:
        from num2words import num2words

        def _w(m: re.Match) -> str:
            try:
                return " " + num2words(int(m.group()), lang=lang) + " "
            except NotImplementedError:
                return " "
        return re.sub(r"\d+", _w, s)
    except ImportError:
        return re.sub(r"\d+", " ", s)


def _norm(s: str, lang: str) -> str:
    s = _expand_numbers(s.lower(), lang)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def segment_wer(cfg: dict, spoken: str, wav: str | Path, lang: str) -> float:
    """Number-normalized WER of one placed segment vs. the text TTS spoke."""
    from jiwer import wer
    if not spoken.strip():
        return 1.0
    hyp = transcribe_wav(cfg, wav, lang)
    return wer(_norm(spoken, lang), _norm(hyp, lang))


def run(cfg: dict, video: str, langs: list[str],
        only: list[str] | None = None) -> None:
    """only: utterance ids to (re)check; None = all. The autopilot passes the
    handful of re-rolled segments so a fix round doesn't re-transcribe the
    whole video (minutes of Whisper per round on a 1h lesson)."""
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    thr = cfg["qc"]["wer_flag_threshold"]

    for lang in langs:
        flagged = []
        checked = 0
        for u in man["utterances"]:
            if only is not None and u["id"] not in only:
                continue
            checked += 1
            tr = u["tr"][lang]
            # Score against the text TTS actually spoke: s5 may substitute a shorter
            # variant to fit the slot (fit=shortened), recorded as fitted_text. Falling
            # back to tr["text"] would wrongly flag every shortened segment.
            spoken = tr.get("fitted_text") or tr["text"]
            score = segment_wer(cfg, spoken, M.scored_path(wd, tr), lang)
            tr["qc_wer"] = round(score, 3)
            # stamp the graded audio (see manifest.stale_qc): an s5/s6 re-run
            # rewrites the placed wav and this WER stops describing it
            M.stamp_qc(wd, tr, "wer")
            if score > thr or tr.get("fit") == "overflow":
                flagged.append((u["id"], tr["qc_wer"], tr.get("fit")))
        M.save(cfg, video, man)
        scope = f"{checked}/{len(man['utterances'])} segments, " \
            if only is not None else ""
        print(f"[qc] {lang}: {scope}{len(flagged)} flagged "
              f"(WER>{thr} or overflow)")
        for fid, w, fit in flagged:
            print(f"      {fid}  wer={w}  fit={fit}")
