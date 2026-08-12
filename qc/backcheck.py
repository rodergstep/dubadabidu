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
from pipeline.text_norm import localize_numbers  # noqa: E402

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


# A hyphen JOINS words here, it does not separate them. num2words emits
# "twenty-five" / "fifty-six", and the old `[^\w\s]` strip glued those into
# "twentyfive" — so a Whisper hypothesis writing "twenty five" scored a
# substitution against a perfectly correct take, for EVERY English number above
# twenty. Ranges ("3-5 layers" -> "three-five") had the same problem.
_HYPHEN = re.compile(r"[-‐-―−]")
# Whisper is inconsistent about the case of unit letters ("20°C" vs "20°c").
# Fold it before symbol expansion or the two sides disagree on a real match.
_DEG_C = re.compile(r"°\s*c\b", re.IGNORECASE)


def _norm(s: str, lang: str) -> str:
    """Put text into the domain the AUDIO is in, so WER compares like with like.

    THE REFERENCE MUST GO THROUGH THE SAME EXPANSION THE VOICE DID. tts_engine
    .synthesize applies pipeline.text_norm.localize_numbers before handing text
    to the engine, so the voice says "fifty percent" while the manifest says
    "50%" — and this function used to strip "%" as punctuation, leaving "fifty"
    against a hypothesis of "fifty percent". WER 1.00 on a 0.15 threshold, for
    audio that was exactly right.

    That is not a display bug. The autopilot re-rolls WER-flagged segments, and
    the re-roll runs a per-take back-transcription veto (tts_engine._measure),
    so a false flag costs a Whisper pass per take on a good segment until
    `_stuck_after` gives up and ESCALATEs.

    Fixed HERE rather than in backcheck.run because segment_wer has four
    callers — run(), the per-take veto, qc/bakeoff.py and qc/stress_wer.py —
    and a fix at one call site is a fix the next caller forgets. _norm is the
    single point both sides of every comparison pass through.
    """
    s = _DEG_C.sub("°C", s)
    # ё -> е FOR THE COMPARISON ONLY. Russian is conventionally written without
    # ё and Whisper transcribes it that way, so `жёлтая` came back as `желтая`
    # and every ё word scored as a substitution. Measured 2026-08-11 on the
    # first real ru backcheck: 7 segments flagged over the 0.15 threshold, and
    # ё accounted for ALL of the difference in four of them (u0007 0.219,
    # u0013 0.200, u0032 0.167, u0040 0.167 -> 0.000). Short segments make it
    # worse — one substitution in a five-word line is already 0.20.
    #
    # This does NOT contradict qc.stress_words.lexicon_key keeping ё and е
    # apart. Different jobs: there, ё is always-stressed and the distinction is
    # a remediation lever; here it is orthographic noise between two spellings
    # of the same sound. Do not "unify" them.
    if lang == "ru":
        s = s.replace("ё", "е").replace("Ё", "Е")   # lowercasing happens below
    # same function the engine saw: digits, %, ° -> target-language words
    s = localize_numbers(s, lang)
    # mop up anything localize_numbers left: languages outside NUM_VERSIONS, and
    # tokens it deliberately refused as ambiguous (both sides refuse alike)
    s = _expand_numbers(s.lower(), lang)
    s = _HYPHEN.sub(" ", s)
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
