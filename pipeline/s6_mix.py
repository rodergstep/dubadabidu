"""s6: assemble dubbed vocals on the original timeline (pydub, frame-rate
normalized), mix with background stem, loudnorm to target LUFS.

Segment hygiene: edges are trimmed at TRIM_DB and faded (synth takes carry
leading silence and can end at full level -> clicks between sentences).
Loudness: two-pass loudnorm with linear=true — the one-pass default is
dynamic compression, which audibly squashes speech.

Background ducking (mix.duck): the background is the separation INSTRUMENTAL
stem, which still carries a faint residual of the original Ukrainian voice
(no separator is perfect). Mixed back flat, that residual plays under the dub
— the classic 'two voices' AI-dub tell. We sidechain-compress the background
against the dub vocals, so it ducks exactly while the dub speaks (which is
where the residual would be exposed) and returns in the real gaps."""
from __future__ import annotations
import json, logging, math, subprocess
from pathlib import Path
from pydub import AudioSegment, silence
from . import manifest as M

log = logging.getLogger("dubadabidu.s6")
TRIM_DB = -45          # edge-trim threshold
TRIM_CUSHION_MS = 30   # keep this much of the quiet edge
FADE_IN_MS, FADE_OUT_MS = 10, 60  # long fade-out: end-of-generation garble hides
                                  # above the trim threshold right at sentence joins


def _mix_filter(mix: dict) -> str:
    """filter_complex mixing dub vocals [0:a] with background [1:a].
    With mix.duck (default on), sidechaincompress ducks the background using
    the vocals as the key; otherwise the background plays at a flat gain."""
    bg_gain = mix["bg_gain_db"]
    if not mix.get("duck", True):
        return (f"[0:a]aresample=44100[v];"
                f"[1:a]volume={bg_gain}dB,apad[bg];"
                f"[v][bg]amix=inputs=2:duration=first:normalize=0[out]")
    thr = mix.get("duck_threshold", 0.03)   # ~ -30 dB key level opens the duck
    ratio = mix.get("duck_ratio", 8)
    attack = mix.get("duck_attack_ms", 5)
    release = mix.get("duck_release_ms", 300)
    return (f"[0:a]aresample=44100,asplit=2[v][key];"
            f"[1:a]volume={bg_gain}dB,apad[bgpad];"
            f"[bgpad][key]sidechaincompress=threshold={thr}:ratio={ratio}:"
            f"attack={attack}:release={release}:makeup=1[bg];"
            f"[v][bg]amix=inputs=2:duration=first:normalize=0[out]")


def _clean(seg: AudioSegment) -> AudioSegment:
    lead = silence.detect_leading_silence(seg, silence_threshold=TRIM_DB)
    tail = silence.detect_leading_silence(seg.reverse(), silence_threshold=TRIM_DB)
    seg = seg[max(0, lead - TRIM_CUSHION_MS):
              len(seg) - max(0, tail - TRIM_CUSHION_MS)]
    return seg.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)


def _normalize_segment(seg: AudioSegment, target_dbfs: float) -> tuple[AudioSegment, float]:
    """Level each utterance to a common target with a PURE GAIN SHIFT (one
    multiply, not dynamics processing) -> (segment, applied_gain_db).

    best_of picks each segment's take independently, so consecutive utterances
    drift in level run-to-run — the 'pasted-in segments' tell. A single gain
    per segment removes that drift while leaving the dynamics INSIDE each
    segment untouched (unlike a compressor, which would flatten prosody and
    feed the monotony complaint). Sets relative consistency only; the s6
    loudnorm still sets absolute program loudness, and the master limiter
    still backstops peaks. Near-silent segments have no reliable level to
    match, so they pass through unchanged."""
    if seg.dBFS == float("-inf"):
        return seg, 0.0
    gain = target_dbfs - seg.dBFS
    return seg.apply_gain(gain), gain


def _assemble(placements: list[tuple[int, AudioSegment]], total_ms: int,
              rate: int) -> AudioSegment:
    """Lay non-overlapping segments onto one silent timeline of `rate` Hz.

    NOT `track.overlay(seg, position=...)` in a loop, which is what this
    replaced. pydub never mutates: every overlay() slices the base, spawns a
    NEW AudioSegment and copies the WHOLE timeline. That is O(n) per segment
    and O(n^2) over a video. A 1 h lesson at 24 kHz mono 16-bit is ~173 MB per
    copy; at ~400 utterances the old loop moved ~35 GB through memory and
    allocated 400 fresh 173 MB buffers to produce one track — the bulk of
    phase C's wall clock on a real lesson, and the reason IMPROVEMENT_PLAN
    listed s6 RAM as a scale risk.

    One int32 accumulator, one clip, one spawn. Segments are summed rather
    than pasted so the result matches overlay() exactly even if two ever touch
    (s5's soft-anchor placement makes overlap impossible, but the mix must not
    depend on that invariant holding); int32 headroom plus the clip reproduces
    audioop.add's saturation.
    """
    import numpy as np
    need_ms = max([total_ms] + [pos + len(seg) for pos, seg in placements])
    n = int(need_ms * rate / 1000)
    acc = np.zeros(n, dtype=np.int32)
    for pos, seg in placements:
        # what pydub's _sync would have done to match the silent base track
        seg = seg.set_frame_rate(rate).set_channels(1).set_sample_width(2)
        data = np.frombuffer(seg._data, dtype="<i2")
        start = int(pos * rate / 1000)
        end = min(start + len(data), n)
        if end > start:
            acc[start:end] += data[:end - start]
    clipped = np.clip(acc, -32768, 32767).astype("<i2")
    return AudioSegment(clipped.tobytes(), frame_rate=rate,
                        sample_width=2, channels=1)


def _master(inp: Path, out: Path, m: dict) -> None:
    """Light mastering chain on the dub vocals BEFORE the background mix
    (Spotify pedalboard). best_of takes are picked per segment, so consecutive
    utterances drift in level and brightness — the 'pasted-in segments' tell.
    A gentle high-pass + compressor evens that out; a small high-shelf cut
    tames TTS sibilance; an optional tiny room lets the voice sit in space
    instead of dead-dry. Conservative by design — loudnorm still runs on the
    full mix afterward, and a brickwall limiter guards the intermediate wav
    from clipping. Opt-in (mix.master.enabled); no-op otherwise.

    pedalboard is GPLv3, but it only processes audio — the WAV it emits is not
    a derivative of the library, so course output stays unencumbered. It is an
    OPT-IN extra (`.[master]`) so the default install carries no GPL dep."""
    try:
        from pedalboard import (Pedalboard, HighpassFilter, Compressor,
                                HighShelfFilter, Reverb, Limiter)
        from pedalboard.io import AudioFile
    except ImportError as e:
        raise FileNotFoundError(
            "pedalboard not importable — mix.master.enabled needs it: "
            f"pip install .[master] (or set mix.master.enabled: false). ({e})")
    chain = [
        HighpassFilter(cutoff_frequency_hz=m.get("hpf_hz", 80)),
        Compressor(threshold_db=m.get("comp_threshold_db", -18.0),
                   ratio=m.get("comp_ratio", 2.5),
                   attack_ms=m.get("comp_attack_ms", 5),
                   release_ms=m.get("comp_release_ms", 120)),
        HighShelfFilter(cutoff_frequency_hz=m.get("deess_hz", 7000),
                        gain_db=m.get("deess_gain_db", -2.0)),
    ]
    wet = float(m.get("reverb_wet", 0.0))
    if wet > 0:   # subtle room; matching the source acoustic is hard, keep it low
        chain.append(Reverb(room_size=m.get("reverb_room", 0.1),
                            wet_level=wet, dry_level=1.0 - wet))
    chain.append(Limiter(threshold_db=m.get("limiter_db", -1.0)))
    board = Pedalboard(chain)
    with AudioFile(str(inp)) as f:
        audio, sr = f.read(f.frames), f.samplerate
    with AudioFile(str(out), "w", sr, audio.shape[0]) as f:
        f.write(board(audio, sr))


def _loudnorm_two_pass(premix: Path, out: Path, lufs: int) -> None:
    spec = f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    probe = subprocess.run(
        ["ffmpeg", "-i", str(premix), "-af", f"{spec}:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True, check=True)
    stats = json.loads(probe.stderr[probe.stderr.rindex("{"):
                                    probe.stderr.rindex("}") + 1])

    def _encode(af: str) -> None:
        # -ar 44100 is NOT redundant. loudnorm resamples to 192 kHz internally,
        # and with no explicit output rate ffmpeg keeps the filter's rate —
        # which AAC then clamps to its 96 kHz maximum. The premix is already
        # aresample=44100, so without this every dubbed track shipped at 96 kHz:
        # double the data for no audible gain, on a file destined for YouTube.
        # Caught by the edge plumbing run 2026-08-02, not by any test.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(premix),
                        "-af", af, "-c:a", "aac", "-b:a", "192k",
                        "-ar", "44100", str(out)],
                       check=True)

    # loudnorm reports input_i = -inf (or a floor around -70) for (near-)silent
    # input; feeding measured_I=-inf into the linear second pass yields garbage or
    # an ffmpeg error. When the measurement isn't usable, fall back to a single
    # dynamic pass (still hits the target loudness, just without linear scaling).
    try:
        measured_i = float(stats["input_i"])
    except (KeyError, ValueError):
        measured_i = float("-inf")
    if not math.isfinite(measured_i) or measured_i < -70.0:
        log.warning("loudnorm: premix too quiet (measured_I=%s) — single-pass "
                    "fallback (no linear correction)", stats.get("input_i"))
        _encode(spec)
        return
    measured = (f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
                f":measured_LRA={stats['input_lra']}"
                f":measured_thresh={stats['input_thresh']}"
                f":offset={stats['target_offset']}:linear=true")
    _encode(spec + measured)


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    total_ms = int(man["duration"] * 1000)

    for lang in langs:
        # timeline stays at the TTS rate; the single upsample to 44.1k happens
        # in ffmpeg below (swr) — pydub's per-segment ratecv audibly aliases
        rate = cfg["tts"]["sample_rate"]
        norm = cfg["mix"].get("segment_norm", {})
        # PASS 1: clean/normalize/persist each segment and record its placement.
        # The timeline is sized AFTER this, to hold every segment in full: pydub's
        # overlay TRUNCATES whatever extends past the base length, so a track cut
        # to the source-video duration silently clips the tail of the final dub
        # segment when it runs long or drifts late under s5's soft-anchor retiming.
        placements = []   # (position_ms, seg)
        for u in man["utterances"]:
            tr = u["tr"][lang]
            seg = _clean(AudioSegment.from_wav(wd / tr["fitted"])
                         .set_channels(1))
            # per-segment loudness normalization: kill take-to-take level drift
            # (the 'pasted-in' tell) with a pure gain shift, dynamics-preserving
            if norm.get("enabled", True):
                seg, gain = _normalize_segment(seg, norm.get("target_dbfs", -20.0))
                tr["norm_gain_db"] = round(gain, 2)
            # persist the exact audio that lands in the mix — evaluate/review/
            # backcheck grade this, not the pre-trim fitted file
            placed = (wd / tr["fitted"]).with_name(f"{u['id']}_placed.wav")
            seg.export(placed, format="wav")
            tr["placed"] = str(placed.relative_to(wd))
            # s5's soft-anchored timeline: overlay at placed_start (may drift
            # within fit.drift_max_s of the source start); overlap-free by
            # construction. Fall back to the source start for old manifests.
            placements.append(
                (int(tr.get("placed_start", u["start"]) * 1000), seg))
        # PASS 2: build the timeline long enough for the source AND every segment,
        # then lay the segments down. Normally the track equals total_ms (no
        # change); it only grows when a late/long segment would otherwise be
        # clipped — a fraction of a second past the video, which the ffmpeg mix
        # (bg apad) tolerates.
        track = _assemble(placements, total_ms, rate)
        voc = wd / f"dub_{lang}_vocals.wav"
        track.export(voc, format="wav")
        if cfg["mix"].get("master", {}).get("enabled"):
            mastered = wd / f"dub_{lang}_vocals_master.wav"
            _master(voc, mastered, cfg["mix"]["master"])
            voc = mastered

        premix = wd / f"dub_{lang}_premix.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(voc), "-i", str(wd / "background.wav"),
            "-filter_complex", _mix_filter(cfg["mix"]),
            "-map", "[out]", str(premix)], check=True)
        mixed = wd / f"dub_{lang}.m4a"
        _loudnorm_two_pass(premix, mixed, cfg["mix"]["lufs"])
        premix.unlink(missing_ok=True)
        man["stages"][f"s6_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %s", lang, mixed)
