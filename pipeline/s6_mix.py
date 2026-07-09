"""s6: assemble dubbed vocals on the original timeline (pydub, frame-rate
normalized), mix with background stem, loudnorm to target LUFS.

Segment hygiene: edges are trimmed at TRIM_DB and faded (synth takes carry
leading silence and can end at full level -> clicks between sentences).
Loudness: two-pass loudnorm with linear=true — the one-pass default is
dynamic compression, which audibly squashes speech."""
from __future__ import annotations
import json, logging, subprocess
from pathlib import Path
from pydub import AudioSegment, silence
from . import manifest as M

log = logging.getLogger("dubadabidu.s6")
TRIM_DB = -45          # edge-trim threshold
TRIM_CUSHION_MS = 30   # keep this much of the quiet edge
FADE_IN_MS, FADE_OUT_MS = 10, 60  # long fade-out: end-of-generation garble hides
                                  # above the trim threshold right at sentence joins


def _clean(seg: AudioSegment) -> AudioSegment:
    lead = silence.detect_leading_silence(seg, silence_threshold=TRIM_DB)
    tail = silence.detect_leading_silence(seg.reverse(), silence_threshold=TRIM_DB)
    seg = seg[max(0, lead - TRIM_CUSHION_MS):
              len(seg) - max(0, tail - TRIM_CUSHION_MS)]
    return seg.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)


def _loudnorm_two_pass(premix: Path, out: Path, lufs: int) -> None:
    spec = f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    probe = subprocess.run(
        ["ffmpeg", "-i", str(premix), "-af", f"{spec}:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True, check=True)
    stats = json.loads(probe.stderr[probe.stderr.rindex("{"):
                                    probe.stderr.rindex("}") + 1])
    measured = (f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
                f":measured_LRA={stats['input_lra']}"
                f":measured_thresh={stats['input_thresh']}"
                f":offset={stats['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(premix),
                    "-af", spec + measured, "-c:a", "aac", "-b:a", "192k",
                    str(out)], check=True)


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    total_ms = int(man["duration"] * 1000)

    for lang in langs:
        # timeline stays at the TTS rate; the single upsample to 44.1k happens
        # in ffmpeg below (swr) — pydub's per-segment ratecv audibly aliases
        rate = cfg["tts"]["sample_rate"]
        track = AudioSegment.silent(duration=total_ms, frame_rate=rate)
        for u in man["utterances"]:
            tr = u["tr"][lang]
            seg = _clean(AudioSegment.from_wav(wd / tr["fitted"])
                         .set_channels(1))
            # persist the exact audio that lands in the mix — evaluate/review/
            # backcheck grade this, not the pre-trim fitted file
            placed = (wd / tr["fitted"]).with_name(f"{u['id']}_placed.wav")
            seg.export(placed, format="wav")
            tr["placed"] = str(placed.relative_to(wd))
            track = track.overlay(seg, position=int(u["start"] * 1000))
        voc = wd / f"dub_{lang}_vocals.wav"
        track.export(voc, format="wav")

        premix = wd / f"dub_{lang}_premix.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(voc), "-i", str(wd / "background.wav"),
            "-filter_complex",
            f"[0:a]aresample=44100[v];"
            f"[1:a]volume={cfg['mix']['bg_gain_db']}dB,apad[bg];"
            f"[v][bg]amix=inputs=2:duration=first:normalize=0[out]",
            "-map", "[out]", str(premix)], check=True)
        mixed = wd / f"dub_{lang}.m4a"
        _loudnorm_two_pass(premix, mixed, cfg["mix"]["lufs"])
        premix.unlink(missing_ok=True)
        man["stages"][f"s6_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %s", lang, mixed)
