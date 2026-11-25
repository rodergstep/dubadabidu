# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fakes for every external boundary the local pipeline crosses.

The neutrality harness must run the REAL stage code end to end, offline and
deterministically, so it fakes only what leaves the process or loads model
weights, at the outermost seam (the third-party module), never an internal
function. A refactor therefore cannot route around a fake by renaming
something of ours.

  audio_separator.separator.Separator   s1 RoFormer separation
  faster_whisper.WhisperModel           s2 ASR and qc.backcheck
  openai.OpenAI                         s3 / s5 translation, adequacy, shorten
  edge_tts.Communicate                  s4 / s5 / evaluate floor synthesis
  speechbrain...EncoderClassifier       ECAPA speaker embeddings (qc.metrics)
  distillmos.ConvTransformerSQAModel    Distill-MOS (qc.metrics)

Every fake is a pure function of its input. The LLM fake records each request
(model, response_format, messages) so prompt bytes are part of the snapshot.
`install(full=False)` puts them into sys.modules; with full=True the two model
fakes are skipped and the real weights load from local caches (offline).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import types
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

LLM_LOG: list[dict[str, object]] = []

# --- the fixture's "transcript": three utterances, per source language -------

_SRC_TEXT = {
    "uk": ["Сьогодні ми змішуємо ультрамарин з білилами.",
           "Тримайте пензель легко, майже без натиску.",
           "Наступний шар кладемо тільки по сухому."],
    "ru": ["Сегодня мы смешиваем ультрамарин с белилами.",
           "Держите кисть легко, почти без нажима.",
           "Следующий слой кладём только по сухому."],
}
_SEG_TIMES = [(0.30, 1.80), (2.30, 3.90), (4.40, 5.70)]
_BACK_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight"]


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


# --- faster_whisper -----------------------------------------------------------

@dataclass
class _Word:
    start: float
    end: float
    word: str


@dataclass
class _Seg:
    start: float
    end: float
    text: str
    words: list[_Word] = field(default_factory=list)


class WhisperModel:
    def __init__(self, model: str, device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self.model = model

    def transcribe(self, audio: str, language: str | None = None,
                   **_kw: object) -> tuple[Iterator[_Seg], dict[str, str]]:
        if language in _SRC_TEXT:
            dur = _duration(audio)
            if dur > 5.0:          # the whole-video ASR pass (s2)
                return iter(_source_segments(language)), {"language": language}
        # back-transcription of one dub segment: words by duration
        n = max(1, int(round(_duration(audio) * 2.5)))
        text = " " + " ".join(_BACK_WORDS[i % len(_BACK_WORDS)] for i in range(n))
        return iter([_Seg(0.0, _duration(audio), text)]), {"language": str(language)}


def _source_segments(lang: str) -> list[_Seg]:
    out = []
    for (s, e), text in zip(_SEG_TIMES, _SRC_TEXT[lang], strict=True):
        words = text.split()
        step = (e - s) / len(words)
        ws = [_Word(round(s + i * step, 3), round(s + (i + 1) * step - 0.02, 3),
                    " " + w) for i, w in enumerate(words)]
        out.append(_Seg(s, e, " " + text, ws))
    return out


def _duration(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


# --- audio_separator ----------------------------------------------------------

class Separator:
    def __init__(self, output_dir: str, model_file_dir: str = "",
                 output_format: str = "WAV", **_kw: object) -> None:
        self.out = Path(output_dir)

    def load_model(self, model_filename: str = "") -> None:
        self.model = model_filename

    def separate(self, path: str) -> list[str]:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        rng = np.random.default_rng(1234)
        bg = (0.02 * rng.standard_normal(len(mono))).astype("float32")
        self.out.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        voc = f"{stem}_(Vocals)_{self.model}.wav"
        ins = f"{stem}_(Instrumental)_{self.model}.wav"
        sf.write(self.out / voc, mono, sr)
        sf.write(self.out / ins, bg, sr)
        return [voc, ins]


# --- openai -------------------------------------------------------------------

@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    model: str
    choices: list[_Choice]


@dataclass
class _ModelId:
    id: str


@dataclass
class _ModelList:
    data: list[_ModelId]


class _Completions:
    def create(self, *, model: str, messages: list[dict[str, str]],
               response_format: dict[str, object] | None = None,
               temperature: float | None = None,
               timeout: float | None = None) -> _Resp:
        system = messages[0]["content"]
        user = messages[1]["content"]
        LLM_LOG.append({"model": model, "temperature": temperature,
                        "response_format": response_format,
                        "system": system, "user": user})
        return _Resp(model=f"{model}-fake-0001",
                     choices=[_Choice(_Msg(json.dumps(_answer(system, user),
                                                      ensure_ascii=False)))])


class _Chat:
    completions = _Completions()


class _Models:
    def list(self) -> _ModelList:
        return _ModelList([_ModelId("deepseek-flash")])


class OpenAI:
    def __init__(self, base_url: str = "", api_key: str = "",
                 timeout: float | None = None) -> None:
        self.chat = _Chat()
        self.models = _Models()


def _target(system: str) -> str:
    for pat in (r"into (\w+) for VOICE-OVER", r"finalizing (\w+) voice-over",
                r"translated\s+from \w+ into (\w+)", r"-> (\w+)\s+dubbing",
                r"given (\w+) line"):
        m = re.search(pat, system)
        if m:
            return m.group(1)
    return "Target"


def _translate(src: str, tag: str, seg_id: str) -> str:
    words = [w[::-1] for w in src.split()]
    if int(seg_id[1:]) % 3 == 0:            # every third segment runs long
        words += ["and", "a", "little", "more", "said", "slowly"]
    return f"{tag[:2]} " + " ".join(words)


def _shorter(text: str, n: int) -> list[str]:
    words = text.split()
    return [" ".join(words[:max(1, len(words) - 2 * (k + 1))]) for k in range(n)]


def _answer(system: str, user: str) -> dict[str, object]:
    tag = _target(system)
    if "extract up to" in system:
        return {"terms": [{"src": "ультрамарин", "tr": f"ultramarine-{tag}"}]}
    if "You shorten dubbing lines" in system:
        cur = json.loads(user)
        return {"variants": _shorter(str(cur["current"]), 2)}
    data = json.loads(user)
    segs = data["segments"] if isinstance(data, dict) else data
    if "bilingual QA judge" in system:
        return {"segments": [{"id": s["id"],
                              "score": 5 if _h(s["id"]) % 3 else 2,
                              "issue": "ok" if _h(s["id"]) % 3 else "drift"}
                             for s in segs]}
    if '"issues"' in system and "draft" in system and "finalizing" not in system:
        return {"segments": [{"id": s["id"],
                              "issues": "ok" if _h(s["id"]) % 2 else "shorter"}
                             for s in segs]}
    out = []
    for s in segs:
        base = s.get("draft") or _translate(str(s.get("src", "")), tag, s["id"])
        text = str(base)
        if s.get("issues") not in (None, "ok"):
            text = " ".join(text.split()[:-1]) or text
        out.append({"id": s["id"], "text": text, "variants": _shorter(text, 2)})
    return {"segments": out}


# --- edge_tts -----------------------------------------------------------------

class Communicate:
    def __init__(self, text: str, voice: str, **_kw: object) -> None:
        self.text, self.voice = text, voice

    async def save(self, path: str) -> None:
        # a harmonic "voice" with true FM vibrato (phase = integral of f0), so
        # pyin sees voiced frames and f0 liveliness varies from text to text
        dur = max(0.4, round(0.065 * len(self.text), 3))
        f0 = 110 + _h(self.voice) % 80
        depth = 15 + _h(self.text) % 50
        rate = 1.5 + (_h(self.text) % 5) * 0.5
        ph = f"(2*PI*{f0}*t-{depth}/{rate}*cos(2*PI*{rate}*t))"
        expr = (f"0.3*(sin({ph})+0.5*sin(2*{ph})+0.33*sin(3*{ph})"
                f"+0.25*sin(4*{ph}))*(0.6+0.4*sin(2*PI*0.7*t))")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"aevalsrc='{expr}':s=24000:d={dur}",
                        "-c:a", "libmp3lame", "-b:a", "48k",
                        "-fflags", "+bitexact", path], check=True)


# --- speechbrain / distillmos -------------------------------------------------

class _Ecapa:
    def encode_batch(self, wav: torch.Tensor) -> torch.Tensor:
        import torch
        x = wav.reshape(-1).float()
        chunks = torch.tensor_split(x.abs(), 192)
        feats = torch.stack([c.mean() if c.numel() else torch.tensor(0.0)
                             for c in chunks])
        return feats.reshape(1, 1, 192) + 1e-3


class EncoderClassifier:
    @classmethod
    def from_hparams(cls, source: str, savedir: str = "",
                     run_opts: dict[str, str] | None = None) -> _Ecapa:
        return _Ecapa()


class ConvTransformerSQAModel:
    def eval(self) -> "ConvTransformerSQAModel":
        return self

    def to(self, device: str) -> "ConvTransformerSQAModel":
        return self

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        import torch
        x = wav.reshape(-1).float()
        rms = float(x.pow(2).mean().sqrt())
        zc = float((x[1:] * x[:-1] < 0).float().mean()) if x.numel() > 1 else 0.0
        return torch.tensor(1.0 + 4.0 / (1.0 + np.exp(-(8 * rms - 20 * zc))))


# --- install ------------------------------------------------------------------

def _module(name: str, **attrs: object) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def install(full: bool = False) -> None:
    """Put the fakes into sys.modules (call before anything imports them)."""
    mods = {
        "audio_separator": _module("audio_separator"),
        "audio_separator.separator": _module("audio_separator.separator",
                                             Separator=Separator),
        "faster_whisper": _module("faster_whisper", WhisperModel=WhisperModel),
        "edge_tts": _module("edge_tts", Communicate=Communicate),
    }
    if not full:
        mods.update({
            "speechbrain": _module("speechbrain"),
            "speechbrain.inference": _module("speechbrain.inference"),
            "speechbrain.inference.speaker": _module(
                "speechbrain.inference.speaker",
                EncoderClassifier=EncoderClassifier),
            "distillmos": _module("distillmos",
                                  ConvTransformerSQAModel=ConvTransformerSQAModel),
        })
    sys.modules.update(mods)
    import openai
    openai.OpenAI = OpenAI          # s3 imports it inside functions


def request_digest() -> list[dict[str, object]]:
    """Recorded LLM requests in a canonical order (threads may interleave)."""
    rows = [dict(r, sha=hashlib.sha256(json.dumps(r, sort_keys=True,
                                                  ensure_ascii=False)
                                       .encode()).hexdigest()[:16])
            for r in LLM_LOG]
    return sorted(rows, key=lambda r: str(r["sha"]))
