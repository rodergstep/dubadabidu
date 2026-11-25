# SPDX-License-Identifier: AGPL-3.0-or-later
"""edge-tts: free Microsoft neural voices, CPU-only, no cloning."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from dubadabidu.core.config.schema import Tts


def _synth_edge(text: str, lang: str, out: Path, t: Tts) -> None:
    import edge_tts
    voice = t.edge_voices[lang]
    mp3 = out.with_suffix(".tmp.mp3")
    asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", str(t.sample_rate), "-ac", "1", str(out)], check=True)
    mp3.unlink(missing_ok=True)
