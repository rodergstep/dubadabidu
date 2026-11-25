# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stage table: name -> how the CLI calls the stage. Order is run order."""
from __future__ import annotations

from dubadabidu.stages import s1_extract
from dubadabidu.stages import s2_transcribe
from dubadabidu.stages import s3_translate
from dubadabidu.stages import s4_synthesize
from dubadabidu.stages import s5_fit
from dubadabidu.stages import s6_mix
from dubadabidu.stages import s7_subtitles
from dubadabidu.stages import s8_mux

STAGES = {
    "s1_extract":    lambda c, v, l: s1_extract.run(c, v),
    "s2_transcribe": lambda c, v, l: s2_transcribe.run(c, v),
    "s3_translate":  lambda c, v, l: s3_translate.run(c, v, l),
    "s4_synthesize": lambda c, v, l: s4_synthesize.run(c, v, l),
    "s5_fit":        lambda c, v, l: s5_fit.run(c, v, l),
    "s6_mix":        lambda c, v, l: s6_mix.run(c, v, l),
    "s7_subtitles":  lambda c, v, l: s7_subtitles.run(c, v, l),
    "s8_mux":        lambda c, v, l: s8_mux.run(c, v, l),
}
ORDER = list(STAGES)
