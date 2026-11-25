# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segment-edge hygiene shared by s6 (the mix) and take selection.

s6 trims every placed segment at TRIM_DB and fades its edges (synth takes
carry leading silence and can end at full level -> clicks between sentences).
Take selection measures f0 on the take AS DELIVERED, i.e. after this exact
trim, so both import it from here rather than selection reaching into s6.
"""
from __future__ import annotations

from pydub import AudioSegment, silence

TRIM_DB = -45          # edge-trim threshold
TRIM_CUSHION_MS = 30   # keep this much of the quiet edge
FADE_IN_MS, FADE_OUT_MS = 10, 60  # long fade-out: end-of-generation garble hides
                                  # above the trim threshold right at sentence joins


def trim_and_fade(seg: AudioSegment) -> AudioSegment:
    lead = silence.detect_leading_silence(seg, silence_threshold=TRIM_DB)
    tail = silence.detect_leading_silence(seg.reverse(), silence_threshold=TRIM_DB)
    seg = seg[max(0, lead - TRIM_CUSHION_MS):
              len(seg) - max(0, tail - TRIM_CUSHION_MS)]
    return seg.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)
