# Reference clips — record 3-4 CANDIDATES, the tune loop picks the winner (R1).
#
# Name them roderg_ref_02.wav, roderg_ref_03.wav, ... (tune.refs_glob = ref/*.wav)
# (roderg_ref_01.wav was a 7s clip — retired; too short, weak timbre match.)
#
# EXCLUDING a bad candidate: the tune glob is ref/*.wav (top level only), so
# move a weak clip into ref/retired/ (or delete it) and it drops out of ref
# selection, calibration, and everything downstream. Retired for bad quality:
# test_ref_01/02/03.wav and roderg_ref_04.wav (deleted 2026-07-14).
#
# Each candidate:
#   - 15-20 seconds (NOT less — 7s is why timbre match was weak)
#   - quiet room, no music, no reverb; mono, 24 kHz or higher
#   - expressive but natural read, like narrating a lesson — vary pitch a little
#   - different candidates = different takes/moods (calm, energetic, mid-range)
#   - trim silence/breaths at start and end
#
# Quick capture check: play it back on headphones; if you hear room echo or
# hiss, re-record — the clone inherits every artifact.
