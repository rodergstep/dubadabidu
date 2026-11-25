#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# How tests/fixtures/neutrality/clip.mp4 was made (ffmpeg 8.1.2). The clip is
# COMMITTED rather than generated per run, so its bytes are the same on every
# machine; re-running this script with another ffmpeg will not reproduce them.
# 6 s, 160x90 @ 5 fps, mono AAC: a frequency-modulated voice-band tone with
# three pauses (three "utterances") over low noise.
set -e
cd "$(dirname "$0")"
ffmpeg -y -loglevel error \
  -f lavfi -i "testsrc=size=160x90:rate=5:duration=6" \
  -f lavfi -i "aevalsrc='(0.4*sin(2*PI*(150+30*sin(2*PI*3*t))*t))*(between(t,0.3,1.8)+between(t,2.3,3.9)+between(t,4.4,5.7))+0.01*sin(2*PI*1234*t)':s=44100:d=6" \
  -c:v libx264 -preset veryslow -crf 40 -pix_fmt yuv420p \
  -c:a aac -b:a 48k -ac 1 -shortest -map_metadata -1 -fflags +bitexact clip.mp4
