# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every place a code default disagrees with the schema, and what happens to it.

Keys are inventory rows ("path kind default"); test_equivalence fails on a row
that drifts without an entry here, and on an entry that no longer drifts.
Decisions are the ones approved in ARCHITECTURE_PLAN.md §3.3 (Q8). Tags:

  SCHEMA      move the code's value into the schema as the field default
              (effective value unchanged; the code default is deleted)
  SCHEMA-WINS latent: shipped configs always set the key, so the schema's
              documented default becomes the only one
  CONST       not config: a named module constant (the key is undeclared, so
              no config could ever set it)
  HASH        a persisted cache-key format constant, kept inside synth_hash
  ABSENT      differs only when the whole block is missing from the config
  EQUIV       the site's comparison makes the two values equivalent
  BUG         preserved during the refactor; fixed separately if approved
"""
from __future__ import annotations

DRIFT: dict[str, str] = {
    # --- bakeoff (block present only with the gpu overlay) ---
    "bakeoff get {}": "ABSENT (Q8c): Config.bakeoff stays optional; each site "
                      "keeps its absent-block meaning through one helper",
    "bakeoff.engines get []": "ABSENT (Q8c)",
}
