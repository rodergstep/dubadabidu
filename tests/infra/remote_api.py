# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the remote lifecycle lives TODAY (the split edits only this file)."""
from __future__ import annotations

from types import ModuleType

from dubadabidu.infra.remote import (bootstrap, diagnostics, orchestrator,
                                     runpod, state, transport)

remote_run = orchestrator.remote_run


def time_users() -> list[ModuleType]:
    """Every infra module that reads the clock through its own `time` name."""
    return [m for m in (bootstrap, diagnostics, orchestrator, runpod, state, transport)
            if hasattr(m, "time")]
