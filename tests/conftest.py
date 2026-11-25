# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every test runs with the project root as its working directory.

That is what the CLI guarantees (it changes into the project before any stage
runs, and stages resolve work/, glossary/, ref/ and the configs from
there), so the suite behaves the same whichever directory pytest is started
from. A test that needs another directory still monkeypatches chdir itself.
"""
from __future__ import annotations

import pytest

from tests.paths import REPO


@pytest.fixture(autouse=True)
def _project_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
