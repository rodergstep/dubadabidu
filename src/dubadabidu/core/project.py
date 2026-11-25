# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which directory is the project: where config.yaml, glossary/, ref/, work/,
output/, the ledger and .env live (ARCHITECTURE_PLAN.md §6).

Resolution, first match wins:
  1. --project DIR
  2. $DUBADABIDU_HOME
  3. the nearest ancestor of the working directory that holds config.yaml
     (git-style walk-up, so any subdirectory of a clone works)
  4. the source checkout this package runs from, if it holds config.yaml —
     an editable install used from anywhere, which is what `dub.py`'s
     os.chdir(ROOT) gave every clone user before
  5. otherwise: a clear error, never site-packages

The CLI changes into the resolved root once, before any stage runs: relative
config values are themselves cache-key inputs (synth_hash keys on the literal
`tts.reference_wav`, e.g. "ref/sketch_ref_08.wav"), so paths stay relative to
the project rather than being rewritten absolute.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MARKER = "config.yaml"
ENV = "DUBADABIDU_HOME"
CHECKOUT = Path(__file__).resolve().parents[3]   # src/dubadabidu/core/ -> repo


@dataclass(frozen=True)
class Project:
    root: Path
    found_by: str        # "--project" | "$DUBADABIDU_HOME" | "cwd" | "checkout"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"


class NoProject(SystemExit):
    """No project directory could be resolved (a clean exit, not a traceback)."""


def resolve(cli: str | None, env: Mapping[str, str], cwd: Path,
            checkout: Path = CHECKOUT) -> Project:
    if cli:
        root = Path(cli).expanduser().resolve()
        if not root.is_dir():
            raise NoProject(f"--project {cli}: not a directory")
        return Project(root, "--project")
    if env.get(ENV):
        root = Path(env[ENV]).expanduser().resolve()
        if not root.is_dir():
            raise NoProject(f"${ENV}={env[ENV]}: not a directory")
        return Project(root, f"${ENV}")
    here = cwd.resolve()
    for d in (here, *here.parents):
        if (d / MARKER).is_file():
            return Project(d, "cwd")
    if (checkout / MARKER).is_file():
        return Project(checkout, "checkout")
    raise NoProject(
        f"no dubadabidu project here: pass --project DIR or set ${ENV} "
        f"(a project is a directory containing {MARKER}; copy the one from the "
        f"repository to start)")
