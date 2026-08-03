"""requirements.txt is a DERIVED file — pyproject.toml is the source of truth.

It exists only because `doctor` and the README hand a human one copy-pasteable
line, and a derived file with nothing checking it drifts silently. It already
did once, badly: the chatterbox-era requirements.txt outlived the engine's
removal (2026-08-02) and still installed chatterbox-tts==0.1.7, whose hard pin
would have DOWNGRADED the torch 2.11.0 the pod bootstrap installs. `doctor`
pointed users straight at it.

These tests are what the requirements.txt header claims exists.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# torch/torchaudio are deliberately in NEITHER file: the pod bootstrap installs
# them from the cu128 index before this project, and audio-separator's
# unbounded torch>=2.3 would otherwise pull 2.13 and break the torchaudio pair.
NOT_DECLARED = {"torch", "torchaudio"}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _requirements() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:==\s*([^\s;]+))?", line)
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def _pyproject_core() -> dict[str, str | None]:
    """Core `dependencies` only — extras are listed as comments, not installs."""
    text = (ROOT / "pyproject.toml").read_text()
    # Read to the line that IS the closing bracket. Splitting on the first "]"
    # truncated the block at a comment mentioning `.[sep]`, which silently made
    # this test compare against three packages and pass a lie.
    lines, block, inside = text.splitlines(), [], False
    for line in lines:
        if line.strip().startswith("dependencies = ["):
            inside = True
            continue
        if inside:
            if line.strip() == "]":
                break
            block.append(line)
    out: dict[str, str | None] = {}
    for line in block:
        line = line.split("#", 1)[0].strip().rstrip(",").strip('"').strip("'")
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:==\s*([^\s;]+))?", line)
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def test_same_packages_in_both_files():
    req, proj = _requirements(), _pyproject_core()
    missing = sorted(set(proj) - set(req) - NOT_DECLARED)
    extra = sorted(set(req) - set(proj) - NOT_DECLARED)
    assert not missing, (
        f"in pyproject but not requirements.txt: {missing} — a human following "
        f"`doctor`'s hint would get an incomplete environment")
    assert not extra, (
        f"in requirements.txt but not pyproject: {extra} — pyproject is the "
        f"source of truth, so this installs something the project does not "
        f"declare (this is how chatterbox-tts outlived its own removal)")


def test_same_versions_in_both_files():
    req, proj = _requirements(), _pyproject_core()
    drift = {k: (proj[k], req[k]) for k in set(proj) & set(req)
             if proj[k] != req[k]}
    assert not drift, (
        "pinned differently in pyproject vs requirements.txt "
        f"(pyproject, requirements): {drift} — requirements.txt is DERIVED; "
        f"change the pin in pyproject and mirror it here")


def test_torch_is_pinned_in_neither():
    """The pod bootstrap owns the torch pin. A pin here would either fight it
    or be silently overridden, and an UNBOUNDED torch lets audio-separator pull
    2.13 past the torchaudio 2.11 ceiling."""
    req, proj = _requirements(), _pyproject_core()
    for name in NOT_DECLARED:
        assert name not in req, f"{name} must not be pinned in requirements.txt"
        assert name not in proj, f"{name} must not be pinned in pyproject"


def test_removed_engines_stay_removed():
    """The failure this file exists for: requirements.txt kept installing an
    engine the repo had deleted, and its transitive pin downgraded torch."""
    text = (ROOT / "requirements.txt").read_text()
    body = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    for gone in ("chatterbox", "cosyvoice", "voxcpm", "indextts"):
        assert gone not in body.lower(), (
            f"{gone} was removed from this repo but requirements.txt still "
            f"installs it")
