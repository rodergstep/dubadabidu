"""Course runner: many videos, ONE pod."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.course import _videos, _stage_done  # noqa: E402


def test_video_discovery_walks_dirs_and_filters_by_extension(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mov").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mkv").touch()
    found = {Path(v).name for v in _videos([str(tmp_path)])}
    assert found == {"a.mp4", "b.mov", "c.mkv"}


def test_stage_done_is_per_language(tmp_path):
    """A course is only resumable if 'done' means done for EVERY language —
    otherwise a partially-translated video is skipped and synthesized with
    missing text."""
    wd = tmp_path / "lesson"
    wd.mkdir()
    (wd / "manifest.json").write_text(json.dumps(
        {"stages": {"s3_en": "done", "s3_ru": "12/62"}, "utterances": []}))
    assert not _stage_done("lesson.mp4", ["en", "ru"], "s3_translate", str(tmp_path))
    assert _stage_done("lesson.mp4", ["en"], "s3_translate", str(tmp_path))


def test_stage_done_false_when_nothing_ran(tmp_path):
    assert not _stage_done("x.mp4", ["en"], "s3_translate", str(tmp_path))


def test_stage_done_survives_a_corrupt_manifest(tmp_path):
    """A half-written manifest must read as 'not done' rather than crash the
    whole course — the point of the runner is that one bad video does not cost
    the others their pod."""
    wd = tmp_path / "lesson"
    wd.mkdir()
    (wd / "manifest.json").write_text("{ this is not json")
    assert not _stage_done("lesson.mp4", ["en"], "s3_translate", str(tmp_path))


def test_gpu_phase_is_narrowed_to_s4_and_reuses_one_pod():
    """The whole reason this exists: `remote run` per video would provision,
    bootstrap, install qwen and download weights EVERY time (~5 min each). The
    GPU phase must pass --reuse --keep-alive and narrow the pod to s4."""
    src = (Path(__file__).resolve().parents[1] / "pipeline" / "course.py").read_text()
    gpu_call = src[src.index('"remote", "run"'):src.index("finally:")]
    for flag in ('"--reuse"', '"--keep-alive"',
                 '"--from", "s4_synthesize"', '"s4_synthesize"'):
        assert flag in gpu_call, f"GPU phase missing {flag}"


def test_pod_is_killed_in_a_finally_block():
    """--keep-alive leaves the pod up on purpose so the next video attaches;
    nothing else will take it down."""
    src = (Path(__file__).resolve().parents[1] / "pipeline" / "course.py").read_text()
    after = src[src.index("finally:"):]
    assert '"remote", "kill"' in after
    assert "leaked pod" in after, "a failed kill must be surfaced, not swallowed"
