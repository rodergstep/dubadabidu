# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu remote` from the CLI: stage plan, batches, --keep-going."""
from tests.cfgs import settings


class _FakeRpi:
    """Stands in for pipeline.runpod_infra in the batch tests: records the
    call and marks the pod stages done in each "synced-back" manifest."""

    def __init__(self, tmp_path, fail: frozenset[str] = frozenset()) -> None:
        self.tmp = tmp_path
        self.calls: list[tuple[list[str], str, bool, bool]] = []
        self.fail = fail

    def remote_run(self, cfg: dict, videos: list[str], langs: list[str],
                   task: str, budget: float | None, keep_alive: bool,
                   reuse: bool, overlays: list[str], stages: str) -> bool:
        import json
        self.calls.append((list(videos), stages, keep_alive, reuse))
        for v in videos:
            st = {} if v in self.fail else {
                f"s{n}_{lg}": "done" for n in (4, 5) for lg in langs}
            d = self.tmp / v
            d.mkdir(exist_ok=True)
            (d / "manifest.json").write_text(json.dumps({"stages": st}))
        return not self.fail


def _batch_args(**kw: object):
    import argparse
    base = {"from_stage": "s1_extract", "to_stage": "s8_mux", "budget": None,
            "keep_alive": False, "reuse": False, "overlay": []}
    base.update(kw)
    return argparse.Namespace(**base)


def test_remote_stages_reads_the_real_argparse_dests():
    """argparse stores --from/--to as from_stage/to_stage. Reading "from"/"to"
    returned None for both, so the pod fell back to the whole pipeline, ran
    s1/s2 and died on the input video that is deliberately never uploaded."""
    import argparse
    from dubadabidu.cli.remote import _remote_stages
    narrowed = argparse.Namespace(from_stage="s4_synthesize",
                                  to_stage="s4_synthesize")
    assert _remote_stages(narrowed) == "--from s4_synthesize --to s4_synthesize"
    # untouched flags stop the pod at s5: s4 is the only stage needing a GPU,
    # and s6/s7 are CPU work on files that sync both ways. Was s7 until
    # 2026-08-09, when s6's pedalboard import (the [master] extra, never
    # installed on a pod) failed a run in which all five languages had already
    # synthesized and fitted successfully.
    default = argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux")
    assert _remote_stages(default) == "--from s4_synthesize --to s5_fit"
    # a later --to cannot drag the pod past s5: s6..s8 run locally instead
    further = argparse.Namespace(from_stage="s1_extract",
                                 to_stage="s7_subtitles")
    assert _remote_stages(further) == "--from s4_synthesize --to s5_fit"


def test_remote_plan_translates_locally_and_finishes_locally():
    """2026-11-24: `remote run` with untouched flags re-ran s2 ON THE POD
    (s2 has no cache: it discards every translation) and crashed there; and
    the local follow-up muxed without s6, i.e. stale dub_<lang>.m4a files.
    The plan is now local s3 -> pod s4..s5 -> local s6..s8."""
    import argparse
    from dubadabidu.cli.remote import _remote_plan
    before, pod, after = _remote_plan(
        argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux"))
    assert before == ["s3_translate"]
    assert pod == ["s4_synthesize", "s5_fit"]
    assert after == ["s6_mix", "s7_subtitles", "s8_mux"]
    # an explicit --from s2 is honoured, locally
    before, pod, _ = _remote_plan(
        argparse.Namespace(from_stage="s2_transcribe", to_stage="s8_mux"))
    assert before == ["s2_transcribe", "s3_translate"]
    # nothing for the GPU -> no pod at all
    _, pod, after = _remote_plan(
        argparse.Namespace(from_stage="s6_mix", to_stage="s8_mux"))
    assert pod == [] and after == ["s6_mix", "s7_subtitles", "s8_mux"]


def test_multi_video_remote_run_is_one_pod_one_call(monkeypatch, tmp_path):
    """All videos go to the pod in ONE remote_run (one process: models load
    once); s3 runs locally for all first, s6..s8 for all after."""
    from dubadabidu.cli import remote as dub
    local: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(dub, "_run_local",
                        lambda cfg, v, langs, st: local.append((v, tuple(st))))
    rpi = _FakeRpi(tmp_path)
    cfg = settings({"work_dir": str(tmp_path)})
    assert dub._remote_run_batch(cfg, ["a", "b", "c"], ["en"], _batch_args(), rpi)
    assert rpi.calls == [(["a", "b", "c"], "--from s4_synthesize --to s5_fit",
                          False, False)]
    assert [v for v, st in local if st == ("s3_translate",)] == ["a", "b", "c"]
    assert [v for v, st in local if st[:1] == ("s6_mix",)] == ["a", "b", "c"]


def test_batch_video_that_did_not_finish_on_the_pod_skips_its_finish(
        monkeypatch, tmp_path):
    from dubadabidu.cli import remote as dub
    local: list[str] = []
    monkeypatch.setattr(dub, "_run_local",
                        lambda cfg, v, langs, st: local.append(f"{v}:{st[0]}")
                        if st else None)
    rpi = _FakeRpi(tmp_path, fail=frozenset({"b"}))
    cfg = settings({"work_dir": str(tmp_path)})
    assert not dub._remote_run_batch(cfg, ["a", "b", "c"], ["en"],
                                     _batch_args(), rpi)
    assert "a:s6_mix" in local and "c:s6_mix" in local
    assert "b:s6_mix" not in local


def test_pod_command_keeps_going_and_carries_every_video():
    """The pod-side `run` must not stop at the first bad video, and every
    path is quoted (lesson names have spaces)."""
    import shlex
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    assert "--keep-going" in REMOTE_TASK["run"]
    vids = ["input/a b.mp4", "input/c.mp4"]
    cmd = REMOTE_TASK["run"].format(
        video=" ".join(shlex.quote(v) for v in vids), langs="en", stages="")
    assert "'input/a b.mp4' input/c.mp4" in cmd


def test_run_keep_going_continues_past_a_failing_video(monkeypatch, tmp_path):
    """dub.py run with --keep-going: video 1 raises, video 2 still runs, and
    the process exits non-zero naming the failure."""
    import sys
    import pytest
    from dubadabidu import cli as dub
    seen: list[str] = []

    def stage(cfg, v, langs):
        seen.append(v)
        if v.endswith("bad.mp4"):
            raise RuntimeError("synth exploded")
    monkeypatch.setattr(dub, "STAGES", {n: stage for n in dub.ORDER})
    for n in ("bad", "good"):
        (tmp_path / f"{n}.mp4").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "dubadabidu", "run", str(tmp_path / "bad.mp4"), str(tmp_path / "good.mp4"),
        "--langs", "en", "--from", "s4_synthesize", "--to", "s4_synthesize",
        "--keep-going"])
    with pytest.raises(SystemExit, match="1 video"):
        dub.main()
    assert seen == [str(tmp_path / "bad.mp4"), str(tmp_path / "good.mp4")]
