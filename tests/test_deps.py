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


# --- FINDINGS.md / runs.jsonl: the knowledge ledger --------------------------

def test_findings_entries_carry_evidence_and_a_verdict():
    """FINDINGS.md is only worth having if every claim is falsifiable. An entry
    without EVIDENCE is an opinion; one without a VERDICT is an open loop that
    reads as settled. Both failure modes already happened in this repo's
    comments — four claims this week asserted behaviour nobody had checked."""
    text = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    claims = text.count("**CLAIM**")
    verdicts = text.count("**VERDICT**")
    evidence = text.count("**EVIDENCE**")
    assert claims >= 8, f"only {claims} claims — findings are drifting elsewhere"
    assert verdicts >= claims, (
        f"{claims} CLAIM vs {verdicts} VERDICT — a claim without a verdict reads "
        f"as settled when it is not")
    assert evidence >= claims - 1, (
        f"{claims} CLAIM vs {evidence} EVIDENCE — an entry without numbers is an "
        f"opinion, and this file exists because opinions were re-derived as fact")
    for word in ("CONFIRMED", "REFUTED", "OPEN"):
        assert word in text, f"verdict vocabulary lost: {word}"


def test_findings_leads_with_the_noise_floor():
    """Every measurement in the file is unreadable without it — two prior
    results were retroactively invalidated once the floor was known."""
    text = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    head = text[:text.index("## 1.")]
    assert "noise floor" in head.lower(), "the noise floor must come first"
    for band in ("sim ±0.010", "mos ±0.007", "f0st ±0.438"):
        assert band in head, f"missing measured band: {band}"


def test_run_ledger_is_machine_written_not_hand_maintained():
    """runs.jsonl must be appended by the code that spends the money. A
    hand-kept stats file rots, which is how 'what does an hour cost' got three
    different answers in one day."""
    src = (ROOT / "pipeline" / "runpod_infra.py").read_text(encoding="utf-8")
    assert "runlog.append" in src, (
        "remote_run no longer records the run — the ledger will silently stop "
        "reflecting reality")
    i = src.index("runlog.append")
    assert "finally" in src[:i].rsplit("def remote_run", 1)[-1], (
        "the ledger write must sit in the finally block: a run that dies after "
        "30 min of bootstrap is exactly the data point that never survives")


def test_ledger_cost_helper_ignores_runs_with_no_audio():
    """A setup-check has no video to divide by; including it would silently
    inflate $/video-hour."""
    from pipeline.runlog import cost_per_video_hour
    rows = [{"task": "setup-check", "wall_s": 200, "price_per_hr": 0.28,
             "langs": []},
            {"task": "run", "wall_s": 2466, "price_per_hr": 0.28,
             "video_minutes": 8.13, "langs": ["en", "ru"], "bootstrap_s": 396}]
    out = cost_per_video_hour(rows)
    assert out["runs"] == 1, "setup-check must not count as a dubbing run"
    assert abs(out["total_cost_usd"] - 0.192) < 0.002


def test_declared_pins_satisfy_the_engine_that_installs_later():
    """qwen installs from runpod.engine_setup, AFTER `pip install -e '.[dev]'`,
    so anything this project declares must already satisfy it or the engine
    install cannot be resolved.

    This is not hypothetical: ruaccent declares `transformers` unpinned, and
    adding it on 2026-08-03 let pip pull 5.14.1 into .[dev]. faster-qwen3-tts
    needs <5, so the pod came up with `qwen unavailable` and a whole ru-stress
    A/B measured nothing."""
    proj = _pyproject_core()
    tf = proj.get("transformers")
    assert tf, ("transformers must be pinned here — ruaccent pulls it unpinned "
                "and will resolve past the engine's <5 ceiling")
    major, minor = (int(x) for x in tf.split(".")[:2])
    assert (major, minor) >= (4, 57), f"faster-qwen3-tts needs >=4.57, got {tf}"
    assert major < 5, f"faster-qwen3-tts needs <5, got {tf}"


def test_ruaccent_is_declared_where_synthesis_runs():
    """It is used at the synthesis boundary, and synthesis runs on the pod,
    which installs `.[dev]` only — an extra would never be present there."""
    proj = _pyproject_core()
    assert "ruaccent" in proj, (
        "ruaccent moved out of core: tts.ru_stress would silently no-op on the "
        "pod, because normalize_for_tts swallows the ImportError by design")


def test_reused_pods_reach_the_ledger_with_a_price():
    """--reuse skips provision(), which is where the ledger's price_per_hr is
    stamped. Without a re-stamp every reused run costs $0.000 in runs.jsonl —
    silent UNDER-reporting of precisely the runs --keep-alive encourages, and
    cost_per_video_hour averages the zeros in. Measured 2026-08-09: two real
    A/B arms billed ~20 min of pod time and recorded as free."""
    src = (ROOT / "pipeline" / "runpod_infra.py").read_text(encoding="utf-8")
    body = src.split("alive = _live_pod() if reuse else None", 1)[1]
    branch = body.split("else:", 1)[0]
    assert "_LAST_POD.update" in branch, (
        "the --reuse branch does not stamp _LAST_POD, so its runs land in "
        "runs.jsonl with price_per_hr 0 and cost $0.000")
    assert "costPerHr" in branch, (
        "the reuse branch must read the REAL rate from the API, not fall back "
        "to assumed_price_per_hr (deliberately the worst case, ~2.5x high)")


def test_ru_icl_arms_differ_in_exactly_one_axis():
    """Arm B (marked) and arm C (unmarked) exist to isolate the stress marks
    from ICL itself. That only works if the reference, the trim and the
    transcript are identical apart from the acutes — the mistake
    config.exp.ru-stress.yaml originally shipped with was a reference pinned
    from a different experiment, which measured the reference, not the axis."""
    import yaml
    b = yaml.safe_load((ROOT / "config.exp.ru-icl-stress.yaml").read_text(
        encoding="utf-8"))["tts"]
    c = yaml.safe_load((ROOT / "config.exp.ru-icl-plain.yaml").read_text(
        encoding="utf-8"))["tts"]
    for k in ("reference_wav", "reference_max_s", "qwen_x_vector_only"):
        assert b[k] == c[k], f"arms differ in {k}: {b[k]!r} vs {c[k]!r}"
    assert b["ru_stress"] is True and c["ru_stress"] is False
    acute = "́"
    assert acute in b["reference_text"], "arm B's reference must be MARKED"
    assert acute not in c["reference_text"], "arm C's reference must be PLAIN"
    assert b["reference_text"].replace(acute, "") == c["reference_text"], (
        "the two transcripts must be the same text apart from the acutes, or "
        "the comparison measures the transcript rather than the marks")
    # ICL needs a transcript; x_vector_only true would silently ignore it and
    # the whole experiment would degrade into the already-refuted test.
    assert b["qwen_x_vector_only"] is False, (
        "ICL off means ref_text is dropped — that is the REFUTED configuration")


def test_eval_weights_sum_to_one_and_sim_stays_out():
    """sim was measured to be noise against 226 ratings (en p 0.42, ru p 0.15)
    and removed 2026-08-09. Re-adding it needs new evidence, not a tidy-up: it
    has failed to predict this listener five separate times, including an ICL
    round it won by 17x the noise band and the ear rated 15/18 unusable."""
    import yaml
    w = yaml.safe_load((ROOT / "config.yaml").read_text(
        encoding="utf-8"))["qc"]["eval"]["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights must sum to 1: {w}"
    assert w["sim"] == 0.0, (
        "sim is back in the take-selection objective — it does not predict the "
        "listener in either language")
    assert w["f0"] > 0, (
        "f0st is the only feature significant in BOTH languages (en p 0.0016, "
        "ru p 0.0386); bakeoff excludes it for RUN-level ranking, which is a "
        "different comparison with a different noise floor")


def test_take_ranking_actually_reads_the_configured_weights():
    """qc.eval.weights was read ONLY by tune.py, while tts_engine._take_rank
    hardcoded its own copy — so editing the weights changed the reference-sweep
    objective and nothing about which take ships. refit gated on a formula the
    pipeline never ran, and dropping sim to 0.0 left it at an effective 0.2 in
    selection. Wired up 2026-08-09; this keeps them connected."""
    from pipeline.tts_engine import _take_rank
    m_hi_sim = {"mos_min": 3.0, "f0st": 2.0, "sim": 0.9}
    m_lo_sim = {"mos_min": 3.0, "f0st": 2.0, "sim": 0.1}
    sim_heavy = {"sim": 1.0, "mos": 0.0, "f0": 0.0, "tempo": 0.0}
    no_sim = {"sim": 0.0, "mos": 1.0, "f0": 0.0, "tempo": 0.0}
    # with all weight on sim the high-sim take must win
    assert _take_rank(m_hi_sim, None, sim_heavy) > _take_rank(m_lo_sim, None, sim_heavy)
    # with sim at zero the two must tie — otherwise sim still leaks in
    assert _take_rank(m_hi_sim, None, no_sim) == _take_rank(m_lo_sim, None, no_sim)


def test_zero_weights_do_not_disable_best_of():
    """If every applicable weight is 0 the rank must not become a constant —
    `max` would then return the FIRST take every time and best_of would quietly
    stop selecting at all."""
    from pipeline.tts_engine import _take_rank
    zero = {"sim": 0.0, "mos": 0.0, "f0": 0.0, "tempo": 0.0}
    better = {"mos_min": 4.0, "f0st": 2.0, "sim": 0.5}
    worse = {"mos_min": 2.0, "f0st": 2.0, "sim": 0.5}
    assert _take_rank(better, None, zero) > _take_rank(worse, None, zero)


def test_s4_and_s5_pass_the_weights_through():
    """The engine takes a TTS-config dict and the weights live under qc, so the
    lookup has to happen at the call site. If a caller forgets, that language
    silently reverts to DEFAULT_RANK_W."""
    for mod in ("s4_synthesize", "s5_fit"):
        src = (ROOT / "pipeline" / f"{mod}.py").read_text(encoding="utf-8")
        assert "rank_weights=" in src, f"{mod} does not pass rank_weights"


def test_the_closed_list_survives_and_stays_near_the_top():
    """FINDINGS opens with a CLOSED list because this project has re-derived the
    same answers more than once — "what does an hour cost" got three numbers in
    one day, and a refuted hypothesis was re-proposed twice. It only works if it
    is the first thing read, so it must stay above the detail sections."""
    text = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    assert "## CLOSED — do not re-attempt" in text
    assert text.index("## CLOSED") < text.index("## 1. ASR"), (
        "the closed list moved below the detail — it is an index, and an index "
        "nobody reaches first is not one")
    # the expensive dead ends, by the words a future session would search for
    for closed in ("Spot instances", "consensus", "RUAccent marks",
                   "Reference choice as a quality lever"):
        assert closed in text, f"{closed!r} dropped out of the closed list"


def test_russian_ships_as_is_is_recorded_as_a_DECISION():
    """A known defect that is shipped on purpose has to be written down, or the
    next session reads ~29% wrong stress as a bug to chase — which is what four
    failed experiments already cost."""
    text = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    assert "ru ships with the defect" in text
    assert "SYSTEMATIC" in text, (
        "the REASON must survive alongside the decision: the errors are partly "
        "systematic, so no selection rule can reach them")


def test_no_comment_states_a_stale_runtime_cap():
    """Comments that assert a value the code no longer has are this project's
    most repeated bug — early_accept_mos calibrated for a dead engine, the
    adoption gate with no incumbent, production routed to chatterbox, and three
    comments still claiming max_runtime_hours "caps a run at 3 h" after it was
    raised to 6.0 on 2026-08-02.

    Deliberately narrow: it matches PRESENT-TENSE claims of a cap value
    ("caps a run at N h", "capped at ... = N h"). Historical notes phrased as
    "the old 3.0 h watchdog" are legitimate and must keep working, so they are
    not matched.
    """
    import re
    import yaml
    text = (ROOT / "config.gpu.yaml").read_text(encoding="utf-8")
    actual = yaml.safe_load(text)["runpod"]["max_runtime_hours"]
    claims = re.findall(r"(?:caps? a run at|capped at[^\n]*?=)\s*([\d.]+)\s*h",
                        text)
    bad = [c for c in claims if abs(float(c) - float(actual)) > 1e-9]
    assert not bad, (
        f"comment(s) claim a runtime cap of {bad} h while "
        f"max_runtime_hours is {actual}")


def _doctor_lines(monkeypatch, *, device, runpod_key):
    """Run doctor with a faked torch device / env and capture what it printed."""
    import io
    import contextlib
    import dub
    monkeypatch.setattr(dub, "torch_device", lambda: device)
    if runpod_key:
        monkeypatch.setenv("RUNPOD_API_KEY", "x")
    else:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    import yaml
    from pipeline.logic import deep_merge
    cfg = deep_merge(yaml.safe_load((ROOT / "config.yaml").read_text()),
                     yaml.safe_load((ROOT / "config.gpu.yaml").read_text()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dub.doctor(cfg)
    return rc, buf.getvalue()


def test_doctor_does_not_fail_over_a_pod_side_engine_on_a_non_cuda_box(monkeypatch):
    """qwen missing on a Mac is CORRECT: only s4 runs on the pod, which installs
    it from runpod.engine_setup. Failing for it ended every local run in
    "doctor: fix items above" over a thing nobody should fix — and a check that
    reports non-problems is one people stop reading."""
    rc, out = _doctor_lines(monkeypatch, device="mps", runpod_key=True)
    assert "[!!] python: qwen_tts" not in out
    assert "not installed locally" in out


def test_doctor_still_fails_when_there_is_nowhere_to_synthesize(monkeypatch):
    """No CUDA and no remote configured means synthesis cannot happen anywhere.
    That is a real fault and must stay loud — otherwise this fix has just
    replaced a false alarm with a silent one."""
    rc, out = _doctor_lines(monkeypatch, device="mps", runpod_key=False)
    assert "[!!] python: qwen_tts" in out and rc != 0


def test_doctor_still_fails_on_a_cuda_box(monkeypatch):
    """On a GPU machine the engine really is required."""
    rc, out = _doctor_lines(monkeypatch, device="cuda", runpod_key=True)
    assert "[!!] python: qwen_tts" in out and rc != 0
