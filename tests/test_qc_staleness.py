"""QC staleness stamps (manifest.stamp_qc / stale_qc).

Regression guard for the defect these were added for: s5/s6 rewrite the placed
wav whenever a fit or mix setting changes, and nothing re-scored it — so
report/batch_report/review pages/autopilot._assess and the ratings rows the
qc-weight re-fit trains on all kept reporting scores for audio that no longer
existed. Measured on work/sketch60: stored qc_f0st 2.58 vs 1.70 actual.
"""
from pathlib import Path

from pipeline import manifest as M


def _seg(tmp: Path, name: str, data: bytes) -> str:
    p = tmp / name
    p.write_bytes(data)
    return name


def _man(*trs: dict) -> dict:
    return {"utterances": [{"id": f"u{i:04d}", "tr": {"en": tr}}
                           for i, tr in enumerate(trs, 1)]}


def test_scored_path_prefers_placed_over_fitted(tmp_path):
    tr = {"fitted": "a.wav", "placed": "b.wav"}
    assert M.scored_path(tmp_path, tr) == tmp_path / "b.wav"
    assert M.scored_path(tmp_path, {"fitted": "a.wav"}) == tmp_path / "a.wav"
    assert M.scored_path(tmp_path, {}) is None


def test_unscored_segment_is_missing_not_stale(tmp_path):
    _seg(tmp_path, "a.wav", b"audio")
    man = _man({"placed": "a.wav"})          # no qc_* at all
    assert M.stale_qc(tmp_path, man, "en") == {"score": [], "wer": []}


def test_fresh_stamp_reads_current(tmp_path):
    _seg(tmp_path, "a.wav", b"audio")
    tr = {"placed": "a.wav", "qc_score": 0.7, "qc_wer": 0.1}
    M.stamp_qc(tmp_path, tr, "score")
    M.stamp_qc(tmp_path, tr, "wer")
    assert M.stale_qc(tmp_path, _man(tr), "en") == {"score": [], "wer": []}


def test_rewritten_audio_reads_stale(tmp_path):
    """The actual s6-re-run scenario: same path, different content."""
    _seg(tmp_path, "a.wav", b"audio")
    tr = {"placed": "a.wav", "qc_score": 0.7, "qc_wer": 0.1}
    M.stamp_qc(tmp_path, tr, "score")
    M.stamp_qc(tmp_path, tr, "wer")
    (tmp_path / "a.wav").write_bytes(b"audio-after-s6-rerun")
    assert M.stale_qc(tmp_path, _man(tr), "en") == {"score": ["u0001"],
                                                    "wer": ["u0001"]}


def test_stamps_are_independent(tmp_path):
    """evaluate and backcheck run separately; re-scoring one must not vouch
    for the other (the sketch60/en case after `dubadabidu evaluate`)."""
    _seg(tmp_path, "a.wav", b"audio")
    tr = {"placed": "a.wav", "qc_score": 0.7, "qc_wer": 0.1}
    M.stamp_qc(tmp_path, tr, "score")
    M.stamp_qc(tmp_path, tr, "wer")
    (tmp_path / "a.wav").write_bytes(b"new")
    M.stamp_qc(tmp_path, tr, "score")        # only evaluate re-ran
    assert M.stale_qc(tmp_path, _man(tr), "en") == {"score": [], "wer": ["u0001"]}


def test_legacy_scores_without_stamp_are_stale(tmp_path):
    """Pre-stamp manifests are unverifiable — reporting them as current is the
    bug. sketch60/test60 both landed here."""
    _seg(tmp_path, "a.wav", b"audio")
    man = _man({"placed": "a.wav", "qc_score": 0.7, "qc_wer": 0.1})
    assert M.stale_qc(tmp_path, man, "en") == {"score": ["u0001"],
                                               "wer": ["u0001"]}


def test_missing_audio_is_stale_not_a_crash(tmp_path):
    man = _man({"placed": "gone.wav", "qc_score": 0.7})
    assert M.stale_qc(tmp_path, man, "en")["score"] == ["u0001"]


def test_stamp_on_missing_audio_is_a_noop(tmp_path):
    tr = {"placed": "gone.wav", "qc_score": 0.7}
    M.stamp_qc(tmp_path, tr, "score")
    assert "qc_of" not in tr          # never stamp what wasn't there to grade


def test_stamps_are_cleared_by_clear_synth_key_prefix():
    """clear_synth and autopilot._reroll drop keys by the `qc_` prefix; the
    stamps must ride along or a re-roll would leave a stamp vouching for a
    deleted take."""
    for kind in ("score", "wer"):
        assert M._QC_STAMPS[kind][0].startswith("qc_")


def test_langs_are_independent(tmp_path):
    _seg(tmp_path, "a.wav", b"audio")
    _seg(tmp_path, "b.wav", b"audio2")
    en = {"placed": "a.wav", "qc_score": 0.7}
    ru = {"placed": "b.wav", "qc_score": 0.7}
    M.stamp_qc(tmp_path, en, "score")
    man = {"utterances": [{"id": "u0001", "tr": {"en": en, "ru": ru}}]}
    assert M.stale_qc(tmp_path, man, "en")["score"] == []
    assert M.stale_qc(tmp_path, man, "ru")["score"] == ["u0001"]


def test_mix_encode_pins_the_output_sample_rate():
    """loudnorm resamples to 192 kHz internally. With no explicit output rate
    ffmpeg keeps the filter's rate and AAC clamps it to 96 kHz, so every dubbed
    track shipped at double the intended 44.1 kHz — more data, no audible gain,
    on files destined for YouTube. The premix is already aresample=44100; this
    keeps the encoder there too.

    Found by an edge plumbing run, not by a test — hence this test."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "pipeline" / "s6_mix.py").read_text()
    enc = src[src.index("def _encode("):]
    enc = enc[:enc.index("check=True)")]
    assert '"-ar", "44100"' in enc, "s6 must pin the encoder sample rate"


# --- audio_sig memoization ----------------------------------------------------

def test_audio_sig_is_memoized_on_size_and_mtime(tmp_path, monkeypatch):
    """batch_report calls stale_qc per video x language and its docstring
    promised it "costs nothing". It does not: on a 20-video course that is
    ~40k sha1 passes, and the autopilot runs it after every batch.

    The CONTENT hash is still what is stored and compared — the cache only
    skips re-reading a file whose size and mtime are both unchanged."""
    from pipeline import manifest as M
    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 4096)
    first = M.audio_sig(p)
    assert (tmp_path / ".audio_sig.json").exists()

    reads = {"n": 0}
    real_open = Path.open

    def counting_open(self, *a, **kw):
        if self.suffix == ".wav":
            reads["n"] += 1
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", counting_open)
    assert M.audio_sig(p) == first
    assert reads["n"] == 0, "cache hit still read the wav"


def test_changed_content_misses_the_cache(tmp_path):
    """Every writer here (s4/s5/s6, rsync) moves mtime, so a changed file must
    always re-hash. Getting this wrong is a stale score reading as current."""
    from pipeline import manifest as M
    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 4096)
    first = M.audio_sig(p)
    p.write_bytes(b"y" * 4096)          # same size, new mtime
    second = M.audio_sig(p)
    assert second != first, "a rewritten take kept its old signature"
    assert M.audio_sig(p) == second


def test_a_corrupt_cache_is_ignored_not_fatal(tmp_path):
    from pipeline import manifest as M
    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 512)
    (tmp_path / ".audio_sig.json").write_text("{not json", encoding="utf-8")
    assert len(M.audio_sig(p)) == 12


def test_the_cache_does_not_grow_without_bound(tmp_path):
    """A segment re-rolled fifty times must not leave fifty entries behind."""
    from pipeline import manifest as M
    import json as _json
    p = tmp_path / "a.wav"
    for i in range(5):
        p.write_bytes(bytes([i]) * (512 + i))
        M.audio_sig(p)
    cache = _json.loads((tmp_path / ".audio_sig.json").read_text(encoding="utf-8"))
    assert len([k for k in cache if k.startswith("a.wav:")]) == 1
