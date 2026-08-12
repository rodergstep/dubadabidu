"""The review-page export key must identify the SEGMENTATION, nothing else.

`review_page` writes `<video>_<lang>_<seghash>` into the page; `verdicts`
recomputes it and refuses the ingest on a mismatch with:

    "the video was re-segmented since these ratings were taken"

Both sides used to hash the worst-first DISPLAY order, so the key moved
whenever any qc_score moved — re-running `evaluate`, re-rolling one segment,
editing qc.eval.weights. That turns a true statement ("these ratings are for
different boundaries") into a false one, and it fires on the exact workflow the
project tells the reviewer to follow: the stale-metrics banner says "run
`dubadabidu qc`, then regenerate this page", and running qc is what breaks the
key.

It matters more here than the bug size suggests: qc/blind.py records that refit
has had ZERO usable ratings since the project began, so rejecting good exports
on a wrong diagnosis is expensive.

The dangerous case — ratings taken against audio that has since been rewritten —
is caught by M.stale_qc on a content hash of the graded wav, and is unaffected.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc.verdicts import _seg_hash  # noqa: E402


def _man(scores):
    """Utterances with the given per-id qc_score."""
    return [{"id": uid, "start": start, "tr": {"en": {"qc_score": sc}}}
            for uid, start, sc in scores]


BASE = [("u0001", 0.0, 0.90), ("u0002", 4.0, 0.40), ("u0003", 9.5, 0.70)]


def test_rescoring_does_not_change_the_key():
    """The headline case: `dubadabidu qc` re-scores, the worst-first order
    changes completely, the boundaries do not."""
    before = _seg_hash(_man(BASE), "en")
    rescored = _man([("u0001", 0.0, 0.20),      # was best, now worst
                     ("u0002", 4.0, 0.95),      # was worst, now best
                     ("u0003", 9.5, 0.55)])
    assert _seg_hash(rescored, "en") == before, (
        "re-scoring invalidated a ratings export — the reviewer is told the "
        "video was re-segmented, which is not what happened")


def test_changing_the_eval_weights_does_not_change_the_key():
    """A weights edit re-scores every segment at once. Same argument."""
    before = _seg_hash(_man(BASE), "en")
    reweighted = _man([(u, s, sc * 0.5 + 0.1) for u, s, sc in BASE])
    assert _seg_hash(reweighted, "en") == before


def test_unscored_segments_hash_the_same_as_scored_ones():
    """Before `evaluate` runs there is no qc_score at all. The page is still
    generatable, and its key must match the one the ingest computes later."""
    unscored = [{"id": u, "start": s, "tr": {"en": {}}} for u, s, _ in BASE]
    assert _seg_hash(unscored, "en") == _seg_hash(_man(BASE), "en")


def test_manifest_order_does_not_change_the_key():
    """Nothing guarantees manifest order survives a hand-edit; the key must be
    a property of the boundaries, not of how they were listed."""
    assert _seg_hash(_man(BASE[::-1]), "en") == _seg_hash(_man(BASE), "en")


# --- what it MUST still catch -------------------------------------------------

def test_a_real_resegmentation_still_changes_the_key():
    moved = _man([("u0001", 0.0, 0.90), ("u0002", 4.6, 0.40),   # boundary moved
                  ("u0003", 9.5, 0.70)])
    assert _seg_hash(moved, "en") != _seg_hash(_man(BASE), "en")


def test_a_dropped_segment_still_changes_the_key():
    assert _seg_hash(_man(BASE[:2]), "en") != _seg_hash(_man(BASE), "en")


def test_an_added_segment_still_changes_the_key():
    added = _man(BASE + [("u0004", 14.0, 0.8)])
    assert _seg_hash(added, "en") != _seg_hash(_man(BASE), "en")


def test_review_page_and_verdicts_compute_the_same_hash():
    """The two live in different files and have drifted before. Pin them to the
    same value by running the page's own expression against verdicts'."""
    import hashlib
    us = _man(BASE)
    page = hashlib.sha1(
        ",".join(u["id"] + str(u["start"])
                 for u in sorted(us, key=lambda u: u["id"])).encode()
    ).hexdigest()[:6]
    assert page == _seg_hash(us, "en"), (
        "review_page and verdicts disagree about the export key — every "
        "ingest will be refused")


def test_review_page_source_sorts_by_id_not_by_score():
    """The page builds its key inline, so a future edit could re-introduce the
    display-order hash without any test noticing. Read the source."""
    src = (ROOT / "qc" / "review_page.py").read_text(encoding="utf-8")
    key_block = src[src.index("seg_hash = hashlib.sha1"):]
    key_block = key_block[:key_block.index("hexdigest")]
    assert 'sorted(us, key=lambda u: u["id"])' in key_block, (
        "review_page is hashing the worst-first display order again")
