"""Word-level stress marking: tokenizer, page rendering, ingest, lexicon.

The label this produces is the input every surviving remediation route needs
(qc/stress_words.py header). Four automated detectors failed their gate; the
human ear is the only labeller that ever worked, and the review page has been
discarding WHICH WORD every time it was used.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc import stress_words as SW  # noqa: E402
from qc import review_page as RP  # noqa: E402


# ------------------------------------------------------------- tokenizer ----

def test_words_are_indexed_in_order():
    assert SW.tokenize("Это краска хром-кобальт.") == \
        ["Это", "краска", "хром-кобальт"]


def test_a_hyphenated_compound_is_one_token():
    """`хром-кобальт` / `сине-зелёная` / `clair-obscur` are single lexical
    items and the reviewer clicks them once. Splitting them would also make the
    index unstable against the same text rendered elsewhere."""
    for text, n in [("сине-зелёная", 1), ("clair-obscur", 1),
                    ("Тёмно-красная краска", 2)]:
        assert len(SW.tokenize(text)) == n, text


def test_digits_are_not_markable():
    """Digits are expanded to words at the synthesis boundary, so what the
    voice said is not what the manifest shows. Marking `50` would attach the
    label to a token the listener never heard."""
    assert SW.tokenize("нанесите 50 процентов") == ["нанесите", "процентов"]


def test_spans_reconstruct_the_text_exactly():
    """The reviewer has to recognise the sentence they just heard, so spacing
    and punctuation must survive rendering untouched."""
    for text in ["Это краска хром-кобальт, сине-зелёная. Интересная!",
                 "  leading and trailing  ", "no-punctuation", ""]:
        assert "".join(f for f, _ in SW.spans(text)) == text


def test_span_indices_agree_with_tokenize():
    text = "Мелкие формы — это деревья, домики и ступени."
    toks = SW.tokenize(text)
    for frag, i in SW.spans(text):
        if i >= 0:
            assert toks[i] == frag


def test_lexicon_key_folds_case_and_strips_stress_marks():
    assert SW.lexicon_key("Молоко") == SW.lexicon_key("молоко")
    assert SW.lexicon_key("молоко́") == SW.lexicon_key("молоко")
    assert SW.lexicon_key("  Охра  ") == "охра"


def test_lexicon_key_does_not_lemmatise():
    """A TTS can be right about one inflected form and wrong about another, and
    the surface form is what a respelling or avoidance rule matches on."""
    assert SW.lexicon_key("молоко") != SW.lexicon_key("молока")


# ---------------------------------------------------------------- verify ----

def test_verify_accepts_marks_that_still_match():
    text = "Это краска хром-кобальт"
    ok, stale = SW.verify(text, [{"i": 2, "w": "хром-кобальт"}])
    assert ok == [{"i": 2, "w": "хром-кобальт"}] and stale == []


def test_verify_rejects_a_mark_whose_text_was_edited():
    """The index would now point at a different word, and recording THAT would
    poison the table with a word the listener never heard."""
    ok, stale = SW.verify("Это другая краска", [{"i": 2, "w": "хром-кобальт"}])
    assert ok == [] and len(stale) == 1


@pytest.mark.parametrize("mark", [
    {"i": 99, "w": "краска"}, {"i": -1, "w": "краска"},
    {"w": "краска"}, {"i": "2", "w": "краска"}, {},
])
def test_verify_rejects_malformed_marks(mark):
    ok, _ = SW.verify("Это краска", [mark])
    assert ok == []


# ------------------------------------------------------------ page render ---

def test_russian_words_are_clickable():
    out = RP._dub_html("Это краска", "u0001", mark_words=True)
    assert 'class="words" data-id="u0001"' in out
    assert 'data-i="0" data-w="Это"' in out
    assert 'data-i="1" data-w="краска"' in out


def test_other_languages_get_plain_text():
    """en/fr/de/es have no equivalent defect (FINDINGS 2.1j). Offering a second
    rating axis where there is nothing to mark invites noise on the axis that
    already feeds refit."""
    out = RP._dub_html("This is paint", "u0001", mark_words=False)
    assert "<span" not in out and out == "This is paint"


def test_the_rating_guidance_reads_as_a_task_not_a_rationale():
    """The first version of this text was written for someone who had read
    FINDINGS — "the per-word table", "FINDINGS 2.1", "the only labeller" — and
    the reviewer said he could not tell what to do with it.

    That is not a cosmetic failure. Instructions a rater cannot follow produce
    noisy labels just as surely as no instructions do, and FINDINGS 2.1d exists
    because he once rated STRESS while the page asked for overall quality. The
    guidance has to name each control and say the three are independent.
    """
    src = (ROOT / "qc" / "review_page.py").read_text(encoding="utf-8")
    axis = src[src.index("class='axis'"):src.index("if mark_words else")]
    for control in ("rate 1", "accept", "reject", "click a word"):
        assert control in axis, f"guidance never names the {control!r} control"
    assert "separate" in axis and "do not affect each other" in axis, (
        "the guidance must say the three axes are independent — a take can be "
        "5 stars AND have a wrong word")
    for jargon in ("FINDINGS", "labeller", "per-word table", "2.1"):
        assert jargon not in axis, (
            f"{jargon!r} is internal vocabulary — the reviewer is mid-task, "
            f"not reading the research log")


def test_stress_langs_are_the_lexical_stress_ones():
    assert "ru" in SW.STRESS_LANGS
    assert "uk" in SW.STRESS_LANGS, "uk is the next target and stress is lexical"
    for lang in ("en", "fr", "de", "es"):
        assert lang not in SW.STRESS_LANGS


def test_markup_is_escaped():
    """Manifest text is hand-editable, so it reaches the page as untrusted
    content. Note the tokenizer takes the `b` of `<b>` as a word, so the
    bracket and the letter are escaped separately — assert on the brackets and
    the attributes, not on a contiguous `&lt;b&gt;`."""
    out = RP._dub_html('paint <b>"x"</b> & co', "u<1>", mark_words=True)
    assert "<b>" not in out and "</b>" not in out
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out
    assert 'data-id="u&lt;1&gt;"' in out       # attribute, quote-escaped
    assert "&quot;" in out                     # quotes in text content


# ---------------------------------------------------------------- ingest ----

@pytest.fixture
def ingested(tmp_path, monkeypatch):
    """Run verdicts.run over a two-segment ru manifest with word marks."""
    from pipeline import manifest as M
    from qc import verdicts as V
    monkeypatch.chdir(tmp_path)
    cfg = {"work_dir": "work"}
    video = "lesson.mp4"
    man = {"video": video, "duration": 20.0, "stages": {}, "utterances": [
        {"id": "u0001", "start": 0.0, "end": 5.0, "text_uk": "a",
         "tr": {"ru": {"text": "Это краска хром-кобальт",
                       "fitted_text": "Это краска хром-кобальт",
                       "placed": "seg/ru/u0001_placed.wav", "qc_score": 0.7}}},
        {"id": "u0002", "start": 6.0, "end": 12.0, "text_uk": "b",
         "tr": {"ru": {"text": "Жёлтые цвета охра",
                       "fitted_text": "Жёлтые цвета охра",
                       "placed": "seg/ru/u0002_placed.wav", "qc_score": 0.6}}}]}
    wd = M.video_workdir(cfg, video)
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    # no scored audio on disk => stale_qc reports nothing scored, not stale
    for u in man["utterances"]:
        u["tr"]["ru"].pop("qc_score")
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    export = {"key": f"lesson_ru_{V._seg_hash(man['utterances'], 'ru')}",
              "ratings": {"u0001": 4},
              "verdicts": {"u0002": "reject"},
              "words": {"u0001": [{"i": 2, "w": "хром-кобальт"}],
                        "u0002": [{"i": 2, "w": "охра"}]}}
    p = tmp_path / "export.json"
    p.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    V.run(cfg, video, str(p))
    return cfg, video, tmp_path, M


def test_marks_land_in_the_manifest(ingested):
    cfg, video, _, M = ingested
    man = M.load(cfg, video)
    assert man["utterances"][0]["tr"]["ru"]["stress_words"] == ["хром-кобальт"]
    assert man["utterances"][1]["tr"]["ru"]["stress_words"] == ["охра"]


def test_lexicon_is_written_with_counts_and_context(ingested):
    _, _, tmp, _ = ingested
    lex = json.loads((tmp / "stress_lexicon_ru.json").read_text(encoding="utf-8"))
    assert set(lex) == {"хром-кобальт", "охра"}
    e = lex["охра"]
    assert e["marked"] == 1 and e["forms"] == ["охра"]
    assert e["occurrences"][0]["context"] == "Жёлтые цвета охра", (
        "context is what makes an ambiguous word judgeable — FINDINGS 2.1h")
    assert e["occurrences"][0]["id"] == "u0002"


def test_re_ingesting_the_same_page_does_not_inflate_the_count(ingested):
    """A reviewer who exports twice must not double the evidence for a word."""
    cfg, video, tmp, M = ingested
    from qc import verdicts as V
    V.run(cfg, video, str(tmp / "export.json"))
    lex = json.loads((tmp / "stress_lexicon_ru.json").read_text(encoding="utf-8"))
    assert lex["охра"]["marked"] == 1


def test_a_second_video_accumulates_into_the_same_entry(ingested):
    """The table is the point: a word marked across several videos is the
    systematic population (FINDINGS 2.1j) and should sort to the top."""
    cfg, _, tmp, M = ingested
    from qc import verdicts as V
    man = {"video": "lesson2.mp4", "duration": 9.0, "stages": {}, "utterances": [
        {"id": "u0001", "start": 0.0, "end": 4.0, "text_uk": "c",
         "tr": {"ru": {"text": "Здесь охра", "fitted_text": "Здесь охра"}}}]}
    wd = M.video_workdir(cfg, "lesson2.mp4")
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    export = {"key": f"lesson2_ru_{V._seg_hash(man['utterances'], 'ru')}",
              "ratings": {}, "verdicts": {},
              "words": {"u0001": [{"i": 1, "w": "охра"}]}}
    p = tmp / "export2.json"
    p.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    V.run(cfg, "lesson2.mp4", str(p))
    lex = json.loads((tmp / "stress_lexicon_ru.json").read_text(encoding="utf-8"))
    assert lex["охра"]["marked"] == 2
    assert {o["video"] for o in lex["охра"]["occurrences"]} == \
        {"lesson", "lesson2"}
    assert list(lex)[0] == "охра", "most-marked word must sort first"


def test_an_export_without_word_marks_still_ingests(tmp_path, monkeypatch):
    """Every export taken before this feature existed has no `words` key."""
    from pipeline import manifest as M
    from qc import verdicts as V
    monkeypatch.chdir(tmp_path)
    cfg = {"work_dir": "work"}
    man = {"video": "old.mp4", "duration": 5.0, "stages": {}, "utterances": [
        {"id": "u0001", "start": 0.0, "end": 4.0, "text_uk": "a",
         "tr": {"ru": {"text": "Это краска", "fitted_text": "Это краска"}}}]}
    wd = M.video_workdir(cfg, "old.mp4")
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    export = {"key": f"old_ru_{V._seg_hash(man['utterances'], 'ru')}",
              "ratings": {"u0001": 5}, "verdicts": {}}
    p = tmp_path / "e.json"
    p.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    V.run(cfg, "old.mp4", str(p))
    assert M.load(cfg, "old.mp4")["utterances"][0]["tr"]["ru"][
        "human_rating"] == 5
    assert not (tmp_path / "stress_lexicon_ru.json").exists(), (
        "no marks means no lexicon file — an empty table is not evidence")


def test_a_stale_mark_is_dropped_not_recorded(tmp_path, monkeypatch):
    """Text hand-edited between review and ingest: the index now points
    elsewhere, so the mark must be dropped rather than silently reattached."""
    from pipeline import manifest as M
    from qc import verdicts as V
    monkeypatch.chdir(tmp_path)
    cfg = {"work_dir": "work"}
    man = {"video": "e.mp4", "duration": 5.0, "stages": {}, "utterances": [
        {"id": "u0001", "start": 0.0, "end": 4.0, "text_uk": "a",
         "tr": {"ru": {"text": "Совсем другой текст",
                       "fitted_text": "Совсем другой текст"}}}]}
    wd = M.video_workdir(cfg, "e.mp4")
    (wd / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    export = {"key": f"e_ru_{V._seg_hash(man['utterances'], 'ru')}",
              "ratings": {}, "verdicts": {},
              "words": {"u0001": [{"i": 2, "w": "хром-кобальт"}]}}
    p = tmp_path / "e.json"
    p.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    V.run(cfg, "e.mp4", str(p))
    tr = M.load(cfg, "e.mp4")["utterances"][0]["tr"]["ru"]
    assert "stress_words" not in tr
    lex_p = tmp_path / "stress_lexicon_ru.json"
    assert not lex_p.exists() or "хром-кобальт" not in json.loads(
        lex_p.read_text(encoding="utf-8"))
