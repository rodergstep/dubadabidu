"""Consensus selection: the rule, not the audio."""
from qc.stress_consensus import consensus, rank_takes, pick, MIN_VOTES


def test_majority_decides_and_the_odd_take_out_loses():
    """Three takes agree that слот 1 is stressed, one dissents — the dissenter
    is the one carrying the error, which is the whole premise (errors are
    stochastic per take, so the outlier is wrong, not the crowd)."""
    per = {"t0": {"краски": 1}, "t1": {"краски": 1},
           "t2": {"краски": 1}, "t3": {"краски": 0}}
    assert consensus(per) == {"краски": 1}
    assert pick(per) in {"t0", "t1", "t2"}
    assert dict((t, d) for t, d, _ in rank_takes(per))["t3"] == 1


def test_two_takes_cannot_form_a_consensus():
    """Two disagreeing takes are not a majority, they are a coin flip. Calling
    one would invent a verdict — and two takes is exactly what production rolls
    today, which is why this needs best_of >= 3."""
    assert consensus({"t0": {"w": 0}, "t1": {"w": 1}}) == {}


def test_an_even_split_is_left_undecided():
    per = {"t0": {"w": 0}, "t1": {"w": 0}, "t2": {"w": 1}, "t3": {"w": 1}}
    assert "w" not in consensus(per)


def test_a_word_below_the_vote_floor_is_ignored():
    per = {f"t{i}": ({"w": 0} if i < MIN_VOTES - 1 else {}) for i in range(5)}
    assert consensus(per) == {}


def test_repeated_words_are_dropped_not_paired_by_name(tmp_path):
    """Two occurrences of one word are different audio; pairing them by name
    would compare unrelated things."""
    from qc.stress_consensus import slots_for
    tg = tmp_path / "x.TextGrid"
    tg.write_text(
        'name = "words"\n'
        'xmin = 0\nxmax = 1\ntext = "краски"\n'
        'xmin = 1\nxmax = 2\ntext = "краски"\n'
        'name = "phones"\n'
        'xmin = 0\nxmax = 1\ntext = "a"\n'
        'xmin = 1\nxmax = 2\ntext = "a"\n', encoding="utf-8")
    assert "краски" not in slots_for(tg)


def test_ranking_prefers_more_evidence_when_deviations_tie():
    """Zero deviations out of one word is weaker than zero out of three."""
    per = {"t0": {"a": 0}, "t1": {"a": 0, "b": 1, "c": 1},
           "t2": {"a": 0, "b": 1, "c": 1}, "t3": {"a": 0, "b": 1, "c": 1}}
    assert pick(per) != "t0"
