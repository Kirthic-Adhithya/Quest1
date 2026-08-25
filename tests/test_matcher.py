from quest1.audio.transcribe import Transcript, Word
from quest1.search.matcher import DEFAULT_THRESHOLD, Candidate, best_match, find_candidates, normalize


def _words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start=s, end=e, prob=0.99) for t, s, e in specs]


def _transcript(*specs: tuple[str, float, float]) -> Transcript:
    return Transcript(words=_words(*specs), language="en", language_prob=1.0)


def test_normalize_casefolds_and_strips_punctuation():
    assert normalize("My Mind Rebels, at Stagnation!") == "my mind rebels at stagnation"


def test_exact_match_scores_near_100():
    t = _transcript(
        ("My", 10.0, 10.2), ("mind", 10.2, 10.4), ("rebels", 10.4, 10.7),
        ("at", 10.7, 10.8), ("stagnation.", 10.8, 11.2),
    )
    match = best_match(t, "My mind rebels at stagnation")
    assert match is not None
    assert match.score > 95
    assert match.start == 10.0


def test_asr_word_substitution_still_matches_above_threshold():
    """The real failure mode observed in practice: ASR heard 'its' instead of
    'at'. This must still clear the default threshold."""
    t = _transcript(
        ("my", 325.68, 325.72), ("mind", 325.72, 326.18), ("rebels", 326.18, 326.20),
        ("its", 326.20, 326.60), ("stagnation", 326.60, 327.66),
    )
    match = best_match(t, "My mind rebels at stagnation")
    assert match is not None
    assert match.score >= DEFAULT_THRESHOLD


def test_absent_dialogue_returns_no_match():
    t = _transcript(
        ("the", 1.0, 1.1), ("weather", 1.1, 1.5), ("is", 1.5, 1.6), ("nice", 1.6, 2.0)
    )
    assert best_match(t, "My mind rebels at stagnation") is None


def test_empty_transcript_returns_no_match():
    t = Transcript(words=[], language="en", language_prob=1.0)
    assert best_match(t, "My mind rebels at stagnation") is None


def test_earliest_wins_not_highest_score():
    """Policy from DESIGN.md: score gates whether a candidate counts as real;
    it never ranks among real candidates. A later, cleaner-sounding match must
    not beat an earlier, noisier-but-still-genuine one."""
    t = _transcript(
        # earlier, slightly noisier occurrence (still clears threshold)
        ("my", 10.0, 10.1), ("mind", 10.1, 10.2), ("rebels", 10.2, 10.3),
        ("it", 10.3, 10.4), ("stagnation", 10.4, 10.6),
        # unrelated filler
        ("later", 50.0, 50.5), ("on", 50.5, 50.6),
        # later, cleaner occurrence
        ("my", 100.0, 100.1), ("mind", 100.1, 100.2), ("rebels", 100.2, 100.3),
        ("at", 100.3, 100.4), ("stagnation", 100.4, 100.6),
    )
    match = best_match(t, "My mind rebels at stagnation")
    assert match is not None
    assert match.start == 10.0, "earlier genuine match must win even though the later one scores higher"


def test_find_candidates_lists_all_survivors_earliest_first():
    t = _transcript(
        ("my", 100.0, 100.1), ("mind", 100.1, 100.2), ("rebels", 100.2, 100.3),
        ("at", 100.3, 100.4), ("stagnation", 100.4, 100.6),
        ("filler", 150.0, 150.5),
        ("my", 10.0, 10.1), ("mind", 10.1, 10.2), ("rebels", 10.2, 10.3),
        ("at", 10.3, 10.4), ("stagnation", 10.4, 10.6),
    )
    candidates = find_candidates(t, "My mind rebels at stagnation")
    assert len(candidates) == 2
    assert candidates[0].start < candidates[1].start


def test_overlapping_window_sizes_collapse_to_one_candidate():
    """Multiple window sizes scanning the same real occurrence must not
    produce several near-duplicate entries for it."""
    t = _transcript(
        ("preamble", 300.0, 300.5),
        ("my", 325.68, 325.72), ("mind", 325.72, 326.18), ("rebels", 326.18, 326.20),
        ("its", 326.20, 326.60), ("stagnation", 326.60, 327.66),
        ("give", 327.66, 328.0),
    )
    candidates = find_candidates(t, "My mind rebels at stagnation")
    assert len(candidates) == 1


def test_short_phrase_coincidental_match_rejected_at_default_threshold():
    """Real false positive observed in practice: "I am the king" wrongly
    matched a coincidental "I am in" at score 70.0 -- exactly the old
    threshold. This must now be rejected by default."""
    t = _transcript(
        ("I", 5.0, 5.1), ("am", 5.1, 5.2), ("in", 5.2, 5.4),
    )
    assert best_match(t, "I am the king") is None


def test_short_phrase_coincidental_match_rejected_at_default_threshold_2():
    """Real false positive observed in practice: "What is it" wrongly matched
    a coincidental "What's" at score 75.0. Must now be rejected by default."""
    t = _transcript(("What's", 5.0, 5.3),)
    assert best_match(t, "What is it") is None
