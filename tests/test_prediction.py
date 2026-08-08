"""Word-prediction tests — offline, instant, and learning from the user."""
import json
import time

import pytest

from interaction.prediction import (
    BUNDLED_RANKS,
    WordBuffer,
    WordPredictor,
)


@pytest.fixture
def predictor():
    return WordPredictor(path=None, autosave=False)


# ------------------------------------------------------------------ suggesting
def test_a_prefix_offers_completions(predictor):
    assert predictor.suggest("he") == ["her", "here", "hello", "hey"]


def test_suggestions_all_start_with_the_prefix(predictor):
    for prefix in ("th", "wa", "com", "pl", "s"):
        for word in predictor.suggest(prefix):
            assert word.startswith(prefix), f"{word} does not extend {prefix}"


def test_the_prefix_itself_is_not_offered_back(predictor):
    """Space already completes it — the slot is worth more as a completion."""
    assert "the" in BUNDLED_RANKS
    assert "the" not in predictor.suggest("the")
    assert all(len(word) > 3 for word in predictor.suggest("the"))


def test_more_common_words_rank_first(predictor):
    suggestions = predictor.suggest("t", limit=4)
    assert suggestions[0] == "the", suggestions
    assert predictor.score("the") < predictor.score("to") < \
        predictor.score("together")


def test_the_limit_is_respected(predictor):
    assert len(predictor.suggest("a", limit=4)) <= 4
    assert len(predictor.suggest("a", limit=2)) == 2


def test_an_empty_prefix_offers_nothing(predictor):
    assert predictor.suggest("") == []
    assert predictor.suggest("   ") == []


def test_an_unknown_prefix_offers_nothing(predictor):
    assert predictor.suggest("zzzq") == []


def test_suggestions_are_case_insensitive(predictor):
    assert predictor.suggest("HE") == predictor.suggest("he")


def test_lookup_is_instant(predictor):
    """NFR: the bar must not lag the keyboard."""
    start = time.perf_counter()
    for _ in range(500):
        predictor.suggest("he")
    elapsed = (time.perf_counter() - start) / 500
    assert elapsed < 0.002, f"{elapsed * 1000:.2f} ms per lookup"


def test_completion_is_the_remaining_letters(predictor):
    assert predictor.completion_for("he", "hello") == "llo"
    assert predictor.completion_for("", "hello") == "hello"
    assert predictor.completion_for("HE", "hello") == "llo"


# ------------------------------------------------------------------- learning
def test_using_a_word_promotes_it(predictor):
    before = predictor.suggest("he")
    assert before[0] != "hello"
    for _ in range(3):
        predictor.record("hello")
    assert predictor.suggest("he")[0] == "hello"
    assert predictor.personal_count("hello") == 3


def test_a_word_the_user_invents_becomes_suggestible(predictor):
    assert predictor.suggest("nitz") == []
    predictor.record("nitzan")
    assert predictor.suggest("nitz") == ["nitzan"]


def test_repeated_personal_use_beats_common_english(predictor):
    for _ in range(6):
        predictor.record("thermos")
    assert predictor.suggest("th")[0] == "thermos"


def test_junk_is_not_learned(predictor):
    for junk in ("", "   ", "12345", "a b", "!!"):
        predictor.record(junk)
    assert predictor.personal_count("12345") == 0
    assert predictor.suggest("123") == []


# ---------------------------------------------------------------- persistence
def test_personal_words_survive_a_restart(tmp_path):
    path = str(tmp_path / "user_words.json")
    first = WordPredictor(path=path)
    for _ in range(4):
        first.record("hello")
    first.save()

    second = WordPredictor(path=path)
    assert second.personal_count("hello") == 4
    assert second.suggest("he")[0] == "hello"


def test_only_words_are_stored_never_sentences(tmp_path):
    path = str(tmp_path / "user_words.json")
    predictor = WordPredictor(path=path)
    predictor.record("hello")
    predictor.record("world")
    stored = json.loads(open(path, encoding="utf-8").read())
    assert set(stored) == {"hello", "world"}
    assert all(isinstance(count, int) for count in stored.values())


def test_a_corrupt_store_is_ignored(tmp_path):
    path = tmp_path / "user_words.json"
    path.write_text("{not json", encoding="utf-8")
    predictor = WordPredictor(path=str(path))
    assert predictor.suggest("he")                 # still works


def test_a_missing_store_is_not_an_error(tmp_path):
    predictor = WordPredictor(path=str(tmp_path / "nope.json"))
    assert predictor.personal_count("hello") == 0


def test_saving_without_a_path_is_a_no_op(predictor):
    predictor.record("hello")
    predictor.save()                               # must not raise


# ------------------------------------------------------------------ the buffer
def test_the_word_buffer_tracks_the_current_word():
    buffer = WordBuffer()
    for char in "hel":
        buffer.add(char)
    assert buffer.text == "hel"
    buffer.backspace()
    assert buffer.text == "he"
    buffer.clear()
    assert not buffer


def test_punctuation_ends_the_word():
    buffer = WordBuffer()
    for char in "hi":
        buffer.add(char)
    buffer.add(".")
    assert buffer.text == ""


def test_backspacing_an_empty_buffer_is_safe():
    buffer = WordBuffer()
    buffer.backspace()
    assert buffer.text == ""
