"""Keystroke injection tests — a fake pynput, so nothing is really typed."""
import pytest

from interaction.injector import (
    InjectionUnavailable,
    KeystrokeInjector,
    make_injector,
)


class FakeKey:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"

    def __eq__(self, other):
        return isinstance(other, FakeKey) and other.name == self.name

    __hash__ = None


class FakeKeys:
    enter = FakeKey("enter")
    backspace = FakeKey("backspace")
    space = FakeKey("space")
    tab = FakeKey("tab")
    esc = FakeKey("esc")


class FakeController:
    """Records what pynput would have been asked to do."""

    def __init__(self):
        self.calls = []

    def type(self, text):
        self.calls.append(("type", text))

    def press(self, key):
        self.calls.append(("press", key.name))

    def release(self, key):
        self.calls.append(("release", key.name))


@pytest.fixture
def injector():
    return KeystrokeInjector(controller=FakeController(), keys=FakeKeys)


def calls(injector):
    return injector._keyboard.calls


# ------------------------------------------------------------------ characters
def test_characters_are_typed_as_text_not_key_positions(injector):
    """Spec Section 8: typing text keeps us off the OS keyboard layout."""
    injector.send("char", "a")
    assert calls(injector) == [("type", "a")]


def test_letters_are_lower_case_without_shift(injector):
    injector.send("char", "A")
    assert calls(injector) == [("type", "a")]


def test_non_ascii_characters_go_through_unchanged(injector):
    injector.send("char", "ש")
    assert calls(injector) == [("type", "ש")]


def test_punctuation_is_untouched(injector):
    for mark in (".", ",", "?", "!", "'", "-"):
        injector.send("char", mark)
    assert [text for _, text in calls(injector)] == [".", ",", "?", "!", "'", "-"]


def test_empty_payload_types_nothing(injector):
    injector.type_text("")
    assert calls(injector) == []


# ----------------------------------------------------------------- special keys
@pytest.mark.parametrize("action", ["enter", "backspace", "space"])
def test_control_keys_use_real_key_codes(injector, action):
    injector.send(action)
    assert calls(injector) == [("press", action), ("release", action)]


def test_an_unknown_special_key_is_rejected(injector):
    with pytest.raises(ValueError):
        injector.press_special("banana")


# ------------------------------------------------------------------ shift latch
def test_shift_latches_for_exactly_one_character(injector):
    assert injector.latch_shift() is True
    assert injector.shift_latched

    injector.send("char", "h")
    injector.send("char", "i")
    assert [text for _, text in calls(injector)] == ["H", "i"]
    assert not injector.shift_latched, "shift must be one-shot"


def test_shift_can_be_cancelled_by_pressing_it_again(injector):
    injector.send("shift")
    injector.send("shift")
    assert not injector.shift_latched
    injector.send("char", "h")
    assert calls(injector) == [("type", "h")]


def test_shift_itself_injects_nothing(injector):
    assert injector.send("shift") is False
    assert calls(injector) == []


def test_close_releases_a_latched_shift(injector):
    injector.latch_shift()
    injector.close()
    assert not injector.shift_latched


# ------------------------------------------------------------- app-level actions
@pytest.mark.parametrize("action", ["page", "lang", "pause", "recalibrate"])
def test_app_level_actions_are_not_injected(injector, action):
    assert injector.send(action) is False
    assert calls(injector) == [], f"{action} must be handled by the app"


def test_send_reports_whether_it_injected(injector):
    assert injector.send("char", "x") is True
    assert injector.send("enter") is True
    assert injector.send("shift") is False


def test_history_records_what_was_sent(injector):
    injector.send("char", "h")
    injector.send("space")
    assert injector.typed == ["h", "<space>"]


# -------------------------------------------------------------- typing a word
def test_typing_hello_produces_the_expected_keystrokes(injector):
    """The M4 checkpoint, minus the eyes."""
    for char in "hello":
        injector.send("char", char)
    injector.send("space")
    assert calls(injector) == [
        ("type", "h"), ("type", "e"), ("type", "l"), ("type", "l"),
        ("type", "o"), ("press", "space"), ("release", "space"),
    ]


def test_shifted_word_capitalises_only_the_first_letter(injector):
    injector.send("shift")
    for char in "hello":
        injector.send("char", char)
    assert "".join(text for kind, text in calls(injector) if kind == "type") == "Hello"


# ------------------------------------------------------------------ degradation
def test_make_injector_reports_failure_instead_of_raising(monkeypatch):
    def explode(*args, **kwargs):
        raise InjectionUnavailable("no permission")

    monkeypatch.setattr("interaction.injector.KeystrokeInjector", explode)
    messages = []
    assert make_injector(log=messages.append) is None
    assert messages and "injection disabled" in messages[0]


def test_a_real_injector_is_returned_when_pynput_works(monkeypatch):
    monkeypatch.setattr(
        "interaction.injector.KeystrokeInjector",
        lambda log=print: "injector",
    )
    assert make_injector() == "injector"
