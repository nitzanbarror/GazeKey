"""The redesigned keyboard: layout, suggestion bar, typed preview, self-view."""
import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from interaction.controller import DwellController, DwellSettings
from interaction.injector import KeystrokeInjector
from interaction.layouts import (
    DEFAULT_HEIGHT_RATIO,
    SUGGESTION_SLOTS,
    build_keyboard,
)
from interaction.prediction import WordPredictor
from ui.overlay import KeyboardOverlay
from tests.test_injector import FakeController, FakeKeys
from tests.test_overlay import ScriptedPipeline

SCREEN = (1366, 768)
ERROR_PX = 95.1
SIZE = (683, 384)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def keyboard():
    return build_keyboard(SCREEN, ERROR_PX)


@pytest.fixture
def overlay(qapp, keyboard):
    widget = KeyboardOverlay(
        keyboard, DwellController(keyboard, DwellSettings()),
        ScriptedPipeline(),
        KeystrokeInjector(controller=FakeController(), keys=FakeKeys),
        screen_size=SCREEN, predictor=WordPredictor(path=None, autosave=False),
        show_webcam=False,
    )
    widget.resize(*SIZE)
    yield widget
    widget.close()


def typed(overlay) -> str:
    return "".join(text for kind, text in overlay.injector._keyboard.calls
                   if kind == "type")


def press(overlay, key_id, start, seconds=1.2):
    key = overlay.keyboard.find(key_id)
    assert key is not None, key_id
    now = overlay.pipeline.look_at(key.centre, seconds, start=start)
    overlay.tick()
    return now + 0.5                    # clear the refractory


def type_word(overlay, letters, start=0.0):
    for char in letters:
        start = press(overlay, f"key.{char.upper()}", start)
    return start


# ------------------------------------------------------------------- geometry
def test_the_board_is_two_thirds_of_the_screen(keyboard):
    assert keyboard.name == "qwerty-tall"
    x, y, w, h = keyboard.rect
    assert x == 0 and w == SCREEN[0]
    assert h == pytest.approx(SCREEN[1] * DEFAULT_HEIGHT_RATIO, rel=0.01)
    assert y + h == pytest.approx(SCREEN[1]), "docked at the bottom"


def test_the_height_is_configurable():
    tall = build_keyboard(SCREEN, ERROR_PX, height_ratio=0.8)
    assert tall.rect[3] > build_keyboard(SCREEN, ERROR_PX).rect[3]
    assert tall.key_size[1] > build_keyboard(SCREEN, ERROR_PX).key_size[1]


def test_every_letter_is_on_one_page(keyboard):
    for char in "QWERTYUIOPASDFGHJKLZXCVBNM":
        key = keyboard.find(f"key.{char}")
        assert key is not None and key.on_page(0), char


def test_the_control_row_has_everything_including_quit(keyboard):
    for action in ("space", "backspace", "enter", "shift", "symbols",
                   "lang", "pause", "recalibrate", "quit"):
        key = keyboard.find(f"fn.{action}")
        assert key is not None, action
        for page in range(keyboard.pages):
            assert key.on_page(page), f"{action} missing from page {page}"


def test_space_is_the_widest_key(keyboard):
    space = keyboard.find("fn.space")
    assert space.rect[2] > keyboard.key_size[0] * 1.5


def test_the_symbols_page_has_digits_and_punctuation(keyboard):
    assert keyboard.pages == 2
    payloads = {key.payload for key in keyboard.selectable_on(1)
                if key.action == "char"}
    assert set("0123456789") <= payloads
    assert {"@", "?", "!", "-"} <= payloads


def test_the_suggestion_bar_has_four_selectable_slots(keyboard):
    slots = keyboard.suggestion_keys
    assert len(slots) == SUGGESTION_SLOTS
    assert all(slot.selectable for slot in slots), "suggestions are dwell targets"
    assert all(slot.rect[1] == keyboard.rect[1] for slot in slots), "top row"


def test_the_preview_line_exists_and_is_not_selectable(keyboard):
    preview = keyboard.find("preview")
    assert preview is not None and not preview.selectable
    assert keyboard.preview_rect is not None
    assert preview.rect[1] > keyboard.suggestion_keys[0].rect[1], "below the bar"


def test_the_webcam_block_does_not_collide_with_any_key(keyboard):
    wx, wy, ww, wh = keyboard.webcam_rect
    for key in keyboard.keys:
        kx, ky, kw, kh = key.rect
        apart = (kx + kw <= wx + 1e-6 or wx + ww <= kx + 1e-6
                 or ky + kh <= wy + 1e-6 or wy + wh <= ky + 1e-6)
        assert apart, f"{key.id} overlaps the webcam block"


def test_the_narrow_keys_are_reported_honestly(keyboard):
    """10 columns on 1366 px cannot meet NFR-2 at this accuracy — say so."""
    assert not keyboard.meets_nfr2
    assert "BELOW NFR-2" in keyboard.summary()
    assert "narrowest column" in keyboard.note
    assert "--layout paged" in keyboard.note


def test_the_warning_names_both_axes_and_the_one_that_binds(keyboard):
    """Height binds here, not width — a width-only warning misleads.

    137 px wide but 93 px tall: quoting the width alone would tell the user to
    aim for a ~68 px calibration when the height only tolerates ~46 px.
    """
    width, height = keyboard.smallest_key()
    assert width == pytest.approx(SCREEN[0] / 11, rel=0.01), "control columns"
    assert height < width, "the tall layout is height-bound at 2/3 screen"

    note = keyboard.note
    assert f"{width:.0f} px wide" in note and f"{height:.0f} px tall" in note
    assert "width BELOW" in note and "height BELOW" in note
    assert "BOTH axes" in note
    assert "height is what binds" in note

    # The accuracy it tells you to aim for is the min-dimension one (46 px),
    # not the width-only one (62 px) — and it rounds DOWN, so following the
    # advice actually clears the gate.
    usable = keyboard.usable_error_px()
    assert usable == pytest.approx(height / 2.0)
    assert f"recalibrate to about {int(usable)} px" in note
    assert build_keyboard(SCREEN, int(usable)).meets_nfr2, "advice must work"
    assert abs(usable - width / 2.0) > 10.0


# ---------------------------------------------------------------- typed text
def test_the_preview_line_mirrors_what_was_typed(overlay):
    type_word(overlay, "hi")
    assert overlay.typed_text == "hi"
    assert typed(overlay) == "hi"


def test_space_and_backspace_update_the_preview(overlay):
    now = type_word(overlay, "hi")
    now = press(overlay, "fn.space", now)
    assert overlay.typed_text == "hi "
    press(overlay, "fn.backspace", now)
    assert overlay.typed_text == "hi"


def test_enter_clears_the_preview(overlay):
    now = type_word(overlay, "hi")
    press(overlay, "fn.enter", now)
    assert overlay.typed_text == ""


def test_shift_shows_a_capital_in_the_preview(overlay):
    now = press(overlay, "fn.shift", 0.0)
    press(overlay, "key.H", now)
    assert overlay.typed_text == "H"


# --------------------------------------------------------------- suggestions
def test_typing_a_prefix_fills_the_suggestion_bar(overlay):
    type_word(overlay, "he")
    assert overlay.suggestions[:4] == ["her", "here", "hello", "hey"]


def test_choosing_a_suggestion_types_the_rest_and_a_space(overlay):
    now = type_word(overlay, "he")
    slot = overlay.suggestions.index("hello")
    press(overlay, f"sug.{slot}", now)

    assert typed(overlay) == "hello"
    assert ("press", "space") in overlay.injector._keyboard.calls
    assert overlay.typed_text == "hello "


def test_choosing_a_suggestion_teaches_the_predictor(overlay):
    now = type_word(overlay, "he")
    slot = overlay.suggestions.index("hello")
    press(overlay, f"sug.{slot}", now)
    assert overlay.predictor.personal_count("hello") == 1


def test_typing_a_word_and_a_space_teaches_the_predictor(overlay):
    now = type_word(overlay, "hey")
    press(overlay, "fn.space", now)
    assert overlay.predictor.personal_count("hey") == 1


def test_the_bar_empties_after_a_space(overlay):
    now = type_word(overlay, "he")
    assert overlay.suggestions
    press(overlay, "fn.space", now)
    assert overlay.suggestions == []


def test_backspace_reopens_the_earlier_suggestions(overlay):
    now = type_word(overlay, "hel")
    assert "hello" in overlay.suggestions
    press(overlay, "fn.backspace", now)
    assert overlay.suggestions[:4] == ["her", "here", "hello", "hey"]


def test_an_empty_slot_does_nothing(overlay):
    press(overlay, "sug.3", 0.0)
    assert typed(overlay) == ""
    assert overlay.typed_text == ""


def test_without_a_predictor_the_bar_stays_empty(qapp, keyboard):
    widget = KeyboardOverlay(
        keyboard, DwellController(keyboard, DwellSettings()), ScriptedPipeline(),
        KeystrokeInjector(controller=FakeController(), keys=FakeKeys),
        screen_size=SCREEN, predictor=None, show_webcam=False)
    try:
        widget.resize(*SIZE)
        type_word(widget, "he")
        assert widget.suggestions == []
    finally:
        widget.close()


# ------------------------------------------------------------------ rendering
def render(widget) -> np.ndarray:
    pixmap = QPixmap(*SIZE)
    pixmap.fill(Qt.black)
    widget.render(pixmap)
    image = pixmap.toImage().convertToFormat(4)
    return np.frombuffer(image.bits().asstring(image.byteCount()),
                         dtype=np.uint8).reshape(image.height(), -1, 4)


def test_the_board_renders_dark(overlay):
    frame = render(overlay)
    board_top = int(overlay.keyboard.rect[1] * SIZE[1] / SCREEN[1])
    board = frame[board_top + 4:, :, :3].astype(float)
    assert board.mean() < 60, "the board should read as near-black"
    assert len(np.unique(frame.reshape(-1, 4), axis=0)) > 3


def test_the_self_view_is_drawn_when_a_frame_is_available(qapp, keyboard):
    class PipelineWithPreview(ScriptedPipeline):
        def __init__(self):
            super().__init__()
            self.preview_enabled = False

        def preview_frame(self):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            frame[:, :40] = (240, 30, 30)          # unmistakably red
            return frame

    pipeline = PipelineWithPreview()
    widget = KeyboardOverlay(keyboard, DwellController(keyboard), pipeline,
                             None, screen_size=SCREEN, show_webcam=True)
    try:
        widget.resize(*SIZE)
        assert pipeline.preview_enabled, "the overlay must ask for frames"
        frame = render(widget)
        red = (frame[..., 2].astype(int) - frame[..., 0].astype(int)) > 100
        assert red.any(), "the self-view never appeared"

        wx, wy, ww, wh = keyboard.webcam_rect
        rows, cols = np.where(red)
        assert cols.min() >= wx * SIZE[0] / SCREEN[0] - 2
        assert rows.min() >= wy * SIZE[1] / SCREEN[1] - 2
    finally:
        widget.close()


def test_the_self_view_can_be_switched_off(qapp, keyboard):
    pipeline = ScriptedPipeline()
    pipeline.preview_enabled = False
    widget = KeyboardOverlay(keyboard, DwellController(keyboard), pipeline,
                             None, screen_size=SCREEN, show_webcam=False)
    try:
        assert not pipeline.preview_enabled
        assert widget.webcam_image() is None
    finally:
        widget.close()
