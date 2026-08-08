"""GazeKey — keyboard geometry and layouts (spec Section 8).

All key geometry is defined here, once, and exposed as a list of :class:`Key`.

**Key size is driven by measured accuracy, not by taste.** Spec NFR-2 wants a
key at least twice the expected gaze error, so a prediction that lands one
error away from where the user was looking still falls inside the intended key.

**NFR-2 is checked on the smallest selectable key's *minimum dimension*** — a
key that is wide but short fails just as surely as one that is narrow, because
an error of one radius in the short direction lands outside it. The check
therefore measures both axes over the real key rectangles (not a nominal cell
size) and every "BELOW NFR-2" message names both, so the axis that actually
binds is never left to be inferred. On a 1366x768 screen the tall QWERTY's
letters are 137 px wide but only 93 px tall at the default height, so *height*
is what binds: it clears NFR-2 up to ~46 px of error, not the ~68 px the width
alone would suggest.

:func:`required_key_px` turns the calibration's validation error into that
minimum, and :func:`build_keyboard` picks the densest layout whose keys clear
it on the actual screen:

``qwerty``
    the spec's 10-column QWERTY, used whenever it fits;
``paged``
    8 columns over two alphabetical pages, for when it does not — a full
    QWERTY row needs 10 x the minimum key width, which a large error on a
    small screen simply cannot provide.

The chosen layout, the key size and whether NFR-2 is actually met are all
reported on the :class:`Keyboard`, never assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from gaze.region import Region, full_screen_region

#: keys whose action is destructive or modal need a longer stare (spec 7).
#: ``touchup`` is here because it *rewrites the calibration* from wherever the
#: user happens to be looking — a mis-dwell on it is worse than a mistyped
#: letter, so it costs the same 2 s stare as Recalibrate.
EXTENDED_DWELL_ACTIONS = frozenset({"pause", "recalibrate", "touchup", "lang",
                                    "quit"})

#: how much of the screen height the main keyboard takes by default
DEFAULT_HEIGHT_RATIO = 2.0 / 3.0

#: most of the screen any keyboard may occupy — the bound used to size the
#: calibration region for layouts whose height depends on the (not yet
#: measured) calibration error
MAX_HEIGHT_RATIO = 0.8

#: How far beyond the board the calibration region reaches, as a fraction of
#: screen height. Only the top edge actually moves: the board is docked to the
#: bottom and spans the full width, so the other three sides are already at the
#: screen edge. The margin keeps the topmost row — the suggestion bar — away
#: from the boundary of the fitted area, where the polynomial is weakest.
#:
#: **7.5%, not 5%, and the difference is the point.** The dots sit at
#: 10/50/90% *of the region*, so their convex hull is inset by a tenth of the
#: region on every side — the margin has to be large enough to push that hull
#: above the board, not merely above the region's own edge. On 1366x768 at the
#: 2/3 height:
#:
#:   margin 5.0% -> hull rows at y 273..713, board 256..768 (top row OUTSIDE)
#:   margin 7.5% -> hull rows at y 255..711, board 256..768 (top row covered)
#:
#: The bottom row cannot be covered at any margin — the board is flush with the
#: screen edge, so the last 10% of the region is always below the lowest dot.
#: Region-scoping still shrinks that overhang (57 px here, against 77 px for a
#: whole-screen calibration), and the ±3σ clamp in the verified core keeps the
#: extrapolation bounded rather than explosive.
REGION_MARGIN_RATIO = 0.075

#: row heights of the tall layout, in units of one key row
SUGGESTION_ROW_UNITS = 1.0
PREVIEW_ROW_UNITS = 0.5
LETTER_ROWS = 3
CONTROL_ROWS = 1
TALL_TOTAL_UNITS = (SUGGESTION_ROW_UNITS + PREVIEW_ROW_UNITS
                    + LETTER_ROWS + CONTROL_ROWS)

#: how many suggestion slots the bar holds
SUGGESTION_SLOTS = 4

#: columns reserved at the right of the suggestion/preview block for the webcam
WEBCAM_COLUMNS = 2

#: NFR-2: a key must be at least this multiple of the expected gaze error
NFR2_KEY_ERROR_RATIO = 2.0

#: never build keys smaller than this, however good the calibration claims to be
MIN_KEY_FLOOR_PX = 90.0

#: the suggestion bar is display-only until M6, so it needs no dwell-sized keys
SUGGESTION_ROW_RATIO = 0.45

#: the control row holds 8 keys, so no layout can be narrower than that. It is
#: also what caps usable accuracy on a given screen: a 1366 px display cannot
#: satisfy NFR-2 beyond about 1366 / 8 / 2 = 85 px of calibration error.
MIN_COLUMNS = 8

#: The tall layout's control row is one column denser than its letter rows, so
#: the M5 touch-up key fits without shrinking Space. Harmless for NFR-2 at the
#: default height: 1366/11 = 124 px of width still clears the 93 px that the
#: row height allows, so height stays the binding axis.
CONTROL_COLUMNS = 11

ACTION_LABELS: Dict[str, Tuple[str, str]] = {
    "space": ("Space", "רווח"),
    "backspace": ("← Del", "← מחק"),
    "enter": ("Enter", "שורה"),
    "shift": ("Shift", "Shift"),
    "page": ("More >", "עוד >"),
    "symbols": ("123", "123"),
    "lang": ("EN / HE", "EN / HE"),
    "pause": ("Pause", "השהה"),
    "touchup": ("Fix aim", "תיקון"),
    "recalibrate": ("Recal.", "כיול"),
    "quit": ("Quit", "יציאה"),
}

#: the symbols page, three rows of ten
SYMBOL_ROWS: Tuple[Tuple[str, ...], ...] = (
    tuple("1234567890"),
    ("-", "/", ":", ";", "(", ")", "$", "&", "@", '"'),
    (".", ",", "?", "!", "'", "+", "=", "*", "#", "%"),
)

#: alphabetical order for paged layouts, chunked into pages of ``columns * 2``
PAGED_CHARS: Tuple[str, ...] = (
    tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + (".", ",", "?", "!", "'", "-")
)


def paged_cells(columns: int) -> List[Tuple[str, ...]]:
    """Split the alphabet into pages of two rows for a given column count."""
    per_page = columns * 2
    return [tuple(PAGED_CHARS[i:i + per_page])
            for i in range(0, len(PAGED_CHARS), per_page)]

#: the spec's QWERTY grid, used whenever the geometry allows it
QWERTY_ROWS: Tuple[Tuple[str, ...], ...] = (
    tuple("QWERTYUIOP"),
    tuple("ASDFGHJKL"),
    tuple("ZXCVBNM"),
)

#: Hebrew letters on the same physical grid (standard Israeli layout order)
HEBREW_QWERTY_ROWS: Tuple[Tuple[str, ...], ...] = (
    tuple("/'קראטוןםפ"),
    tuple("שדגכעיחלך"),
    tuple("זסבהנמצ"),
)


@dataclass(frozen=True)
class Key:
    """One key. ``rect`` is ``(x, y, w, h)`` in screen pixels."""

    id: str
    label_en: str
    label_he: str
    rect: Tuple[float, float, float, float]
    is_function: bool = False
    action: str = "char"
    payload: str = ""
    selectable: bool = True
    pages: Tuple[int, ...] = ()      # empty = present on every page

    def label(self, language: str = "en") -> str:
        return self.label_he if language == "he" else self.label_en

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Is ``(x, y)`` inside this key, optionally grown by ``margin`` sides?"""
        kx, ky, kw, kh = self.rect
        grow_x, grow_y = kw * margin, kh * margin
        return (kx - grow_x <= x <= kx + kw + grow_x
                and ky - grow_y <= y <= ky + kh + grow_y)

    @property
    def centre(self) -> Tuple[float, float]:
        kx, ky, kw, kh = self.rect
        return kx + kw / 2.0, ky + kh / 2.0

    def on_page(self, page: int) -> bool:
        return not self.pages or page in self.pages


@dataclass
class Keyboard:
    """A built keyboard: every key, plus how and why it was sized."""

    keys: List[Key]
    name: str
    pages: int
    rect: Tuple[float, float, float, float]
    key_size: Tuple[float, float]
    min_key_px: float
    columns: int
    rows: int
    meets_nfr2: bool = True
    note: str = ""
    language: str = "en"
    #: where the webcam thumbnail goes, if the layout reserves room for one
    webcam_rect: Optional[Tuple[float, float, float, float]] = None
    #: where the typed-text preview line is drawn
    preview_rect: Optional[Tuple[float, float, float, float]] = None

    def keys_on(self, page: int) -> List[Key]:
        return [key for key in self.keys if key.on_page(page)]

    def selectable_on(self, page: int) -> List[Key]:
        return [key for key in self.keys_on(page) if key.selectable]

    def find(self, key_id: str) -> Optional[Key]:
        return next((key for key in self.keys if key.id == key_id), None)

    @property
    def suggestion_keys(self) -> List[Key]:
        return sorted((key for key in self.keys if key.action == "suggestion"),
                      key=lambda key: key.rect[0])

    def has_action(self, action: str) -> bool:
        """Is there a key for this action anywhere on the board?"""
        return any(key.action == action for key in self.keys)

    # ------------------------------------------------------------------ NFR-2
    def smallest_key(self) -> Tuple[float, float]:
        """``(width, height)`` of the smallest *selectable* key, per axis.

        Measured over the real rectangles, so a control row at a different
        column count or a short strip cannot hide behind the nominal cell size.
        The two minima may come from different keys — that is deliberate: NFR-2
        has to hold for every key on both axes.
        """
        boxes = [(key.rect[2], key.rect[3]) for key in self.keys if key.selectable]
        if not boxes:
            return self.key_size
        return min(w for w, _ in boxes), min(h for _, h in boxes)

    def nfr2_axes(self) -> Tuple[bool, bool]:
        """Does the smallest key clear the requirement on ``(width, height)``?"""
        width, height = self.smallest_key()
        return (width >= self.min_key_px - 1e-6,
                height >= self.min_key_px - 1e-6)

    def check_nfr2(self) -> bool:
        """Recompute :attr:`meets_nfr2` from the actual keys; returns it."""
        self.meets_nfr2 = all(self.nfr2_axes())
        return self.meets_nfr2

    def nfr2_line(self) -> str:
        """Both axes, spelled out — never just the one that happens to pass."""
        width, height = self.smallest_key()
        ok_w, ok_h = self.nfr2_axes()
        return (f"smallest key {width:.0f}x{height:.0f} px, needs "
                f"{self.min_key_px:.0f} px on both axes "
                f"(width {'OK' if ok_w else 'BELOW'}, "
                f"height {'OK' if ok_h else 'BELOW'})")

    def usable_error_px(self) -> float:
        """Largest calibration error this board's keys still satisfy NFR-2 for."""
        return min(self.smallest_key()) / NFR2_KEY_ERROR_RATIO

    def summary(self) -> str:
        fit = "meets NFR-2" if self.meets_nfr2 else "BELOW NFR-2"
        return (f"{self.name} layout, {self.columns}x{self.rows} keys of "
                f"{self.key_size[0]:.0f}x{self.key_size[1]:.0f} px "
                f"({self.pages} page{'s' if self.pages > 1 else ''}), "
                f"{self.nfr2_line()} - {fit}")


def required_key_px(validation_error_px: float,
                    floor: float = MIN_KEY_FLOOR_PX) -> float:
    """Smallest key that satisfies NFR-2 for a given calibration accuracy."""
    if (validation_error_px is None
            or validation_error_px != validation_error_px    # NaN
            or validation_error_px in (float("inf"), 0.0)):
        return floor
    return max(floor, NFR2_KEY_ERROR_RATIO * float(validation_error_px))


# --------------------------------------------------------------------- builders
def _grid_rect(rect, columns, rows, col, row, span=1) -> Tuple[float, ...]:
    x, y, w, h = rect
    cell_w, cell_h = w / columns, h / rows
    return (x + col * cell_w, y + row * cell_h, cell_w * span, cell_h)


def _function_key(action: str, rect, page_label: Optional[str] = None) -> Key:
    label_en, label_he = ACTION_LABELS[action]
    return Key(
        id=f"fn.{action}",
        label_en=page_label or label_en,
        label_he=page_label or label_he,
        rect=rect,
        is_function=True,
        action=action,
    )


def _suggestion_keys(rect) -> List[Key]:
    """Three display-only slots; they become dwell targets in M6."""
    x, y, w, h = rect
    slot = w / 3.0
    return [
        Key(id=f"sug.{i}", label_en="", label_he="",
            rect=(x + i * slot, y, slot, h),
            is_function=True, action="suggestion", selectable=False)
        for i in range(3)
    ]


def _letter_key(char: str, rect, page: int, hebrew: str = "") -> Key:
    return Key(
        id=f"key.{char}",
        label_en=char,
        label_he=hebrew or char,
        rect=rect,
        action="char",
        payload=char,
        pages=(page,),
    )


def _build_qwerty(screen, key_px, language) -> Keyboard:
    columns, rows = 10, 4
    width, height = screen
    key_w, key_h = width / columns, key_px
    strip_h = key_px * SUGGESTION_ROW_RATIO
    board_h = rows * key_h + strip_h
    top = height - board_h
    rect = (0.0, top + strip_h, float(width), rows * key_h)

    keys: List[Key] = _suggestion_keys((0.0, top, float(width), strip_h))
    hebrew_rows = HEBREW_QWERTY_ROWS
    for row_index, chars in enumerate(QWERTY_ROWS):
        offset = 1 if row_index == 2 else 0        # Shift sits before Z
        for col, char in enumerate(chars):
            hebrew = ""
            if row_index < len(hebrew_rows) and col < len(hebrew_rows[row_index]):
                hebrew = hebrew_rows[row_index][col]
            keys.append(_letter_key(
                char, _grid_rect(rect, columns, rows, col + offset, row_index),
                page=0, hebrew=hebrew))

    keys.append(_function_key(
        "backspace", _grid_rect(rect, columns, rows, 9, 1)))
    keys.append(_function_key("shift", _grid_rect(rect, columns, rows, 0, 2)))
    keys.append(Key(id="key.,", label_en=",", label_he=",",
                    rect=_grid_rect(rect, columns, rows, 8, 2),
                    action="char", payload=","))
    keys.append(Key(id="key..", label_en=".", label_he=".",
                    rect=_grid_rect(rect, columns, rows, 9, 2),
                    action="char", payload="."))

    keys.append(_function_key("lang", _grid_rect(rect, columns, rows, 0, 3)))
    keys.append(_function_key("space", _grid_rect(rect, columns, rows, 1, 3, span=5)))
    keys.append(_function_key("touchup", _grid_rect(rect, columns, rows, 6, 3)))
    keys.append(_function_key("enter", _grid_rect(rect, columns, rows, 7, 3)))
    keys.append(_function_key("pause", _grid_rect(rect, columns, rows, 8, 3)))
    keys.append(_function_key("recalibrate", _grid_rect(rect, columns, rows, 9, 3)))

    return Keyboard(
        keys=keys, name="qwerty", pages=1,
        rect=(0.0, top, float(width), board_h),
        key_size=(key_w, key_h), min_key_px=key_px,
        columns=columns, rows=rows, language=language,
    )


def _build_paged(screen, key_px, language, columns: int = 8) -> Keyboard:
    rows = 3                                   # two letter rows + one control row
    width, height = screen
    key_w, key_h = width / columns, key_px
    strip_h = key_px * SUGGESTION_ROW_RATIO
    board_h = rows * key_h + strip_h
    top = height - board_h
    rect = (0.0, top + strip_h, float(width), rows * key_h)

    cells = paged_cells(columns)
    keys: List[Key] = _suggestion_keys((0.0, top, float(width), strip_h))
    for page, chars in enumerate(cells):
        for index, char in enumerate(chars):
            col, row = index % columns, index // columns
            keys.append(_letter_key(char, _grid_rect(rect, columns, rows, col, row),
                                    page=page))

    controls = ["space", "backspace", "enter", "shift",
                "page", "lang", "pause", "recalibrate"]
    if columns < len(controls):
        raise ValueError(f"{columns} columns cannot hold the control row")
    # "Fix aim" only joins the row if one more column still clears the required
    # key size — on a cramped board a touch-up key is not worth shrinking every
    # control below NFR-2 for. Where it does not fit, Recalibrate still does the
    # same job, slower.
    control_columns = columns
    if width / (columns + 1) >= key_px - 1e-6:
        controls.append("touchup")
        control_columns = columns + 1
    for col, action in enumerate(controls):
        keys.append(_function_key(
            action, _grid_rect(rect, control_columns, rows, col, 2)))

    return Keyboard(
        keys=keys, name="paged", pages=len(cells),
        rect=(0.0, top, float(width), board_h),
        key_size=(key_w, key_h), min_key_px=key_px,
        columns=columns, rows=rows, language=language,
    )


def _build_tall_qwerty(screen, height_ratio: float, language: str) -> Keyboard:
    """The main layout: suggestion bar, typed preview, QWERTY, control row.

    Modelled on commercial gaze keyboards — everything reachable without
    paging, with the generous height making up for keys that a 10-column row
    keeps narrow.
    """
    columns = 10
    width, height = screen
    board_h = height * height_ratio
    unit = board_h / TALL_TOTAL_UNITS
    key_w = width / columns
    top = height - board_h

    suggestion_h = unit * SUGGESTION_ROW_UNITS
    preview_h = unit * PREVIEW_ROW_UNITS
    grid_top = top + suggestion_h + preview_h
    grid = (0.0, grid_top, float(width), unit * (LETTER_ROWS + CONTROL_ROWS))
    rows = LETTER_ROWS + CONTROL_ROWS

    # suggestion bar and webcam share the top block; the webcam sits at the
    # right because the bottom-right of the board is control keys and every
    # alternative there would either shrink Space or displace Quit
    webcam_w = key_w * WEBCAM_COLUMNS
    bar_w = width - webcam_w
    slot_w = bar_w / SUGGESTION_SLOTS
    keys: List[Key] = [
        Key(id=f"sug.{i}", label_en="", label_he="",
            rect=(i * slot_w, top, slot_w, suggestion_h),
            is_function=True, action="suggestion", payload=str(i))
        for i in range(SUGGESTION_SLOTS)
    ]
    webcam_rect = (bar_w, top, webcam_w, suggestion_h + preview_h)
    preview_rect = (0.0, top + suggestion_h, bar_w, preview_h)
    keys.append(Key(id="preview", label_en="", label_he="", rect=preview_rect,
                    is_function=True, action="preview", selectable=False))

    for row_index, chars in enumerate(QWERTY_ROWS):
        for col, char in enumerate(chars):
            hebrew = ""
            if (row_index < len(HEBREW_QWERTY_ROWS)
                    and col < len(HEBREW_QWERTY_ROWS[row_index])):
                hebrew = HEBREW_QWERTY_ROWS[row_index][col]
            keys.append(_letter_key(
                char, _grid_rect(grid, columns, rows, col, row_index),
                page=0, hebrew=hebrew))
    for col, char in ((9, "'"), ):
        keys.append(Key(id=f"key.{char}", label_en=char, label_he=char,
                        rect=_grid_rect(grid, columns, rows, col, 1),
                        action="char", payload=char, pages=(0,)))
    for col, char in ((7, ","), (8, "."), (9, "?")):
        keys.append(Key(id=f"key.{char}", label_en=char, label_he=char,
                        rect=_grid_rect(grid, columns, rows, col, 2),
                        action="char", payload=char, pages=(0,)))

    for row_index, chars in enumerate(SYMBOL_ROWS):
        for col, char in enumerate(chars):
            keys.append(Key(id=f"sym.{row_index}.{col}", label_en=char,
                            label_he=char,
                            rect=_grid_rect(grid, columns, rows, col, row_index),
                            action="char", payload=char, pages=(1,)))

    # The control row runs on its own, denser column grid (CONTROL_COLUMNS) so
    # "Fix aim" fits beside Recalibrate without shrinking Space or displacing
    # Quit. Both grids span the full width, so the rows stay gapless.
    controls = ("backspace", "enter", "shift", "symbols", "lang",
                "pause", "touchup", "recalibrate", "quit")
    keys.append(_function_key(
        "space", _grid_rect(grid, CONTROL_COLUMNS, rows, 0, 3, span=2)))
    for offset, action in enumerate(controls):
        keys.append(_function_key(
            action, _grid_rect(grid, CONTROL_COLUMNS, rows, 2 + offset, 3)))

    return Keyboard(
        keys=keys, name="qwerty-tall", pages=2,
        rect=(0.0, top, float(width), board_h),
        key_size=(key_w, unit), min_key_px=0.0,
        columns=columns, rows=rows, language=language,
        webcam_rect=webcam_rect, preview_rect=preview_rect,
    )


def keyboard_region(screen_size: Tuple[int, int],
                    height_ratio: float = DEFAULT_HEIGHT_RATIO,
                    margin_ratio: float = REGION_MARGIN_RATIO) -> Region:
    """The screen area worth calibrating: the keyboard, plus a top margin.

    Derived from the height ratio rather than from a built board, because the
    board cannot be built until the calibration error is known and the
    calibration cannot run until the region is. For ``qwerty-tall`` the two
    agree exactly (its geometry depends only on the ratio); for the
    error-sized layouts pass :data:`MAX_HEIGHT_RATIO`, which bounds them.
    """
    width, height = screen_size
    board_h = float(height) * height_ratio
    top = max(0.0, float(height) - board_h - float(height) * margin_ratio)
    return Region((0.0, top, float(width), float(height) - top), name="keyboard")


def build_choice_board(screen_size: Tuple[int, int],
                       options: Sequence[Tuple[str, str]],
                       language: str = "en",
                       region: Optional[Region] = None) -> Keyboard:
    """A row of very large keys for a gaze question ("YES, QUIT / KEEP TYPING").

    Returned as a :class:`Keyboard` so the dwell controller and the key
    renderer drive it unchanged — same timings, same look, nothing new to learn.

    The options are placed **inside the calibration region**: a target at the
    centre of the screen would sit outside the fitted area, and asking someone
    to confirm a quit by aiming at accuracy nobody measured is how a user ends
    up unable to answer.
    """
    region = region or full_screen_region(screen_size)
    count = max(1, len(options))
    box_w = region.w * 0.8 / count
    box_h = region.h * 0.35
    left = region.x + region.w * 0.1
    top = region.y + (region.h - box_h) / 2.0
    keys = [
        Key(id=f"choice.{key_id}", label_en=label, label_he=label,
            rect=(left + index * box_w, top, box_w, box_h),
            is_function=True, action="choice", payload=key_id)
        for index, (key_id, label) in enumerate(options)
    ]
    return Keyboard(
        keys=keys, name="choice", pages=1,
        rect=(left, top, box_w * count, box_h),
        key_size=(box_w, box_h), min_key_px=0.0,
        columns=count, rows=1, language=language,
    )


def _target_px(error_px: float) -> int:
    """An accuracy target the user can actually hit — rounded *down*.

    "Recalibrate to about 47 px" when 46.5 is the real limit is advice that
    fails when followed, so every accuracy target in these messages floors.
    """
    return int(math.floor(max(0.0, error_px)))


def _axis_note(board: Keyboard, validation_error_px: float) -> str:
    """The head of every BELOW-NFR-2 message: both axes, measured, named."""
    width, height = board.smallest_key()
    ok_w, ok_h = board.nfr2_axes()
    binding = "height" if height <= width else "width"
    return (f"keys are {width:.0f} px wide x {height:.0f} px tall - "
            f"width {'OK' if ok_w else 'BELOW'}, "
            f"height {'OK' if ok_h else 'BELOW'} - but NFR-2 wants "
            f"{board.min_key_px:.0f} px on BOTH axes for a "
            f"{validation_error_px:.0f} px calibration; {binding} is what binds, "
            f"and as built this board is only good to "
            f"{_target_px(board.usable_error_px())} px")


def _tall_note(board: Keyboard, validation_error_px: float,
               screen_size: Tuple[int, int], height_ratio: float) -> str:
    """Why the tall QWERTY misses, and which of the two remedies applies."""
    width, height = screen_size
    key_w, _key_h = board.smallest_key()
    ok_w, ok_h = board.nfr2_axes()
    remedies: List[str] = []
    if not ok_h:
        needed_ratio = board.min_key_px * TALL_TOTAL_UNITS / max(height, 1)
        if needed_ratio <= 1.0:
            remedies.append(
                f"raise keyboard_height_ratio to {needed_ratio:.2f} "
                f"(now {height_ratio:.2f}) to fix the height"
            )
        else:
            remedies.append(
                f"no keyboard height fits {board.min_key_px:.0f} px rows on a "
                f"{height} px screen (it would need "
                f"keyboard_height_ratio {needed_ratio:.2f})"
            )
    if not ok_w:
        remedies.append(
            f"the board's narrowest column is {key_w:.0f} px on a {width} px "
            f"screen (10 letter columns, {CONTROL_COLUMNS} control columns), so "
            f"this layout tops out at a "
            f"{_target_px(key_w / NFR2_KEY_ERROR_RATIO)} px calibration "
            f"however tall it gets"
        )
    remedies.append(
        f"recalibrate to about {_target_px(board.usable_error_px())} px, or use "
        f"--layout paged (8 columns, up to "
        f"{_target_px(width / MIN_COLUMNS / NFR2_KEY_ERROR_RATIO)} px)"
    )
    return _axis_note(board, validation_error_px) + ". " + "; ".join(remedies)


def build_keyboard(
    screen_size: Tuple[int, int],
    validation_error_px: float,
    language: str = "en",
    layout: str = "qwerty-tall",
    height_ratio: float = DEFAULT_HEIGHT_RATIO,
    max_height_ratio: float = 0.8,
) -> Keyboard:
    """Build the densest keyboard whose keys satisfy NFR-2 on this screen.

    Args:
        screen_size: ``(width, height)`` in the same pixels as the calibration.
        validation_error_px: the measured calibration error.
        language: ``"en"`` or ``"he"`` (labels only; geometry is shared).
        max_height_ratio: most of the screen the keyboard may occupy.

    Returns:
        A :class:`Keyboard`. If nothing fits at the NFR-2 size it still returns
        a usable board, with ``meets_nfr2=False`` and ``note`` explaining what
        accuracy would be needed — it never silently pretends to comply.
    """
    width, height = screen_size
    key_px = required_key_px(validation_error_px)
    budget = height * max_height_ratio

    if layout == "qwerty-tall":
        board = _build_tall_qwerty(screen_size, height_ratio, language)
        board.min_key_px = key_px
        if not board.check_nfr2():
            board.note = _tall_note(board, validation_error_px, screen_size,
                                    height_ratio)
        return board
    if layout == "paged":
        board = _build_paged(screen_size, max(key_px, width / MIN_COLUMNS),
                             language, MIN_COLUMNS)
        board.min_key_px = key_px
        if not board.check_nfr2():
            board.note = _axis_note(board, validation_error_px) + (
                f". Recalibrate to about {_target_px(board.usable_error_px())} px, or "
                f"use a wider screen"
            )
        return board

    for builder, columns in ((_build_qwerty, 10), (_build_paged, MIN_COLUMNS)):
        if width / columns < key_px:
            continue
        board = (builder(screen_size, key_px, language) if columns == 10
                 else builder(screen_size, key_px, language, columns))
        if board.rect[3] <= budget and board.check_nfr2():
            return board

    # Nothing fits at the NFR-2 size. Build the largest board that does fit and
    # name the constraint that actually binds, because the two have different
    # remedies: a narrow screen needs a better calibration, a short one needs
    # more height.
    rows = 3
    by_width = width / MIN_COLUMNS
    by_height = budget / (rows + SUGGESTION_ROW_RATIO)
    board = _build_paged(screen_size, min(by_width, by_height), language,
                         MIN_COLUMNS)
    board.min_key_px = key_px
    if not board.check_nfr2():
        head = _axis_note(board, validation_error_px)
        if by_width < key_px:
            board.note = (
                f"{head}. A {MIN_COLUMNS}-column keyboard needs "
                f"{MIN_COLUMNS * key_px:.0f} px of width for a "
                f"{validation_error_px:.0f} px calibration, but this screen is "
                f"{width} px wide; recalibrate to about "
                f"{_target_px(by_width / NFR2_KEY_ERROR_RATIO)} px "
                f"or use a wider screen"
            )
        else:
            board.note = (
                f"{head}. Keys need {key_px:.0f} px for a "
                f"{validation_error_px:.0f} px calibration but only "
                f"{by_height:.0f} px of height is allowed; "
                f"raise --max-height-ratio"
            )
    return board
