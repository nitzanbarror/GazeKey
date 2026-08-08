"""GazeKey — keyboard rendering and dwell feedback (spec Section 9).

A painter, not a widget: :class:`KeyboardRenderer` draws the board and its
focus/progress feedback onto whatever :class:`QPainter` it is handed, so the
overlay can composite the keyboard, the gaze cursor and the status icons into
one translucent click-through window without child-widget layering games.

Feedback, in the order the user notices it:

* the focused key lightens and gains a border;
* a circular **progress ring** fills around its centre as the dwell advances;
* activation leaves a brief flash on the key.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen

from interaction.layouts import Key, Keyboard

# Near-black keys, white labels, thin borders — the commercial gaze-keyboard
# look: maximum contrast for the labels, minimum glare from the surrounding
# screen while the user stares at it for long stretches.
BOARD_BACKING = QColor(6, 6, 8, 242)
KEY_FACE = QColor(18, 18, 22, 255)
KEY_FACE_FUNCTION = QColor(12, 12, 15, 255)
KEY_EDGE = QColor(58, 60, 70)
KEY_TEXT = QColor(255, 255, 255)
KEY_TEXT_FUNCTION = QColor(214, 218, 228)

FOCUS_FACE = QColor(30, 58, 88, 255)
FOCUS_EDGE = QColor(120, 190, 255)
PROGRESS = QColor(120, 200, 255)
FLASH = QColor(150, 235, 190)

SUGGESTION_FACE = QColor(14, 16, 20, 255)
SUGGESTION_TEXT = QColor(140, 200, 255)
SUGGESTION_EMPTY = QColor(70, 74, 86)

PREVIEW_FACE = QColor(9, 9, 12, 255)
PREVIEW_TEXT = QColor(232, 234, 240)
PREVIEW_CARET = QColor(120, 200, 255)

WEBCAM_EDGE = QColor(48, 50, 60)

PAUSED_TINT = QColor(255, 190, 70, 26)

#: how long an activated key keeps its flash
FLASH_S = 0.22


class KeyboardRenderer:
    """Draws a :class:`~interaction.layouts.Keyboard` and its dwell feedback."""

    def __init__(self, keyboard: Keyboard, language: str = "en") -> None:
        self.keyboard = keyboard
        self.language = language
        self._flashes: Dict[str, float] = {}

    def flash(self, key: Key, now: Optional[float] = None) -> None:
        """Mark a key as just activated."""
        self._flashes[key.id] = (time.monotonic() if now is None else now) + FLASH_S

    def _flash_strength(self, key: Key, now: float) -> float:
        expiry = self._flashes.get(key.id)
        if expiry is None or expiry <= now:
            return 0.0
        return max(0.0, min(1.0, (expiry - now) / FLASH_S))

    # ---------------------------------------------------------------- drawing
    def render(
        self,
        painter: QPainter,
        page: int = 0,
        focused: Optional[Key] = None,
        progress: float = 0.0,
        paused: bool = False,
        shift_latched: bool = False,
        suggestions: Sequence[str] = (),
        typed_text: str = "",
        scale: Tuple[float, float] = (1.0, 1.0),
        now: Optional[float] = None,
    ) -> None:
        """Paint the board for one frame."""
        now = time.monotonic() if now is None else now
        painter.setRenderHint(QPainter.Antialiasing, True)

        board = self._rect(self.keyboard.rect, scale)
        painter.setPen(Qt.NoPen)
        painter.setBrush(BOARD_BACKING)
        painter.drawRect(board)

        for key in self.keyboard.keys_on(page):
            if key.action == "suggestion":
                self._draw_suggestion(painter, key, suggestions,
                                      focused, progress, scale, now)
            elif key.action == "preview":
                self._draw_preview(painter, key, typed_text, scale)
            else:
                is_focused = focused is not None and key.id == focused.id
                self._draw_key(painter, key, is_focused,
                               progress if is_focused else 0.0,
                               shift_latched, scale, now)

        if paused:
            painter.setPen(Qt.NoPen)
            painter.setBrush(PAUSED_TINT)
            painter.drawRect(board)

    @staticmethod
    def _rect(rect, scale) -> QRectF:
        sx, sy = scale
        x, y, w, h = rect
        return QRectF(x * sx, y * sy, w * sx, h * sy)

    def _draw_suggestion(self, painter, key, suggestions, focused, progress,
                         scale, now) -> None:
        box = self._rect(key.rect, scale)
        inset = min(box.width(), box.height()) * 0.03
        face_box = box.adjusted(inset, inset, -inset, -inset)
        index = int(key.payload or key.id.rsplit(".", 1)[-1])
        text = suggestions[index] if index < len(suggestions) else ""
        is_focused = focused is not None and key.id == focused.id

        painter.setPen(QPen(FOCUS_EDGE if is_focused else KEY_EDGE,
                            2.0 if is_focused else 1.0))
        painter.setBrush(FOCUS_FACE if is_focused else SUGGESTION_FACE)
        painter.drawRoundedRect(face_box, 6, 6)

        strength = self._flash_strength(key, now)
        if strength > 0.0:
            flash = QColor(FLASH)
            flash.setAlphaF(0.5 * strength)
            painter.setPen(Qt.NoPen)
            painter.setBrush(flash)
            painter.drawRoundedRect(face_box, 6, 6)

        painter.setPen(SUGGESTION_TEXT if text else SUGGESTION_EMPTY)
        painter.setFont(self._font(face_box.height() * 0.42, bold=bool(text)))
        painter.drawText(face_box, Qt.AlignCenter, text or "·")

        if is_focused and progress > 0.0:
            self._draw_progress_ring(painter, face_box, progress)

    def _draw_preview(self, painter, key, typed_text: str, scale) -> None:
        """The line of text typed so far, tail-anchored like a real field."""
        box = self._rect(key.rect, scale)
        painter.setPen(Qt.NoPen)
        painter.setBrush(PREVIEW_FACE)
        painter.drawRect(box)

        painter.setFont(self._font(box.height() * 0.58))
        metrics = painter.fontMetrics()
        padding = box.height() * 0.35
        available = max(1.0, box.width() - padding * 2)
        text = typed_text
        while text and metrics.horizontalAdvance(text) > available:
            text = text[1:]                       # keep the end in view

        painter.setPen(PREVIEW_TEXT)
        text_box = box.adjusted(padding, 0, -padding, 0)
        painter.drawText(text_box, Qt.AlignLeft | Qt.AlignVCenter, text)

        caret_x = text_box.left() + metrics.horizontalAdvance(text) + 2
        if caret_x < text_box.right():
            painter.setPen(QPen(PREVIEW_CARET, max(1.5, box.height() * 0.04)))
            painter.drawLine(QPointF(caret_x, box.top() + box.height() * 0.22),
                             QPointF(caret_x, box.bottom() - box.height() * 0.22))

    def _draw_key(self, painter, key: Key, focused: bool, progress: float,
                  shift_latched: bool, scale, now: float) -> None:
        box = self._rect(key.rect, scale)
        inset = min(box.width(), box.height()) * 0.035
        face_box = box.adjusted(inset, inset, -inset, -inset)
        radius = min(face_box.width(), face_box.height()) * 0.12

        face = FOCUS_FACE if focused else (
            KEY_FACE_FUNCTION if key.is_function else KEY_FACE)
        painter.setPen(QPen(FOCUS_EDGE if focused else KEY_EDGE,
                            2.5 if focused else 1.0))
        painter.setBrush(face)
        painter.drawRoundedRect(face_box, radius, radius)

        strength = self._flash_strength(key, now)
        if strength > 0.0:
            flash = QColor(FLASH)
            flash.setAlphaF(0.55 * strength)
            painter.setPen(Qt.NoPen)
            painter.setBrush(flash)
            painter.drawRoundedRect(face_box, radius, radius)

        label = key.label(self.language)
        if key.action == "char" and shift_latched:
            label = label.upper()
        elif key.action == "char":
            label = label.upper()          # letters read better capitalised
        if label:
            painter.setPen(KEY_TEXT_FUNCTION if key.is_function else KEY_TEXT)
            painter.setFont(self._font(
                face_box.height() * (0.30 if key.is_function else 0.44),
                bold=not key.is_function))
            painter.drawText(face_box, Qt.AlignCenter, label)

        if key.action == "shift" and shift_latched:
            painter.setPen(QPen(FOCUS_EDGE, 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(face_box, radius, radius)

        if focused and progress > 0.0:
            self._draw_progress_ring(painter, face_box, progress)

    @staticmethod
    def _draw_progress_ring(painter: QPainter, box: QRectF, progress: float) -> None:
        size = min(box.width(), box.height()) * 0.58
        centre = box.center()
        ring = QRectF(centre.x() - size / 2, centre.y() - size / 2, size, size)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(90, 100, 120), max(2.0, size * 0.07)))
        painter.drawEllipse(ring)
        painter.setPen(QPen(PROGRESS, max(3.0, size * 0.09), Qt.SolidLine,
                            Qt.RoundCap))
        painter.drawArc(ring, 90 * 16, int(-360 * 16 * min(1.0, progress)))

    def draw_webcam(self, painter: QPainter, image: Optional[QImage],
                    scale: Tuple[float, float]) -> None:
        """Live self-view, so the user can see they are actually being tracked."""
        rect = self.keyboard.webcam_rect
        if rect is None:
            return
        box = self._rect(rect, scale)
        inset = min(box.width(), box.height()) * 0.06
        frame = box.adjusted(inset, inset, -inset, -inset)
        painter.setPen(QPen(WEBCAM_EDGE, 1.0))
        painter.setBrush(KEY_FACE)
        painter.drawRoundedRect(frame, 5, 5)
        if image is None or image.isNull():
            painter.setPen(SUGGESTION_EMPTY)
            painter.setFont(self._font(frame.height() * 0.16))
            painter.drawText(frame, Qt.AlignCenter, "camera")
            return
        scaled = image.scaled(int(frame.width()), int(frame.height()),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target = QRectF(frame.center().x() - scaled.width() / 2.0,
                        frame.center().y() - scaled.height() / 2.0,
                        scaled.width(), scaled.height())
        painter.drawImage(target, scaled)

    @staticmethod
    def _font(pixel_size: float, bold: bool = False) -> QFont:
        font = QFont("Segoe UI")
        font.setPixelSize(max(9, int(pixel_size)))
        font.setBold(bold)
        return font
