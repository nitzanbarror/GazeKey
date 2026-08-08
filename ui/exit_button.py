"""GazeKey — the clickable exit button.

The keyboard overlay is click-through, which is what lets the user click into
the app they are typing into — but it also means nothing on the overlay can be
clicked, including a close button. So the X lives in its own tiny always-on-top
window that is *not* click-through, parked in the keyboard's top-right corner.

It still refuses keyboard focus (``WindowDoesNotAcceptFocus`` maps to
``WS_EX_NOACTIVATE`` on Windows), so clicking it does not pull focus away from
whatever the user was typing into.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

FACE = QColor(30, 32, 40, 235)
FACE_HOVER = QColor(120, 46, 52, 245)
EDGE = QColor(120, 126, 142)
GLYPH = QColor(238, 238, 244)

SIZE_PX = 44


class ExitButton(QWidget):
    """A small always-on-top X. Clicking it emits :attr:`clicked_quit`."""

    clicked_quit = pyqtSignal()

    def __init__(self, size: int = SIZE_PX, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hover = False

        self.setWindowTitle("GazeKey — quit")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self.setToolTip("Quit GazeKey")
        self.resize(size, size)

    def place_at(self, corner: Tuple[float, float], inset: int = 10) -> None:
        """Park the button with its top-right at ``corner`` (screen pixels)."""
        x = int(corner[0] - self.width() - inset)
        y = int(corner[1] + inset)
        self.move(max(0, x), max(0, y))

    # ------------------------------------------------------------------ mouse
    def enterEvent(self, event) -> None:            # noqa: N802 - Qt API
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:            # noqa: N802 - Qt API
        self._hover = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:   # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.clicked_quit.emit()

    # ---------------------------------------------------------------- drawing
    def paintEvent(self, event: QPaintEvent) -> None:        # noqa: N802 - Qt API
        painter = QPainter(self)
        try:
            self.render_into(painter)
        finally:
            painter.end()

    def render_into(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(EDGE, 1.2))
        painter.setBrush(FACE_HOVER if self._hover else FACE)
        painter.drawRoundedRect(box, 8, 8)

        inset = min(box.width(), box.height()) * 0.32
        painter.setPen(QPen(GLYPH, max(2.0, box.width() * 0.07), Qt.SolidLine,
                            Qt.RoundCap))
        painter.drawLine(QPointF(box.left() + inset, box.top() + inset),
                         QPointF(box.right() - inset, box.bottom() - inset))
        painter.drawLine(QPointF(box.right() - inset, box.top() + inset),
                         QPointF(box.left() + inset, box.bottom() - inset))
