"""GazeKey — the one-point touch-up screen (spec Section 5.5).

The whole interruption is one dot in the middle of a dark screen and about two
and a half seconds. It borrows the calibration screen's visual language on
purpose — a still, neutral dot with a thin ring quietly filling around it,
nothing animating near the fixation point — because the user has already
learned to hold a stare on exactly that.

There is no result screen: the moment the measurement lands, the keyboard comes
back with the outcome in its corner. Anything longer and the sentence the user
was in the middle of is lost.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

from gaze.drift import TouchUpPhase, TouchUpSession

BACKGROUND = QColor(12, 12, 16)
TEXT = QColor(232, 232, 238)
DIM = QColor(140, 140, 152)
TARGET_NEUTRAL = QColor(226, 226, 232)
TRACK = QColor(58, 58, 70)
PROGRESS = QColor(120, 200, 255)


class TouchUpScreen(QWidget):
    """Fullscreen single-target correction.

    Signals:
        finished: the :class:`~gaze.drift.TouchUpResult`, or ``None`` if the
            user cancelled with Esc.
    """

    finished = pyqtSignal(object)

    def __init__(
        self,
        session: TouchUpSession,
        pipeline=None,
        sound: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
        tick_ms: int = 16,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.pipeline = pipeline
        self.sound = sound

        self._gaze: Optional[Tuple[float, float]] = None
        self._done_emitted = False

        self.setWindowTitle("GazeKey — fix aim")
        self.setStyleSheet("background: #0c0c10;")
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(tick_ms)

    # ------------------------------------------------------------------ input
    def _tick(self) -> None:
        if self.pipeline is not None:
            for sample in self.pipeline.drain():
                self._gaze = ((sample.x, sample.y)
                              if sample.has_gaze and sample.stream_valid else None)
                result = self.session.update(
                    sample.x, sample.y, sample.is_fixating,
                    sample.stream_valid, sample.timestamp,
                )
                if result is not None:
                    self._emit(result)
        self.update()

    def _emit(self, result) -> None:
        if self._done_emitted:
            return
        self._done_emitted = True
        if result is not None and self.sound is not None:
            self.sound()
        self.finished.emit(result)

    def keyPressEvent(self, event: QKeyEvent) -> None:   # noqa: N802 - Qt API
        if event.key() == Qt.Key_Escape:
            self.session.cancel()
            self._emit(None)

    # -------------------------------------------------------------------- copy
    def text_lines(self) -> List[str]:
        """On-screen copy (the offscreen test platform has no fonts)."""
        if self.session.phase is TouchUpPhase.SETTLING:
            return ["Look at the dot", "Fixing your aim — this takes a moment",
                    "Esc to cancel"]
        return ["Look at the dot", "Hold still…", "Esc to cancel"]

    # ---------------------------------------------------------------- painting
    def paintEvent(self, event: QPaintEvent) -> None:    # noqa: N802 - Qt API
        painter = QPainter(self)
        try:
            self.render_into(painter)
        finally:
            painter.end()

    def render_into(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
        self._paint_target(painter)
        self._paint_text(painter)

    def _scale(self) -> Tuple[float, float]:
        width, height = self.session.screen_size
        return self.width() / max(width, 1), self.height() / max(height, 1)

    def _to_widget(self, position) -> QPointF:
        sx, sy = self._scale()
        return QPointF(position[0] * sx, position[1] * sy)

    def _paint_target(self, painter: QPainter) -> None:
        centre = self._to_widget(self.session.target)
        base = self.height() * 0.042

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(TRACK, max(1.5, self.height() * 0.0022)))
        painter.drawEllipse(centre, base, base)

        progress = self.session.progress
        if progress > 0.0:
            painter.setPen(QPen(PROGRESS, max(3.0, self.height() * 0.0045),
                                Qt.SolidLine, Qt.RoundCap))
            box = QRectF(centre.x() - base, centre.y() - base, base * 2, base * 2)
            painter.drawArc(box, 90 * 16, int(-360 * 16 * progress))

        painter.setPen(Qt.NoPen)
        painter.setBrush(TARGET_NEUTRAL)
        dot = self.height() * 0.009
        painter.drawEllipse(centre, dot, dot)

    def _paint_text(self, painter: QPainter) -> None:
        for line, top, size, colour in zip(
            self.text_lines(), (0.14, 0.20, 0.92), (34, 22, 18),
            (TEXT, DIM, DIM),
        ):
            font = QFont("Segoe UI")
            font.setPixelSize(max(9, int(size * self.height() / 1080)))
            painter.setFont(font)
            painter.setPen(colour)
            painter.drawText(
                QRectF(self.width() * 0.1, self.height() * top,
                       self.width() * 0.8, self.height() * 0.12),
                Qt.AlignHCenter | Qt.AlignTop, line,
            )

    def closeEvent(self, event) -> None:                 # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
