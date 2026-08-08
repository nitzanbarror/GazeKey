"""GazeKey — the setup-check screen that runs before the nine dots.

Two still dots, about five seconds, and one number: how far the vertical iris
ratio travels between the top and the bottom of the calibration region. It
borrows the calibration screen's visual language exactly — a neutral dot that
never pulses, with a thin ring filling around it — because it is the same
instruction ("look at the dot") and the user is about to do it nine more times.

When the check passes the screen never appears again: it emits its result and
calibration starts. When it fails it says what to move, shows the measurement,
and waits — the whole point of the gate is that nothing is calibrated on a
sitting that was already known to be bad.

**Answered from the keyboard, not by gaze**, and deliberately so: there is no
calibration yet, so nothing on screen could be aimed at reliably, and the thing
being asked for — raise the camera — needs a pair of hands anyway. This is why
the check runs at startup only and never from the in-session Recal. key.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

from gaze.setup_check import SetupCheckSession, SetupPhase

BACKGROUND = QColor(12, 12, 16)
TEXT = QColor(232, 232, 238)
DIM = QColor(140, 140, 152)
WARN = QColor(245, 190, 70)
TARGET_NEUTRAL = QColor(226, 226, 232)
TRACK = QColor(58, 58, 70)


class SetupCheckScreen(QWidget):
    """Fullscreen two-target camera check.

    Signals:
        finished: the :class:`~gaze.setup_check.SetupCheckResult` — passed, or
            failed and overridden by the user — or ``None`` if they cancelled.
    """

    finished = pyqtSignal(object)

    def __init__(
        self,
        session: SetupCheckSession,
        pipeline=None,
        parent: Optional[QWidget] = None,
        tick_ms: int = 16,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.pipeline = pipeline
        self.log = log if log is not None else (
            lambda message: print(f"[GazeKey] {message}"))
        self._emitted = False

        self.setWindowTitle("GazeKey — camera check")
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
                result = self.session.update(sample.features)
                if result is not None:
                    self.log(result.console_line())
                    if result.passed:
                        self._emit(result)
        self.update()

    def _emit(self, result) -> None:
        if self._emitted:
            return
        self._emitted = True
        self.finished.emit(result)

    def keyPressEvent(self, event: QKeyEvent) -> None:   # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key_Escape:
            self._emit(None)
        elif self.session.phase is SetupPhase.FAILED:
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._emit(self.session.override())      # calibrate anyway
            elif key in (Qt.Key_Space, Qt.Key_R):
                self.session.restart()

    # -------------------------------------------------------------------- copy
    def text_lines(self) -> List[Tuple[str, float, int, QColor]]:
        """``(text, y fraction, size, colour)`` — what the painter draws."""
        if self.session.phase is SetupPhase.MEASURING:
            return self._measuring_lines()
        return self._failed_lines()

    def _measuring_lines(self):
        target = self.session.current_target()
        counter = "" if target is None else f"   {target.index} / {target.total}"
        return [
            (f"Quick camera check{counter}", 0.06, 24, DIM),
            ("Look at the dot", 0.13, 34, TEXT),
            (f"About {self.session.budget_s:.0f} seconds — checking that the "
             f"camera can see your eyes move up and down", 0.19, 22, DIM),
            ("Esc to cancel", 0.93, 18, DIM),
        ]

    def _failed_lines(self):
        result = self.session.result
        headline, guidance = result.text
        return [
            ("Before we calibrate", 0.20, 24, DIM),
            (headline, 0.26, 42, WARN),
            (guidance, 0.36, 28, TEXT),
            (result.measurement_line(), 0.54, 22, DIM),
            ("Space to check again     Enter to calibrate anyway"
             "     Esc to quit", 0.66, 22, DIM),
        ]

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
        if self.session.phase is SetupPhase.MEASURING:
            self._paint_target(painter)
        for text, y, size, colour in self.text_lines():
            font = QFont("Segoe UI")
            font.setPixelSize(max(9, int(size * self.height() / 1080)))
            painter.setFont(font)
            painter.setPen(colour)
            painter.drawText(
                QRectF(self.width() * 0.12, self.height() * y,
                       self.width() * 0.76, self.height() * 0.2),
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, text,
            )

    def _to_widget(self, position) -> QPointF:
        width, height = self.session.screen_size
        return QPointF(position[0] * self.width() / max(width, 1),
                       position[1] * self.height() / max(height, 1))

    def _paint_target(self, painter: QPainter) -> None:
        target = self.session.current_target()
        if target is None:
            return
        centre = self._to_widget(target.position)
        base = self.height() * 0.042

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(TRACK, max(1.5, self.height() * 0.0022)))
        painter.drawEllipse(centre, base, base)

        if not target.settling and target.progress > 0.0:
            painter.setPen(QPen(TARGET_NEUTRAL, max(1.5, self.height() * 0.0026)))
            ring = QRectF(centre.x() - base, centre.y() - base, base * 2, base * 2)
            painter.drawArc(ring, 90 * 16, int(-360 * 16 * target.progress))

        painter.setPen(Qt.NoPen)
        painter.setBrush(TARGET_NEUTRAL)
        dot = self.height() * 0.009
        painter.drawEllipse(centre, dot, dot)

    def closeEvent(self, event) -> None:                 # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
