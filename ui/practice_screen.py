"""GazeKey — target practice screen (post-calibration aiming drill).

Visual language matches the keyboard: a faint track ring marks the hit radius —
so what counts as a hit is never a guess — and a bright arc fills around it as
the hold accumulates, exactly like the dwell ring on a key. A hit pops with one
short expanding ring and an optional click.

The live gaze dot uses the same colours as the M3 demo (blue moving, green
fixating, amber holding through a blink), so nothing has to be re-learned.
"""
from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

from interaction.practice import PracticePhase, PracticeSession

BACKGROUND = QColor(12, 12, 16)
TEXT = QColor(232, 232, 238)
DIM = QColor(140, 140, 152)
TARGET_NEUTRAL = QColor(226, 226, 232)
TRACK = QColor(58, 58, 70)
PROGRESS = QColor(120, 200, 255)
POP = QColor(150, 235, 190)
GOOD = QColor(90, 210, 140)

CURSOR_MOVING = QColor(90, 190, 255)
CURSOR_FIXATING = QColor(90, 210, 140)
CURSOR_HELD = QColor(245, 190, 70)

#: how long the pop animation lasts
POP_S = 0.28


class PracticeScreen(QWidget):
    """Fullscreen aiming drill.

    Signals:
        finished: emitted with the :class:`~interaction.practice.PracticeStats`
            when the drill ends, or ``None`` if the user skipped it.
    """

    finished = pyqtSignal(object)

    def __init__(
        self,
        session: PracticeSession,
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
        self._is_fixating = False
        self._stream_valid = False
        self._held = False
        self._pop_at: Optional[float] = None
        self._pop_where: Optional[Tuple[float, float]] = None
        self._done_emitted = False

        self.setWindowTitle("GazeKey — practice")
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
                self._is_fixating = sample.is_fixating
                self._stream_valid = sample.stream_valid
                self._held = sample.held
                self._gaze = ((sample.x, sample.y)
                              if sample.has_gaze and sample.stream_valid else None)
                target = self.session.current_target
                event = self.session.update(sample.x, sample.y, sample.is_fixating,
                                            sample.stream_valid, sample.timestamp)
                if event == "hit":
                    self._pop(target)
        if self.session.phase is PracticePhase.DONE and not self._done_emitted:
            self._done_emitted = True
            self.finished.emit(self.session.stats)
        self.update()

    def _pop(self, where: Optional[Tuple[float, float]]) -> None:
        self._pop_at = time.monotonic()
        self._pop_where = where
        if self.sound is not None:
            self.sound()

    def keyPressEvent(self, event: QKeyEvent) -> None:   # noqa: N802 - Qt API
        if event.key() == Qt.Key_Escape:
            self.session.skip()
            if not self._done_emitted:
                self._done_emitted = True
                self.finished.emit(None)
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if self.session.phase is PracticePhase.DONE:
                self.finished.emit(self.session.stats)

    # -------------------------------------------------------------------- copy
    def text_lines(self) -> List[str]:
        """On-screen copy (the offscreen test platform has no fonts)."""
        if self.session.phase is PracticePhase.DONE:
            return (["Practice complete"] + self.session.stats.summary_lines()
                    + ["Space to continue     Esc to skip"])
        return [
            f"Look at the dot and hold   {self.session.index + 1} "
            f"/ {self.session.targets}",
            f"hit radius {self.session.hit_radius_px:.0f} px     "
            f"hold {self.session.hold_s:.1f} s",
            "Esc to skip",
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
        if self.session.phase is PracticePhase.DONE:
            self._paint_summary(painter)
        else:
            self._paint_target(painter)
            self._paint_pop(painter)
            self._paint_gaze(painter)
        self._paint_text(painter)

    def _scale(self) -> Tuple[float, float]:
        width, height = self.session.screen_size
        return self.width() / max(width, 1), self.height() / max(height, 1)

    def _to_widget(self, position) -> QPointF:
        sx, sy = self._scale()
        return QPointF(position[0] * sx, position[1] * sy)

    def _radius(self) -> float:
        sx, sy = self._scale()
        return self.session.hit_radius_px * (sx + sy) / 2.0

    def _paint_target(self, painter: QPainter) -> None:
        target = self.session.current_target
        if target is None:
            return
        centre = self._to_widget(target)
        radius = self._radius()

        # the faint ring IS the hit region — what counts is never a guess
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(TRACK, max(1.5, self.height() * 0.0022)))
        painter.drawEllipse(centre, radius, radius)

        progress = self.session.progress
        if progress > 0.0:
            painter.setPen(QPen(PROGRESS, max(3.0, self.height() * 0.005),
                                Qt.SolidLine, Qt.RoundCap))
            box = QRectF(centre.x() - radius, centre.y() - radius,
                         radius * 2, radius * 2)
            painter.drawArc(box, 90 * 16, int(-360 * 16 * progress))

        painter.setPen(Qt.NoPen)
        painter.setBrush(TARGET_NEUTRAL)
        dot = self.height() * 0.009
        painter.drawEllipse(centre, dot, dot)

    def _paint_pop(self, painter: QPainter) -> None:
        if self._pop_at is None or self._pop_where is None:
            return
        age = time.monotonic() - self._pop_at
        if age > POP_S:
            self._pop_at = None
            return
        fade = 1.0 - age / POP_S
        colour = QColor(POP)
        colour.setAlphaF(0.8 * fade)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(colour, max(2.0, self.height() * 0.004)))
        radius = self._radius() * (0.6 + 1.1 * (1.0 - fade))
        painter.drawEllipse(self._to_widget(self._pop_where), radius, radius)

    def _paint_gaze(self, painter: QPainter) -> None:
        if self._gaze is None:
            return
        colour = QColor(CURSOR_HELD if self._held else
                        (CURSOR_FIXATING if self._is_fixating else CURSOR_MOVING))
        centre = self._to_widget(self._gaze)
        radius = self.height() * 0.014
        halo = QColor(colour)
        halo.setAlphaF(0.25)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(centre, radius, radius)
        painter.setBrush(colour)
        painter.drawEllipse(centre, radius * 0.36, radius * 0.36)

    def _paint_summary(self, painter: QPainter) -> None:
        stats = self.session.stats
        if not stats.total:
            return
        # a simple bar of the hit rate, in the keyboard's progress colour
        width, height = self.width() * 0.34, max(6.0, self.height() * 0.012)
        x, y = (self.width() - width) / 2.0, self.height() * 0.46
        painter.setPen(Qt.NoPen)
        painter.setBrush(TRACK)
        painter.drawRoundedRect(QRectF(x, y, width, height),
                                height / 2, height / 2)
        painter.setBrush(GOOD)
        painter.drawRoundedRect(QRectF(x, y, width * stats.hit_rate, height),
                                height / 2, height / 2)

    def _paint_text(self, painter: QPainter) -> None:
        lines = self.text_lines()
        done = self.session.phase is PracticePhase.DONE
        # one row per line: headline, the four summary numbers, then the hint
        tops = ([0.20, 0.28, 0.33, 0.38, 0.43, 0.60] if done
                else [0.06, 0.10, 0.93])
        sizes = ([34, 26, 26, 26, 26, 22] if done else [24, 19, 18])
        colours = ([TEXT, TEXT, TEXT, TEXT, TEXT, DIM] if done
                   else [DIM, DIM, DIM])
        for line, top, size, colour in zip(lines, tops, sizes, colours):
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
