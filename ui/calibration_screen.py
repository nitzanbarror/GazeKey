"""GazeKey — fullscreen calibration UI (spec Section 5.2 / 5.3, Section 9).

Renders the four screens of a calibration session and feeds the session with
samples drained from the vision pipeline:

* **Stage A — head sweep**: a centre dot with animated arrows, a coverage
  read-out for yaw/pitch, and a repeat screen explaining *why* a sweep was
  refused when ``fit_pose_compensation()`` returns False.
* **Stage B — 9 points**: animated shrinking targets with a progress ring, a
  "4 / 9" counter, and a quiet note when a point is being repeated.
* **Validation**: the same targets on three fresh positions.
* **Results**: the measured error in pixels and the verdict, always shown.

What each screen *says* comes from :meth:`CalibrationScreen.text_lines` and
what it *draws* from :meth:`CalibrationScreen.render_into`, so both the copy
and the geometry can be checked in a headless test.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

from gaze.calibration import MARGINAL_PX, MIN_SWEEP_POSE_RANGE_DEG, PASS_PX
from gaze.calibration_session import CalibrationSession, Phase

BACKGROUND = QColor(12, 12, 16)
TEXT = QColor(232, 232, 238)
DIM = QColor(140, 140, 152)
ACCENT = QColor(90, 190, 255)
TARGET = QColor(255, 255, 255)
#: the single neutral colour used for the calibration dot and its ring
TARGET_NEUTRAL = QColor(226, 226, 232)
GOOD = QColor(90, 210, 140)
WARN = QColor(245, 190, 70)
BAD = QColor(240, 100, 100)

VERDICT_COLOUR = {"PASS": GOOD, "MARGINAL": WARN, "FAIL": BAD}

#: how long without a valid frame before the "tracking lost" badge appears
TRACKING_LOST_S = 0.6


def _residual_colour(residual_px: float) -> QColor:
    """Green / amber / red against the spec's own accuracy thresholds."""
    if not math.isfinite(residual_px):
        return DIM
    if residual_px <= PASS_PX / 2.0:
        return GOOD
    if residual_px <= MARGINAL_PX:
        return WARN
    return BAD


@dataclass
class Line:
    """One line of on-screen copy, positioned as a fraction of screen height."""

    text: str
    y: float
    size: int
    colour: QColor = field(default_factory=lambda: TEXT)
    bold: bool = False


class CalibrationScreen(QWidget):
    """Fullscreen calibration session UI.

    Signals:
        finished: emitted with the :class:`CalibrationSession` when the user
            accepts a completed session, or ``None`` if they cancelled.
    """

    finished = pyqtSignal(object)

    def __init__(
        self,
        session: CalibrationSession,
        pipeline=None,
        parent: Optional[QWidget] = None,
        tick_ms: int = 16,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.pipeline = pipeline
        self._started = time.monotonic()
        self._last_valid = time.monotonic()
        self._saw_any_frame = False
        self._diag = None
        self._diag_key = None

        self.setWindowTitle("GazeKey — calibration")
        self.setStyleSheet("background: #0c0c10;")
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(tick_ms)

    # ------------------------------------------------------------------ input
    def _tick(self) -> None:
        """Drain the pipeline on the Qt thread and repaint."""
        if self.pipeline is not None:
            for sample in self.pipeline.drain():
                self._saw_any_frame = True
                if sample.valid:
                    self._last_valid = time.monotonic()
                self.session.update(sample.features)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key_Escape:
            self.finished.emit(None)
        elif key == Qt.Key_R:
            self.session.restart()
        elif key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._advance()

    def _advance(self) -> None:
        if self.session.phase is Phase.SWEEP_FAILED:
            self.session.retry_sweep()
        elif self.session.phase is Phase.RESULTS:
            if self.session.should_save:
                self.finished.emit(self.session)
            else:
                self.session.restart()

    # -------------------------------------------------------------------- copy
    def text_lines(self) -> List[Line]:
        """Every line of copy for the current phase, top to bottom."""
        phase = self.session.phase
        if phase is Phase.SWEEP:
            return self._sweep_lines()
        if phase is Phase.SWEEP_FAILED:
            return self._sweep_failed_lines()
        if phase in (Phase.POINTS, Phase.VALIDATION):
            return self._target_lines()
        return self._results_lines()

    def _sweep_lines(self) -> List[Line]:
        yaw_range, pitch_range = self.session.sweep_pose_range
        return [
            Line("Step 1 of 2 — head sweep", 0.10, 26, DIM, bold=True),
            Line("Keep your eyes on the dot and slowly turn your head "
                 "left and right, then up and down.", 0.16, 34),
            Line(f"left–right {self._coverage(yaw_range)}     "
                 f"up–down {self._coverage(pitch_range)}", 0.82, 22, DIM),
            Line("Esc to cancel", 0.92, 18, DIM),
        ]

    def _sweep_failed_lines(self) -> List[Line]:
        headline, guidance = self.session.sweep_failure_message
        return [
            Line("Let's try that again", 0.24, 30, WARN, bold=True),
            Line(headline, 0.32, 40, TEXT, bold=True),
            Line(guidance, 0.42, 30),
            Line("Space to repeat the head sweep     Esc to cancel", 0.62, 24, DIM),
        ]

    def _target_lines(self) -> List[Line]:
        target = self.session.current_target()
        if target is None:
            return []
        if self.session.refit_pass:
            label = f"Improving accuracy   {target.index} / {target.total}"
        elif target.stage is Phase.VALIDATION:
            label = f"Checking accuracy   {target.index} / {target.total}"
        elif self.session.fixed_head:
            label = f"Look at the dot   {target.index} / {target.total}"
        else:
            label = (f"Step 2 of 2 — look at the dot   "
                     f"{target.index} / {target.total}")
        # No settle/measure text: the ring around the dot is the only cue, so
        # nothing near the fixation point changes while the user is staring.
        lines = [Line(label, 0.06, 24, DIM, bold=True)]
        if target.retry:
            lines.append(Line("repeating this point", 0.10, 20, WARN))
        lines.append(Line("Esc to cancel", 0.93, 18, DIM))
        return lines

    def _results_lines(self) -> List[Line]:
        headline, guidance = self.session.verdict_text
        colour = VERDICT_COLOUR.get(self.session.verdict, BAD)
        error = self.session.error_px
        diagnostics = self.diagnostics()
        measured = "not measured" if not math.isfinite(error) else f"{error:.0f} px"
        if self.session.should_save:
            actions = "Space to continue     R to calibrate again     Esc to quit"
        else:
            actions = "Space or R to calibrate again     Esc to quit"
        lines = [
            Line("Calibration result", 0.035, 24, DIM, bold=True),
            Line(f"Accuracy: {measured} — {headline}", 0.075, 42, colour, bold=True),
            Line(guidance, 0.145, 24),
            Line(diagnostics.feature_line(), 0.670, 19, DIM),
            Line(diagnostics.sweep_line(), 0.702, 19, DIM),
            Line(diagnostics.region_line(), 0.734, 19, DIM),
        ]
        if math.isfinite(error):
            lines.append(Line(f"pass ≤ {PASS_PX:.0f} px      "
                              f"usable ≤ {MARGINAL_PX:.0f} px", 0.766, 19, DIM))
        lines.append(Line(diagnostics.likely_cause, 0.800, 23, ACCENT))
        lines.append(Line(actions, 0.905, 21, DIM))
        return lines

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self):
        """Session diagnostics, recomputed only when the verdict changes."""
        key = (self.session.verdict, self.session.error_px, self.session.phase)
        if self._diag_key != key:
            self._diag = self.session.diagnostics()
            self._diag_key = key
        return self._diag

    def diagnostics_rect(self) -> QRectF:
        """The map area, matching the screen's aspect ratio."""
        top, bottom = self.height() * 0.215, self.height() * 0.655
        available_h = bottom - top
        available_w = self.width() * 0.82
        screen_w, screen_h = self.session.screen_size
        aspect = max(screen_w, 1) / max(screen_h, 1)
        width = min(available_w, available_h * aspect)
        height = width / aspect
        return QRectF((self.width() - width) / 2.0,
                      top + (available_h - height) / 2.0, width, height)

    def _to_map(self, position) -> QPointF:
        rect = self.diagnostics_rect()
        screen_w, screen_h = self.session.screen_size
        return QPointF(rect.x() + position[0] / max(screen_w, 1) * rect.width(),
                       rect.y() + position[1] / max(screen_h, 1) * rect.height())

    def diagnostic_labels(self) -> List[Tuple[str, QPointF]]:
        """Per-point map labels: residual and kept/collected sample counts."""
        diagnostics = self.diagnostics()
        labels: List[Tuple[str, QPointF]] = []
        offset = self.height() * 0.028
        for point in diagnostics.points:
            residual = ("—" if not math.isfinite(point.residual_px)
                        else f"{point.residual_px:.0f} px")
            centre = self._to_map(point.target)
            labels.append((f"{residual}  {point.kept}/{point.collected}",
                           QPointF(centre.x(), centre.y() + offset)))
        for point in diagnostics.validation:
            if point.prediction is None:
                continue
            centre = self._to_map(point.prediction)
            labels.append((f"{point.residual_px:.0f} px",
                           QPointF(centre.x(), centre.y() + offset)))
        return labels

    def badge_text(self) -> Optional[str]:
        """The tracking indicator, or ``None`` when tracking is healthy."""
        if self.pipeline is None:
            return None
        if not self._saw_any_frame:
            return "waiting for the camera…"
        if time.monotonic() - self._last_valid < TRACKING_LOST_S:
            return None
        return "tracking lost — face the camera"

    @staticmethod
    def _coverage(degrees: float) -> str:
        target = MIN_SWEEP_POSE_RANGE_DEG * 2.0
        return "OK" if degrees >= target else f"{degrees:4.1f}° / {target:.0f}°"

    # ---------------------------------------------------------------- painting
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        try:
            self.render_into(painter)
        finally:
            painter.end()

    def render_into(self, painter: QPainter) -> None:
        """Draw the current phase. Separated out so tests can render offscreen."""
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)

        phase = self.session.phase
        if phase is Phase.SWEEP:
            self._paint_sweep(painter)
        elif phase in (Phase.POINTS, Phase.VALIDATION):
            self._paint_target(painter)
        elif phase is Phase.RESULTS:
            self._paint_diagnostics(painter)

        for line in self.text_lines():
            self._paint_line(painter, line)

        badge = self.badge_text()
        if badge:
            painter.setFont(self._font(20))
            painter.setPen(WARN)
            painter.drawText(
                QRectF(0, self.height() * 0.02, self.width() * 0.97,
                       self.height() * 0.06),
                Qt.AlignRight | Qt.AlignTop, badge,
            )

    # ------------------------------------------------------------- primitives
    def _scale(self) -> Tuple[float, float]:
        session_w, session_h = self.session.screen_size
        return self.width() / max(session_w, 1), self.height() / max(session_h, 1)

    def _to_widget(self, position: Tuple[int, int]) -> QPointF:
        sx, sy = self._scale()
        return QPointF(position[0] * sx, position[1] * sy)

    def _font(self, size: int, bold: bool = False) -> QFont:
        font = QFont("Segoe UI", max(8, int(size * self.height() / 1080)))
        font.setBold(bold)
        return font

    def _paint_line(self, painter: QPainter, line: Line) -> None:
        painter.setFont(self._font(line.size, line.bold))
        painter.setPen(line.colour)
        rect = QRectF(self.width() * 0.1, self.height() * line.y,
                      self.width() * 0.8, self.height() * 0.2)
        painter.drawText(
            rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, line.text
        )

    def _progress_bar(self, painter: QPainter, fraction: float, y_ratio: float,
                      colour: QColor = ACCENT, width_ratio: float = 0.34) -> None:
        width = self.width() * width_ratio
        height = max(4.0, self.height() * 0.008)
        x = (self.width() - width) / 2.0
        y = self.height() * y_ratio
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(60, 60, 70))
        painter.drawRoundedRect(QRectF(x, y, width, height), height / 2, height / 2)
        painter.setBrush(colour)
        painter.drawRoundedRect(
            QRectF(x, y, width * max(0.0, min(1.0, fraction)), height),
            height / 2, height / 2,
        )

    def _chevron(self, painter: QPainter, centre: QPointF, size: float,
                 angle_deg: float, alpha: float) -> None:
        """A single animated arrow head pointing along ``angle_deg``."""
        colour = QColor(ACCENT)
        colour.setAlphaF(max(0.0, min(1.0, alpha)))
        pen = QPen(colour, max(2.0, size * 0.16))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rad = math.radians(angle_deg)
        forward = QPointF(math.cos(rad), math.sin(rad))
        side = QPointF(-forward.y(), forward.x())
        tip = centre + forward * size
        painter.drawPolyline(QPolygonF([
            tip - forward * size + side * size * 0.75,
            tip,
            tip - forward * size - side * size * 0.75,
        ]))

    # ------------------------------------------------------------- stage A art
    def _paint_sweep(self, painter: QPainter) -> None:
        now = time.monotonic() - self._started
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)

        # animated arrows: horizontal for a few seconds, then vertical, looping
        span = self.height() * 0.16
        size = self.height() * 0.030
        pulse = 0.45 + 0.55 * abs(math.sin(now * 1.6))
        if (now % 6.0) < 3.0:
            self._chevron(painter, centre + QPointF(-span, 0), size, 180, pulse)
            self._chevron(painter, centre + QPointF(span, 0), size, 0, pulse)
        else:
            self._chevron(painter, centre + QPointF(0, -span * 0.8), size, 270, pulse)
            self._chevron(painter, centre + QPointF(0, span * 0.8), size, 90, pulse)

        radius = self.height() * 0.011
        painter.setPen(Qt.NoPen)
        painter.setBrush(TARGET)
        painter.drawEllipse(centre, radius, radius)
        ring = QColor(ACCENT)
        ring.setAlphaF(0.35 + 0.35 * abs(math.sin(now * 2.2)))
        painter.setPen(QPen(ring, max(1.5, self.height() * 0.0025)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius * 2.6, radius * 2.6)

        self._progress_bar(painter, self.session.sweep_progress, 0.78)

    # ------------------------------------------------------------- stage B art
    def _paint_target(self, painter: QPainter) -> None:
        target = self.session.current_target()
        if target is None:
            return
        centre = self._to_widget(target.position)
        base = self.height() * 0.042

        # Deliberately still: a steady neutral dot that never pulses, never
        # changes colour and never flashes, because anything moving near the
        # fixation point makes it harder to hold a steady stare. The only
        # feedback is a thin ring quietly filling around it — empty while
        # settling, filling while the samples are actually being measured.
        track = QColor(58, 58, 70)
        painter.setPen(QPen(track, max(1.5, self.height() * 0.0022)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, base, base)

        if not target.settling and target.progress > 0.0:
            painter.setPen(QPen(TARGET_NEUTRAL, max(1.5, self.height() * 0.0026)))
            ring = QRectF(centre.x() - base, centre.y() - base, base * 2, base * 2)
            painter.drawArc(ring, 90 * 16, int(-360 * 16 * target.progress))

        painter.setPen(Qt.NoPen)
        painter.setBrush(TARGET_NEUTRAL)
        dot = self.height() * 0.009
        painter.drawEllipse(centre, dot, dot)

    # ---------------------------------------------------------- results art
    def _paint_diagnostics(self, painter: QPainter) -> None:
        """A miniature of the screen: fit residuals and validation arrows."""
        diagnostics = self.diagnostics()
        rect = self.diagnostics_rect()

        painter.setPen(QPen(QColor(48, 48, 60), max(1.0, self.height() * 0.0015)))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        # The calibrated region, drawn inside the screen outline, so it is
        # obvious at a glance which part of the display these numbers cover.
        region = self.session.region
        screen_w, screen_h = self.session.screen_size
        if region is not None and (region.w < screen_w or region.h < screen_h):
            pen = QPen(ACCENT, max(1.0, self.height() * 0.0018))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            top_left = self._to_map((region.x, region.y))
            painter.drawRect(QRectF(
                top_left.x(), top_left.y(),
                region.w / max(screen_w, 1) * rect.width(),
                region.h / max(screen_h, 1) * rect.height()))

        marker = max(3.0, self.height() * 0.011)
        for point in diagnostics.points:
            centre = self._to_map(point.target)
            painter.setPen(Qt.NoPen)
            painter.setBrush(_residual_colour(point.residual_px))
            painter.drawEllipse(centre, marker, marker)
            if point.retries:
                painter.setPen(QPen(WARN, max(1.5, self.height() * 0.002)))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(centre, marker * 1.9, marker * 1.9)

        for point in diagnostics.validation:
            centre = self._to_map(point.target)
            painter.setPen(QPen(TEXT, max(1.5, self.height() * 0.002)))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(centre.x() - marker, centre.y() - marker,
                                    marker * 2, marker * 2))
            if point.prediction is None:
                continue
            self._arrow(painter, centre, self._to_map(point.prediction),
                        _residual_colour(point.residual_px))

        painter.setFont(self._font(16))
        painter.setPen(DIM)
        for text, position in self.diagnostic_labels():
            width = self.width() * 0.18
            painter.drawText(
                QRectF(position.x() - width / 2, position.y(), width,
                       self.height() * 0.05),
                Qt.AlignHCenter | Qt.AlignTop, text,
            )

    def _arrow(self, painter: QPainter, start: QPointF, end: QPointF,
               colour: QColor) -> None:
        """Line from a validation target to where the model predicted it."""
        painter.setPen(QPen(colour, max(2.0, self.height() * 0.003)))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(start, end)

        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        head = max(5.0, self.height() * 0.012)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        if length < 1e-6:
            painter.drawEllipse(end, head * 0.4, head * 0.4)
            return
        ux, uy = dx / length, dy / length
        painter.drawPolygon(QPolygonF([
            end,
            QPointF(end.x() - head * ux + head * 0.45 * -uy,
                    end.y() - head * uy + head * 0.45 * ux),
            QPointF(end.x() - head * ux - head * 0.45 * -uy,
                    end.y() - head * uy - head * 0.45 * ux),
        ]))

    # ------------------------------------------------------------- lifecycle
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
