"""GazeKey — ask the user a question they can answer by looking.

Used for the startup question ("use the saved calibration, or recalibrate?")
and for confirming Quit. Both are built on the same machinery as the keyboard —
a :class:`~interaction.layouts.Keyboard` of two enormous keys driven by the
real :class:`~interaction.controller.DwellController` and painted by the real
:class:`~ui.keyboard_widget.KeyboardRenderer` — so the timings and the look are
identical to typing and there is nothing new to learn.

Every question is answerable without a mouse, a keyboard or a helper. That is
the point: a gaze user who cannot answer a dialog is stuck.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent
from PyQt5.QtWidgets import QWidget

from gaze.region import Region
from interaction.controller import DwellController, DwellSettings
from interaction.layouts import build_choice_board
from ui.keyboard_widget import KeyboardRenderer

BACKGROUND = QColor(12, 12, 16)
TEXT = QColor(236, 236, 242)
DIM = QColor(150, 154, 168)

CURSOR_MOVING = QColor(90, 190, 255)
CURSOR_FIXATING = QColor(90, 210, 140)


class ChoiceScreen(QWidget):
    """A fullscreen either/or question answered by dwelling on an option.

    Signals:
        chosen: emitted with the payload of the selected option, or ``None``
            if the user pressed Esc.
    """

    chosen = pyqtSignal(object)

    def __init__(
        self,
        prompt: str,
        options: Sequence[Tuple[str, str]],
        screen_size: Tuple[int, int],
        pipeline=None,
        settings: Optional[DwellSettings] = None,
        subtitle: str = "",
        parent: Optional[QWidget] = None,
        tick_ms: int = 16,
        region: Optional[Region] = None,
    ) -> None:
        super().__init__(parent)
        self.prompt = prompt
        self.subtitle = subtitle
        self.screen_size = screen_size
        self.pipeline = pipeline
        self.region = region
        self.board = build_choice_board(screen_size, options, region=region)
        self.controller = DwellController(self.board, settings or DwellSettings())
        self.renderer = KeyboardRenderer(self.board)

        self._gaze: Optional[Tuple[float, float]] = None
        self._is_fixating = False
        self._answered = False

        self.setWindowTitle("GazeKey")
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
                self._gaze = ((sample.x, sample.y)
                              if sample.has_gaze and sample.stream_valid else None)
                activation = self.controller.update(
                    sample.x, sample.y, sample.is_fixating,
                    sample.stream_valid, sample.timestamp,
                )
                if activation is not None:
                    self._answer(activation.key.payload, activation.key)
        self.update()

    def _answer(self, payload: str, key=None) -> None:
        if self._answered:
            return
        self._answered = True
        if key is not None:
            self.renderer.flash(key)
        self.chosen.emit(payload)

    def keyPressEvent(self, event: QKeyEvent) -> None:   # noqa: N802 - Qt API
        if event.key() == Qt.Key_Escape and not self._answered:
            self._answered = True
            self.chosen.emit(None)

    def text_lines(self) -> List[str]:
        lines = [self.prompt]
        if self.subtitle:
            lines.append(self.subtitle)
        lines.append("Look at an option and hold. Esc cancels.")
        return lines

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
        scale = (self.width() / max(self.screen_size[0], 1),
                 self.height() / max(self.screen_size[1], 1))
        self.renderer.render(painter, page=0,
                             focused=self.controller.focused_key,
                             progress=self.controller.progress, scale=scale)
        self._draw_gaze(painter, scale)
        self._draw_text(painter)

    def _draw_gaze(self, painter: QPainter, scale) -> None:
        if self._gaze is None:
            return
        from PyQt5.QtCore import QPointF

        colour = QColor(CURSOR_FIXATING if self._is_fixating else CURSOR_MOVING)
        centre = QPointF(self._gaze[0] * scale[0], self._gaze[1] * scale[1])
        radius = self.height() * 0.013
        halo = QColor(colour)
        halo.setAlphaF(0.25)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(centre, radius, radius)
        painter.setBrush(colour)
        painter.drawEllipse(centre, radius * 0.36, radius * 0.36)

    def text_tops(self) -> Tuple[float, float, float]:
        """Where the three lines sit, as fractions of screen height.

        Derived from the option boxes rather than fixed, because those move
        with the calibration region: prompt and subtitle above them, the hint
        below. Only the boxes have to be inside the region — text is read, not
        aimed at.
        """
        height = max(self.screen_size[1], 1)
        top = self.board.rect[1] / height
        bottom = (self.board.rect[1] + self.board.rect[3]) / height
        return (max(0.04, top - 0.20), max(0.12, top - 0.11),
                min(0.94, bottom + 0.04))

    def _draw_text(self, painter: QPainter) -> None:
        for line, top, size, colour in zip(
            self.text_lines(), self.text_tops(), (40, 26, 20),
            (TEXT, DIM, DIM),
        ):
            font = QFont("Segoe UI")
            font.setPixelSize(max(10, int(size * self.height() / 1080)))
            painter.setFont(font)
            painter.setPen(colour)
            painter.drawText(
                QRectF(self.width() * 0.08, self.height() * top,
                       self.width() * 0.84, self.height() * 0.14),
                Qt.AlignHCenter | Qt.AlignTop, line,
            )

    def closeEvent(self, event) -> None:                 # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
