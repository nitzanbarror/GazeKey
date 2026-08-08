"""GazeKey — the always-on-top keyboard overlay (spec Sections 8 and 9).

A frameless, translucent, **non-focusable and click-through** window covering
the screen: the keyboard is docked along the bottom, the optional gaze cursor
and the status icons float above whatever application is really focused.

The window flags matter more than they look. Without
``WindowDoesNotAcceptFocus`` + ``WA_ShowWithoutActivating`` the overlay would
take keyboard focus and every injected keystroke would land in GazeKey itself
instead of the app the user is typing into; without
``WA_TransparentForMouseEvents`` it would swallow every mouse click on the
desktop underneath.

Each timer tick drains the pipeline, feeds the dwell controller, and injects
whatever fired.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import QWidget

from PyQt5.QtGui import QImage

from gaze.drift import DriftMonitor
from interaction.controller import Activation, DwellController
from interaction.layouts import Keyboard
from interaction.prediction import WordBuffer, WordPredictor
from ui.keyboard_widget import KeyboardRenderer

CURSOR_MOVING = QColor(90, 190, 255)
CURSOR_FIXATING = QColor(90, 210, 140)
CURSOR_HELD = QColor(245, 190, 70)

STATUS_TEXT = QColor(226, 226, 234)
STATUS_WARN = QColor(245, 190, 70)
STATUS_BAD = QColor(240, 100, 100)
STATUS_BACKING = QColor(18, 18, 24, 200)

#: the status panel never grows past this fraction of the screen width
STATUS_MAX_WIDTH_RATIO = 0.4


class KeyboardOverlay(QWidget):
    """The gaze keyboard, floating over the focused application.

    Signals:
        recalibrate_requested: the RECALIBRATE key completed its extended dwell.
        touchup_requested: the FIX AIM key completed its extended dwell.
        closed: the overlay is going away.
    """

    recalibrate_requested = pyqtSignal()
    touchup_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(
        self,
        keyboard: Keyboard,
        controller: DwellController,
        pipeline=None,
        injector=None,
        screen_size: Optional[Tuple[int, int]] = None,
        show_cursor: bool = True,
        language: str = "en",
        predictor: Optional[WordPredictor] = None,
        show_webcam: bool = True,
        drift: Optional[DriftMonitor] = None,
        diagnostics=None,
        tick_ms: int = 16,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.keyboard = keyboard
        self.controller = controller
        self.pipeline = pipeline
        self.injector = injector
        self.screen_size = screen_size or (int(keyboard.rect[2]),
                                           int(keyboard.rect[1] + keyboard.rect[3]))
        self.show_cursor = show_cursor
        self.renderer = KeyboardRenderer(keyboard, language)
        self.language = language
        self.predictor = predictor
        self.show_webcam = show_webcam and keyboard.webcam_rect is not None
        if self.show_webcam and pipeline is not None:
            pipeline.preview_enabled = True

        self.typed_text = ""
        self.word = WordBuffer()
        self.suggestions: List[str] = []
        self.drift = drift if drift is not None else DriftMonitor()
        #: optional TypingDiagnostics (--debug-typing); observational only
        self.diagnostics = diagnostics

        self._notice = ""
        self._notice_until = 0.0
        self._gaze: Optional[Tuple[float, float]] = None
        self._is_fixating = False
        self._stream_valid = False
        self._held = False
        self._saw_frame = False
        self.history: List[str] = []

        self.setWindowTitle("GazeKey")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

        self._tick_ms = tick_ms
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(tick_ms)

    # -------------------------------------------------------------- suspending
    def suspend(self) -> None:
        """Step aside while another fullscreen screen owns the gaze.

        Hiding is not enough: a hidden widget's timer keeps firing, so the
        overlay would go on draining the *same* pipeline queue as the screen
        that replaced it — halving that screen's samples and, worse, still
        completing dwells and injecting keystrokes behind it.
        """
        self._timer.stop()
        self.controller.reset()
        self._gaze = None
        self.hide()

    def resume(self) -> None:
        """Come back, discarding whatever gaze piled up while away."""
        if self.pipeline is not None:
            self.pipeline.drain()
        self.controller.reset()
        self._gaze = None
        self.show()
        self._timer.start(self._tick_ms)

    @property
    def consuming_gaze(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------- loop
    def tick(self) -> None:
        """Drain the pipeline, advance the dwell, inject what fired."""
        last_timestamp = None
        if self.pipeline is not None:
            for sample in self.pipeline.drain():
                self._saw_frame = True
                last_timestamp = sample.timestamp
                if self.diagnostics is not None:
                    self.diagnostics.record(sample)
                self._is_fixating = sample.is_fixating
                self._stream_valid = sample.stream_valid
                self._held = sample.held
                self._gaze = ((sample.x, sample.y)
                              if sample.has_gaze and sample.stream_valid else None)
                # Held samples repeat the last position through a blink, so they
                # would bias the drift centroid towards wherever the eye closed.
                if self._gaze is not None and not sample.held:
                    self.drift.record_sample(sample.x, sample.y,
                                             sample.timestamp)
                activation = self.controller.update(
                    sample.x, sample.y, sample.is_fixating,
                    sample.stream_valid, sample.timestamp,
                )
                if activation is not None:
                    self.dispatch(activation)
        if self.diagnostics is not None and last_timestamp is not None:
            self.diagnostics.tick(last_timestamp)
        self.update()

    def dispatch(self, activation: Activation) -> None:
        """Act on an activated key: inject, or handle it as an app command."""
        key = activation.key
        self.renderer.flash(key)

        if activation.suppressed:
            self.history.append(f"({key.label_en} ignored - paused)")
            return

        # Weak ground truth for the drift monitor: the user was looking at this
        # key when it fired (spec 5.5).
        self.drift.record_activation(key, activation.timestamp)

        action = key.action
        if action in ("page", "symbols"):
            self.controller.next_page()
        elif action == "pause":
            self.controller.toggle_pause()
        elif action == "lang":
            self.language = "he" if self.language == "en" else "en"
            self.renderer.language = self.language
        elif action == "recalibrate":
            self.recalibrate_requested.emit()
        elif action == "touchup":
            self.touchup_requested.emit()
        elif action == "quit":
            self.quit_requested.emit()
        elif action == "suggestion":
            self._accept_suggestion(int(key.payload or 0))
        elif self.injector is not None:
            shifted = bool(getattr(self.injector, "shift_latched", False))
            self.injector.send(action, key.payload)
            self._track_typing(action, key.payload, shifted)
        else:
            self.history.append(f"({key.label_en} - no injector)")
            return

        self.history.append(key.label_en or action)
        del self.history[:-12]
        self._refresh_suggestions()

    # ------------------------------------------------------------ typed text
    def _track_typing(self, action: str, payload: str,
                      shifted: bool = False) -> None:
        """Mirror what was injected into the preview line and the word buffer."""
        if action == "char":
            typed = payload.upper() if shifted else payload.lower()
            self.typed_text += typed
            self.word.add(typed)
        elif action == "space":
            if self.word.text and self.predictor is not None:
                self.predictor.record(self.word.text)
            self.typed_text += " "
            self.word.clear()
        elif action == "enter":
            self.typed_text = ""
            self.word.clear()
        elif action == "backspace":
            self.typed_text = self.typed_text[:-1]
            self.word.backspace()

    def _accept_suggestion(self, slot: int) -> None:
        """Type the rest of the suggested word plus a space."""
        if slot >= len(self.suggestions) or self.injector is None:
            return
        word = self.suggestions[slot]
        completion = word[len(self.word.text):] if \
            word.lower().startswith(self.word.text.lower()) else word
        for char in completion:
            self.injector.send("char", char)
        self.injector.send("space")
        self.typed_text += completion + " "
        if self.predictor is not None:
            self.predictor.record(word)
        self.word.clear()

    def _refresh_suggestions(self) -> None:
        if self.predictor is None:
            self.suggestions = []
            return
        slots = len(self.keyboard.suggestion_keys) or 4
        self.suggestions = self.predictor.suggest(self.word.text, limit=slots)

    # -------------------------------------------------------------- session state
    def session_state(self) -> dict:
        """What has to survive a recalibration: the sentence in progress."""
        return {
            "typed_text": self.typed_text,
            "word": self.word.text,
            "language": self.language,
            "page": self.controller.page,
            "paused": self.controller.paused,
        }

    def restore_session(self, state: Optional[dict]) -> None:
        """Put a saved :meth:`session_state` back after recalibrating.

        The board itself may have been rebuilt at a different key size, so the
        state is carried, not the widget — but from the user's side the sentence
        they were typing is simply still there.
        """
        if not state:
            return
        self.typed_text = str(state.get("typed_text", ""))
        self.word.text = str(state.get("word", ""))
        self.language = str(state.get("language", self.language))
        self.renderer.language = self.language
        self.controller.set_page(int(state.get("page", 0)))
        self.controller.paused = bool(state.get("paused", False))
        self._refresh_suggestions()

    # ---------------------------------------------------------------- drawing
    def paintEvent(self, event: QPaintEvent) -> None:   # noqa: N802 - Qt API
        painter = QPainter(self)
        try:
            self.render_into(painter)
        finally:
            painter.end()

    def render_into(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.transparent)

        scale = self._scale()
        self.renderer.render(
            painter,
            page=self.controller.page,
            focused=self.controller.focused_key,
            progress=self.controller.progress,
            paused=self.controller.paused,
            shift_latched=bool(getattr(self.injector, "shift_latched", False)),
            suggestions=self.suggestions,
            typed_text=self.typed_text,
            scale=scale,
        )
        if self.show_webcam:
            self.renderer.draw_webcam(painter, self.webcam_image(), scale)
        self._draw_cursor(painter, scale)
        self._draw_status(painter)

    def webcam_image(self) -> Optional[QImage]:
        """The pipeline's latest self-view thumbnail as a QImage."""
        frame = getattr(self.pipeline, "preview_frame", lambda: None)()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        contiguous = frame if frame.flags["C_CONTIGUOUS"] else frame.copy()
        image = QImage(contiguous.data, width, height, 3 * width,
                       QImage.Format_RGB888)
        return image.copy()          # detach from the numpy buffer

    def _scale(self) -> Tuple[float, float]:
        width, height = self.screen_size
        return self.width() / max(width, 1), self.height() / max(height, 1)

    def _draw_cursor(self, painter: QPainter, scale) -> None:
        if not self.show_cursor or self._gaze is None:
            return
        colour = QColor(CURSOR_HELD if self._held else
                        (CURSOR_FIXATING if self._is_fixating else CURSOR_MOVING))
        centre = QPointF(self._gaze[0] * scale[0], self._gaze[1] * scale[1])
        radius = self.height() * 0.014
        halo = QColor(colour)
        halo.setAlphaF(0.25)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(centre, radius, radius)
        painter.setBrush(colour)
        painter.drawEllipse(centre, radius * 0.36, radius * 0.36)

    # ------------------------------------------------------------ drift / notes
    def notice(self, text: str, seconds: float = 4.0,
               now: Optional[float] = None) -> None:
        """Show a short message in the corner, then let it fade by itself.

        Used for touch-up and recalibration outcomes: the user gets the result
        without a dialog to dismiss, and typing is never blocked.
        """
        self._notice = text
        self._notice_until = (time.monotonic() if now is None else now) + seconds

    def _active_notice(self) -> str:
        if self._notice and time.monotonic() < self._notice_until:
            return self._notice
        self._notice = ""
        return ""

    def drift_line(self) -> Optional[str]:
        """The quiet drift indicator, or ``None`` when the aim looks fine.

        It only ever *reports*. Nothing here interrupts typing, opens a screen
        or recalibrates on its own — the correction is the user's to start.
        """
        if not self.drift.drifting:
            return None
        state = self.drift.state()
        fix = ("look at Fix aim to correct it"
               if self.keyboard.has_action("touchup")
               else "look at Recal. to correct it")
        return (f"LOW ACCURACY: aim has drifted about "
                f"{state.offset_px:.0f} px - {fix}")

    def status_lines(self) -> List[str]:
        """Corner status, exposed so headless tests can read it."""
        lines: List[str] = []
        if self.controller.paused:
            lines.append("PAUSED - dwell still works, nothing is typed")
        if not self._saw_frame:
            lines.append("waiting for the camera...")
        elif not self._stream_valid:
            lines.append("tracking lost")
        if self.injector is None:
            lines.append("no keystroke injection")
        if not self.keyboard.meets_nfr2:
            # Both axes, always: the one that binds is usually the height, and
            # a warning that only quotes the width sends the user chasing the
            # wrong fix (see interaction/layouts.py). Named "keys below spec"
            # rather than "low accuracy" because spec Section 9 reserves that
            # phrase for the drift indicator below, and the two have completely
            # different remedies: a bigger keyboard vs a corrected aim.
            lines.append(f"keys below spec: {self.keyboard.nfr2_line()}")
        drift = self.drift_line()
        if drift is not None:
            lines.append(drift)
        notice = self._active_notice()
        if notice:
            lines.append(notice)
        if self.keyboard.pages > 1:
            lines.append(f"page {self.controller.page + 1} of {self.keyboard.pages}")
        return lines

    def _draw_status(self, painter: QPainter) -> None:
        lines = self.status_lines()
        if not lines:
            return
        font = QFont("Segoe UI")
        font.setPixelSize(max(11, int(self.height() * 0.018)))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        # Size the panel to its text and cap it: a wide slab would sit on top of
        # the application the user is actually typing into.
        limit = self.width() * STATUS_MAX_WIDTH_RATIO
        lines = [metrics.elidedText(line, Qt.ElideRight, int(limit))
                 for line in lines]
        text_w = min(limit, max((metrics.horizontalAdvance(line)
                                 for line in lines), default=0))
        text_h = metrics.height() * len(lines)
        box = QRectF(self.width() * 0.012, self.height() * 0.012,
                     float(text_w), float(text_h))
        painter.setPen(Qt.NoPen)
        painter.setBrush(STATUS_BACKING)
        if text_w > 0:
            painter.drawRoundedRect(box.adjusted(-8, -5, 8, 5), 6, 6)
        colour = STATUS_BAD if not self._stream_valid and self._saw_frame else (
            STATUS_WARN if self.controller.paused or self.drift.drifting
            else STATUS_TEXT)
        painter.setPen(colour)
        painter.drawText(box, Qt.AlignLeft | Qt.AlignTop, "\n".join(lines))

    # ------------------------------------------------------------- lifecycle
    def keyPressEvent(self, event: QKeyEvent) -> None:   # noqa: N802 - Qt API
        # The overlay never holds focus, so this only fires if something else
        # forwards a key to it; Esc is kept as an escape hatch during testing.
        if event.key() == Qt.Key_Escape:
            self.closed.emit()
            self.close()

    def closeEvent(self, event) -> None:                 # noqa: N802 - Qt API
        self._timer.stop()
        super().closeEvent(event)
