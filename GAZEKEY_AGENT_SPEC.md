# GazeKey — Implementation Spec

**This file is the single source of truth.** `SPEC_AMENDMENTS.md` has been
merged into it and deleted; there is no second document to reconcile against.

You are building **GazeKey**: a desktop virtual keyboard controlled entirely by
eye gaze, using a standard webcam. Target users have motor disabilities (e.g.
ALS). The app estimates where the user is looking, shows an on-screen keyboard,
selects keys by dwell-time (staring for ~1 second), and injects real OS
keystrokes into whatever application is focused (browser, WhatsApp Web, etc.).

**The #1 priority of this rebuild is a calibration system that actually works.**
The previous implementation used raw iris pixel positions with linear
interpolation over 5 points and no head-pose compensation — it drifted within a
minute and was unusable. Follow the calibration design in Section 5 exactly; do
not simplify it.

**Status (2026-08-08): Milestones 1–5 are complete, plus word prediction pulled
forward from M6, region-scoped calibration (Section 5.6), point-level
calibration repair (Section 5.3), the typing-stability fixes of NFR-7 (sticky
focus and a dwell that survives a steal) and the setup check of Section 5.1c.
660 tests passing.**
The NFR-7 fixes are implemented and tested but **not yet measured on hardware**
— the after-measurement with `--debug-typing` is the next thing to do.
Remaining scope: M6 (Hebrew + RTL only), then the two end-to-end scenarios in
Section 11.

> **Where this document and the code disagree, what to do depends on the kind
> of disagreement** (CLAUDE.md rule 9):
>
> * **descriptive detail** — sizes, geometry, defaults, timings, structure,
>   wording, counts: the **code wins**, and this document is corrected to match.
>   Every number here was read out of the implementation, not assumed.
> * **product decision** — flows and screens, default modes, NFR thresholds and
>   formulas, which keys exist, privacy rules: **this document wins until the
>   owner approves a change.** Being implemented does not make a product
>   decision correct. Raise the conflict; change neither side until told.
> * **VERIFIED CORE** sections: those files must not change either way.

---

## 1. Tech stack (fixed — do not substitute)

- Python 3.10+ (developed and tested on 3.13)
- `opencv-python` — webcam capture
- `mediapipe` — Face Mesh with iris landmarks (478 landmarks)
- `numpy` — all math (calibration fitting via least squares; do NOT add sklearn)
- `PyQt5` — overlay and fullscreen UI. NOT tkinter.
- `pynput` — OS keystroke injection and the global quit hotkey

**No `pyenchant`, ever.** The original spec listed it as an optional M6
dependency for word prediction. It is a *spell checker*: it can say whether a
word exists but not which of several candidates is more likely, so it cannot
rank suggestions — and its Windows builds need external enchant DLLs that
frequently fail to resolve. Word prediction ships instead on a small bundled
frequency list plus per-user counts (Section 8.4). Do not reintroduce it.

Target OS: Windows 10/11 primarily; keep code OS-agnostic where cheap.

## 2. Repository structure

The package root **is** the repository root (flat layout). The original spec
nested everything under `gazekey/`, but the verified core already lives at
`gaze/` and `tests/test_calibration_pipeline.py` imports it as
`gaze.calibration`; nothing was moved.

```
main.py                  the application: one command, one flow
config.py                load/save JSON config with defaults
calibrate.py             dev tool: calibration session + gaze-dot demo
debug_vision.py          dev tool: M1 live debug view
vision/
  camera.py              CameraSource: threaded capture, reconnect handling
  face_tracker.py        FaceTracker: mediapipe -> FrameFeatures (both APIs)
  head_pose.py           solvePnP on a rigid 6-point subset -> yaw/pitch/roll
  model_assets.py        one-time model bundle download + checksum
  pipeline.py            GazePipeline: worker thread -> queue of GazeSample
gaze/                    VERIFIED CORE plus what wraps it
  features.py            FrameFeatures + extraction math      [VERIFIED CORE]
  calibration.py         CalibrationModel: fit/predict/save/load [VERIFIED CORE]
  smoothing.py           OneEuroFilter, FixationDetector       [VERIFIED CORE]
  calibration_session.py session sequencing over that core (no math of its own)
  drift.py               DriftMonitor + TouchUpSession (M5)
  region.py              the calibration region + its persistence
  setup_check.py         the 5 s pre-calibration camera check (5.1c)
interaction/
  controller.py          dwell state machine, hysteresis, hit-testing
  layouts.py             key geometry, NFR-2 sizing, layout selection
  injector.py            pynput keystroke injection
  practice.py            post-calibration aiming drill
  diagnostics.py         --debug-typing: focus churn, dwell losses, jitter
  prediction.py          offline word suggestions + personal counts
  hotkey.py              global Ctrl+Alt+Q quit listener
ui/
  overlay.py             frameless always-on-top click-through keyboard window
  keyboard_widget.py     key rendering, dwell feedback, suggestions, self-view
  calibration_screen.py  fullscreen calibration & validation UI
  setup_check_screen.py  the two-target camera check before it (5.1c)
  practice_screen.py     the aiming drill
  touchup_screen.py      the single-target "Fix aim" screen (M5)
  choice_screen.py       gaze-answerable questions (the Quit confirmation)
  gaze_demo.py           M2 checkpoint gaze dot
  exit_button.py         the clickable X, in its own clickable window
tests/
requirements.txt
README.md
```

## 3. Threading model (important)

- **Worker thread**: camera capture → mediapipe → feature extraction → gaze
  prediction → smoothing. Results are pushed onto a thread-safe queue.
- **Main (Qt) thread**: all UI, interaction controller, keystroke injection.
  Every screen drains the queue from a 16 ms `QTimer`.
- The queue option is used rather than Qt signals so `vision/pipeline.py` has
  no Qt import at all and the whole gaze policy is testable headless.
- The Qt UI must never block on vision processing. Target ≥ 20 FPS end-to-end
  on CPU (NFR-1).

**Only one screen may consume the queue at a time.** Two widgets draining the
same queue each get roughly half the samples, and a hidden-but-still-ticking
keyboard will keep completing dwells and injecting keystrokes behind whatever
replaced it. `KeyboardOverlay.suspend()` / `.resume()` exist for this: suspend
stops the timer and resets the dwell, resume discards the backlog first.

## 4. Per-frame feature extraction (`gaze/features.py` — VERIFIED CORE)

For each frame, produce:

```python
@dataclass
class FrameFeatures:
    valid: bool          # face found AND not blinking
    hx: float            # horizontal iris ratio, averaged over both eyes
    hy: float            # vertical iris ratio, averaged over both eyes
    yaw: float           # head pose (degrees)
    pitch: float
    roll: float
    timestamp: float
```

Rules:
- **Iris ratios, not pixels.** Per eye:
  `hx = (iris_cx - inner_corner_x) / (outer_corner_x - inner_corner_x)`,
  `hy = (iris_cy - upper_lid_y) / (lower_lid_y - upper_lid_y)`. Mediapipe iris
  landmarks (right eye 468–472, left 473–477; corners 33/133 and 362/263; lids
  159/145 and 386/374). Average both eyes, **both measured along the image
  axis** (the one approved historical fix to this file).
- **Blink gate:** eye aspect ratio (EAR) < 0.18 for either eye → `valid=False`.
- **Head pose:** `cv2.solvePnP` (SQPnP + LM refinement) on landmarks
  1/152/33/263/61/291 against a canonical 3D face model, `f ≈ image width`.
  Sign conventions — yaw > 0 turning to the user's right, pitch > 0 looking up,
  roll > 0 tilting toward the user's left shoulder — are pinned by tests,
  because changing them would invalidate every saved calibration.
- No face detected → `valid=False`, all values NaN.
- Features and head pose are always computed on the **raw, un-mirrored** frame;
  only the debug view mirrors what it draws.

## 5. Calibration (`gaze/calibration.py`) — THE CORE

### 5.1 VERIFIED REFERENCE CODE — use as-is

`gaze/features.py`, `gaze/calibration.py`, `gaze/smoothing.py` and
`tests/test_calibration_pipeline.py` are **verified, tested implementations of
the calibration math. Do NOT rewrite, simplify or "improve" them.** Wire the
camera, UI and interaction layers around them. If a change seems necessary,
ask first.

Everything else in `gaze/` (`calibration_session.py`, `drift.py`) is ordinary
code that *calls* the core; it contains no mathematics of its own.

### 5.1b Head modes — fixed-head is the DEFAULT

**Fixed head (default).** The target user's head rests on a physical support,
so pose does not vary during a session. In this mode:

- **Stage A (the head sweep) does not run.** Calibration starts on the first
  9-point target.
- The `PoseCompensator` keeps its zero sensitivities, so `compensate()` is the
  identity and no pose correction is applied.
- Selected by `fixed_head: true` in config (the default) or
  `calibrate.py --fixed-head`.

**Free head (optional, not default).** For a user whose head is *not*
supported, the head sweep is still available and still necessary — this is the
one thing that stops head movement wrecking accuracy, and the measurement
behind that is unchanged: **~590 px of error without pose compensation versus
~10 px with it** when the head moves. Never delete this path, and never
present free-head use without it.

> Stage A exists because pose compensation **cannot** be learned from the
> 9-point session: users naturally turn their head toward each target, which
> confounds the head and eye signals. The sweep isolates pose sensitivity by
> holding gaze fixed while pose varies.

*Stage A, when it runs:* the user fixates a single CENTER dot (~5 s) while
slowly rotating the head left/right and up/down (±10–15°). This yields
`s = d(iris ratio)/d(degree of yaw or pitch)` via `PoseCompensator.fit`.
`fit_pose_compensation()` returns False if the user did not move enough (< 40
valid frames, or < 6° of range on both axes) — the UI must then explain why and
repeat the step. Selected by `calibrate.py --with-head-sweep`, or
`fixed_head: false` in config (which `main.py` also honours).

### 5.1c The setup check — five seconds before the nine dots

**A bad sitting is visible before calibration and invisible during it.** When
the camera sits below eye height the eyelids crop the iris, the vertical iris
ratio barely moves between the top and the bottom of the screen, and the fit
has almost no vertical signal to work with. The user finds this out forty
seconds later as a MARGINAL verdict they cannot interpret — it just feels like
the app is bad. Two measured sittings on the development machine:

| sitting | hy span | calibration |
|---|---|---|
| camera at eye height | **0.051** | PASS 44.3 px |
| camera at eye height, another session | **0.042** | PASS 57.3 px |
| camera below eye height | **0.024** | MARGINAL 117.5 px, 0 keys typed in 47 s |

So `python main.py` measures that number **first**, in about five seconds
(`gaze/setup_check.py`, `ui/setup_check_screen.py`):

1. A **lead-in** (~1.5 s): the instructions, and **no dot and no measurement**.
   Everything the user has to read is here, so that reading it is not charged
   to a target's median, and the first dot's appearance is itself the cue to
   look. See "Pacing" below — this is not decoration.
2. Two still targets in the same visual language as calibration — a neutral dot
   with a thin ring filling around it — at the **top and bottom centre of the
   calibration region**, at exactly the 10% and 90% heights the 3×3 grid uses,
   so the number is directly comparable with the `hy span` printed after a
   session. 1000 ms settle, then 30 valid samples or 2.0 s per target. While
   they are showing, the only copy is "Look at the dot   1 / 2".
3. Each target is aggregated by the **verified `aggregate_point`** — the same
   1.5×IQR rejection and median a calibration point gets — so the span is the
   same quantity the diagnostics print rather than a lookalike computed a
   second way. The span between the two medians is the measurement.
4. Below `setup_check_min_hy_span` (**0.035**, between the measured sittings
   and deliberately nearer the bad one) the screen says
   **"Camera looks too low — raise the laptop so the camera is at eye height"**
   and shows the measurement. Fewer than `MIN_SAMPLES_PER_POINT` kept samples
   on either target is reported differently — that is lighting or distance, not
   camera height.
5. **It never blocks.** Space re-runs the check *from the lead-in*, Enter
   calibrates anyway (and the console says the result was overridden), Esc
   quits. A passing check shows no screen at all: it prints the span and goes
   straight to the dots.
6. The measured span is printed **either way**, pass or fail, with the
   kept/collected counts.
7. `--skip-setup-check` skips it entirely.

**Pacing — why the check is paced at all** (found in production, P0). The first
version measured from the instant the screen appeared and put the explanation
next to the first dot. A median flips **wholesale** once more than half its
samples are stale, so a user still reading was not measured approximately — they
were measured *entirely* on where they had been looking, and the span collapsed
toward zero instead of degrading. On a sitting that calibrates to 0.042 it read
**0.006 / 0.008 / 0.009 / 0.011 / 0.016 across five consecutive runs**, failing
every time, with the two medians landing on the centre of the screen (0.478 and
0.486) rather than on the dots. Three rules follow, and each is load-bearing:

- **A target tolerates `settle + collect/2` of reaction**, and nothing more.
  At the original 500 ms / 30 samples that was 1.05 s — less than it takes to
  read three lines of instructions. At 1000 ms / 30 samples it is **1.5 s**,
  against the ~0.3 s a saccade to a dot appearing on a still screen takes.
  `SetupCheckSession.reaction_budget_s` states it, and a test pins it.
- **Everything the user must read happens in the lead-in**, before any target.
- **The screen discards the gaze queued before it existed** (spec Section 3:
  the queue is unbounded and nothing else drains it between `wait_for_camera`
  and the first tick), and again on a retry. Frames captured before a dot was
  shown are not evidence about that dot.

The rendered dot positions are checked against the calibration grid's extreme
rows **in screen pixels on a full-size window**, not in region-normalised
space — checking the session's own numbers against themselves is what let this
go unnoticed.

**Answered from the keyboard, not by gaze, and therefore startup-only — a
product rule** (approved 2026-08-08; CLAUDE.md rule 9 applies). There is no
calibration yet, so nothing on screen could be aimed at reliably, and the fix
being asked for needs a pair of hands anyway. It follows that the check must
**never** be put in front of the in-session `Recal.` key: a gaze-only user
mid-session could not answer a keyboard-rendered screen, and would be trapped
between a keyboard they cannot use and a check they cannot dismiss.

**Stage B — 9-point mapping (both modes):** pose-compensated ratios
`(hx_c, hy_c)` are mapped to screen pixels by ridge-regularized 2nd-order
polynomial regression per axis:

```
X = a0 + a1*hx_c + a2*hy_c + a3*hx_c² + a4*hy_c² + a5*hx_c*hy_c
```

Features are standardized with std floors and clamped to ±3σ at prediction time
(prevents catastrophic extrapolation). All of this is already implemented.

### 5.2 Calibration session UI flow

0. **Setup check** — the two-target camera check of Section 5.1c, at startup,
   unless `--skip-setup-check`.
0b. **Head-sweep screen** — free-head mode only (Section 5.1b). Skipped
   entirely in the default fixed-head mode.
1. Fullscreen dark screen. **9 targets, one at a time**, on a 3×3 grid at
   10%/50%/90% of the **calibration region** (Section 5.6), in randomized
   order. With `--cal-region full` the region is the whole display and this is
   10%/50%/90% of screen width and height.
2. **The target is calm and deliberately still.** No shrinking circle, no
   pulsing, no colour change, no flash — anything animating near the fixation
   point makes it harder to hold a steady stare. The *only* feedback is a thin
   ring around the dot: **empty while it settles, filling while it measures.**
3. Each target waits **700 ms** (saccade settling, config
   `calibration_settle_ms`), then collects until it has **45 valid samples**
   (`calibration_collect_samples`) or **4.0 s** have passed
   (`calibration_collect_max_s`), whichever comes first. Driving collection by
   *valid sample count* rather than wall clock means a blink extends a target
   instead of corrupting it. `--slow` presets 1000 ms / 60 samples for
   first-time and accessibility use.
4. **Outlier rejection per point:** discard samples outside 1.5×IQR of the
   median on hx or hy. If < 15 survive, repeat that point automatically (max 2
   retries, then continue with the best available aggregate).
5. The point's representative feature = **median** of survivors.
6. Fit the model on the 9 (feature → target) pairs. If fewer than 6 points
   produced usable data, fail out with an explanation rather than fitting.

### 5.3 Point-level repair, then the validation gate (mandatory)

**Point-level repair — before validation, in every verdict band.** A ridge
polynomial has only nine points to work with, so one target the user glanced
away from drags the whole surface toward it. Waiting for the verdict to fail
misses the case this exists for: one bad point turning an otherwise ~20–25 px
session into a MARGINAL 102.7 px, which passes the FAIL threshold and is
therefore never repaired.

So immediately after the initial fit, and before any validation target is
shown:

- compute each calibration point's residual against that fit;
- a point is **suspect** when its residual is **both > 3× the median residual
  and > 60 px** — the ratio catches a point far worse than its peers, the
  absolute floor stops ordinary scatter in a tight session being disturbed;
- if **exactly one or two** points are suspect, re-collect those points once
  and refit before validating. Three or more is not a glance away, it is
  systemic (lighting, the head slipping, a knocked camera); re-collecting will
  not fix it, and the diagnostics name the cause instead;
- the console says what it is doing:
  `point 2 fit poorly (283 px vs 77 px median) - re-collecting`, and the screen
  shows the same "Improving accuracy" copy as the FAIL-path refit;
- **keep the better of the two.** Both collections are scored against the same
  model — the fit that flagged the point — so the comparison is fair; a
  re-collection that came back worse, or produced no usable samples at all, is
  discarded and the original kept. A repair never loses a point.
- **once, ever.** The pass runs at most one time per session and cannot chain
  into the FAIL-path refit below, so no input can make it loop.

*Known limits of residual-based detection.* With six terms fitted to nine
points, a single outlier has enough leverage to bend the surface toward itself
and inflate its **neighbours'** residuals too. Measured against the synthetic
eye, one glance away raises 1–2 points over the threshold in most target
geometries (where the repair then fires and recovers the session — e.g. 59.6 px
→ 19.0 px), but at the two corner positions it raises three or four, and the
"exactly one or two" gate then declines to act. Leave-one-out residuals would
unmask that, at the cost of nine extra least-squares solves; the trigger above
is what is implemented.

### 5.3b The validation gate (mandatory)

7. Show **3 fresh validation targets** not on the 3×3 grid — at (30%, 30%),
   (70%, 70%) and (70%, 30%) **of the same region**. Collect the same way,
   predict with the fitted model, compute mean Euclidean error in pixels.
   The result is therefore an **in-region error** (Section 5.6): the accuracy
   that applies where keys are, which is what NFR-2 is judged against.
8. Verdict:
   - **error ≤ 80 px → PASS.** Save.
   - **80–130 px → MARGINAL.** Save, with a warning and a larger-keyboard hint.
   - **> 130 px → FAIL.** Not saved. Re-collect the 2 calibration points with
     the worst fit residuals, refit, re-validate once. If it still fails, show
     the guidance screen (face the camera, improve lighting, sit ~60 cm away,
     keep the head still) and offer a restart. This is the same mechanism as
     the point-level repair above, triggered by the verdict instead; the two
     are each one-shot and cannot chain.
9. **Display the measured error, always** (e.g. "Accuracy: 54 px — Good"),
   together with a diagnostics map: per-point residuals, kept/collected sample
   counts, retry and re-collection markers, validation arrows from target to
   prediction, the hx/hy feature spread, which head mode ran, which region was
   measured, and one line naming the most likely limiting cause. The same
   table — including every repair line — is printed to the console every run.

### 5.4 Persistence — and why there is no startup question

Save to `~/.gazekey/calibration.json`: coefficients, scaler, pose
sensitivities, validation error, screen resolution, camera index, timestamp.

**`python main.py` always calibrates.** There is deliberately no "use the saved
calibration?" screen. For this user population the honest answer is almost
always *no*: the head is re-seated on its support between sessions, so a
calibration measured against yesterday's head position is stale in a way the
user cannot perceive — it simply feels like the keyboard got worse. Asking put
the hardest judgement call of the session on the user before they had any
evidence to answer it with.

- Persistence **remains**: the M5 touch-up and in-place recalibration both
  write to the file.
- `--use-saved` is a **dev-only** flag that skips calibration and loads the
  stored model, for working on the UI. It still refuses a model whose screen
  resolution or camera index does not match, and calibrates instead; it also
  adopts the region that model was fitted over (Section 5.6).
- `--recalibrate` has been **removed** — calibrating *is* the default.

The default flow is therefore **calibrate → keyboard**. The aiming drill sits
behind `--practice`.
- `calibrate.py` remains as a dev tool for working on calibration alone.

### 5.5 Drift handling (`gaze/drift.py`) — as built in M5

Calibration decays during a session: the head settles deeper into its support,
the chair rolls back a centimetre, the light changes. The mapping is still the
right *shape*, it just sits a few dozen pixels off — so the fix is a
**translation**, not another nine points.

#### 5.5.1 The monitor — it reports, it never acts

Two independent signals:

- **Every `char` key activation is weak ground truth.** The user *was* looking
  at that key when it fired, so `key centre − gaze centroid` is one measurement
  of the offset; the estimate is an exponentially-weighted mean of those
  (α = 0.3), and the centroid is the mean gaze over the last **300 ms**,
  excluding blink-held samples (which repeat the last position and would bias
  it toward wherever the eye closed).
  **Only `char` keys count.** Space is ~250 px wide and the control keys are
  oddly shaped; their centres say almost nothing about where the user was
  looking, and including them would bias the estimate toward key geometry.
  Offsets from fewer than **5** activations are treated as noise.
- **Type-then-backspace within 3 s** counts a correction. **3 inside 60 s** is
  enough on its own.

`score = offset_px / threshold + corrections / 3`; drift is flagged at
`score ≥ 1`, so either signal can trip it alone and partial evidence from both
adds up. The flag is **sticky**: once raised it only clears below 0.7, so the
indicator cannot blink on and off between keystrokes. `threshold` defaults to
the measured calibration error with a 50 px floor; `drift_offset_px` in config
pins it instead.

> **A limit worth stating.** A dwell only completes *inside* a key, so the
> measurable offset can never exceed half a key. Drift larger than that shows
> up as keys that stop activating at all — which is exactly why the correction
> count is the second signal and not a nicety.

When it trips, **one line appears in the corner status panel**:
`LOW ACCURACY: aim has drifted about NN px - look at Fix aim to correct it`
(or `look at Recal.` on a board with no Fix aim key). That is the entire
intervention. **Nothing opens a screen, nothing recalibrates itself, and typing
is never interrupted** — deciding to stop mid-sentence is the user's call.
Recalibration and touch-up reset the monitor, because the accumulated evidence
was about the old model.

#### 5.5.2 "Fix aim" — the 1-point touch-up

Reachable by gaze from the keyboard: a **Fix aim** key on the control row with
the 2 s extended dwell (it rewrites the calibration from wherever the user
happens to be looking, so a mis-dwell on it is worse than a mistyped letter).

- One still centre dot on a dark fullscreen screen, same visual language as
  calibration: **0.4 s settle + 2.0 s measure = 2.4 s**, well inside the 10 s
  that decides whether the sentence survives.
- The **median** of the collected gaze against the dot gives `(dx, dy)`,
  applied through `CalibrationModel.apply_offset` — which touches **only
  `wx[0]` and `wy[0]`**, the two constant terms. The curvature of the verified
  fit is never altered.
- **Refusals, reported rather than silently applied:**

  | condition | message |
  |---|---|
  | fewer than 10 samples | *"could not see your eyes for long enough"* |
  | offset > 25% of the screen diagonal | *"too far to be drift — recalibrate instead"* |

- An accepted touch-up saves the corrected model, resets the One-Euro filters
  and clears the drift evidence. Esc cancels and changes nothing. Either way
  the outcome appears as a transient line in the keyboard's corner panel —
  there is no result screen to dismiss.

#### 5.5.3 Recalibrate — in place

The **Recal.** key (2 s dwell) runs the full 9-point session and returns **to
the keyboard**, never to a startup screen (there is none). The typed line, the
word in progress, the language, the page and the pause state are all carried
across; the personal dictionary is untouched. The board is rebuilt at whatever
key size the *new* accuracy calls for, so improving the calibration can change
the layout — that is intended. The new model must be installed into the
pipeline before returning, or the session comes back looking identical while
still predicting from the old fit. Esc mid-recalibration returns to typing on
the old model rather than ending the session.

### 5.6 The calibration region (`gaze/region.py`)

A webcam gaze estimator has a roughly fixed angular error, so spreading nine
dots over the whole display spends most of that accuracy on places the user
never looks. **The nine calibration dots and the three validation targets span
the interaction region — the keyboard — not the screen.**

**The default region** is the board as built from the current height ratio,
plus a margin of **7.5% of screen height above it** (`REGION_MARGIN_RATIO`).
Only the top edge moves: the board is docked to the bottom and spans the full
width, so the other three sides are already at the screen edge. On 1366×768 at
the 2/3 height this is a 1366×570 region, 74% of the screen.

**The margin is sized against the dot hull, not the region edge.** Dots sit at
10/90% of the region, so their convex hull is inset by a tenth of it; the
margin has to be large enough to lift that hull above the top row of keys. At
5% it was not — hull top at y 273 against a board top of 256, leaving the
suggestion bar extrapolated. At 7.5% the hull top is y 255, just inside.

For `qwerty-tall` the region matches the board exactly, because that layout's
geometry depends only on the height ratio. The **error-sized layouts (`paged`,
`auto`) cannot be measured before the calibration exists**, so their region is
bounded by `MAX_HEIGHT_RATIO` (0.8) instead — 85% of the screen, which covers
every board they can produce.

Consequences, all deliberate:

- **The reported validation error is an in-region error.** It is the number
  NFR-2 is evaluated against (Section 14), because it is the accuracy that
  applies where keys are.
- **Every gaze-selectable target must be inside the region.** The Quit
  confirmation's YES/NO boxes and the Fix aim dot are placed within it rather
  than at the centre of the screen; the aiming drill's targets too. A target
  outside the fitted area is aimed at with accuracy nobody measured. Startup
  prints a warning naming any key that ends up outside.
- **Outside the region the model extrapolates.** The optional gaze cursor may
  still be drawn up there and is expected to be loose. That is fine — nothing
  is selected up there.
- **The bottom row is always outside the hull.** No margin can fix it: the
  board is flush with the screen edge, so the last tenth of the region sits
  below the lowest dot row — 57 px here, against 77 px for a whole-screen
  calibration. The ±3σ clamp in the verified core keeps that extrapolation
  bounded rather than explosive.

**Persistence.** The region is written into `~/.gazekey/calibration.json`
alongside the model, as an extra key — `CalibrationModel.save` is verified core
and must not learn about regions. `--use-saved` reads it back and adopts it, so
the stored error is always interpreted against the area it was measured in. The
touch-up and the in-place recalibration reuse the session's region, and the
touch-up re-attaches it after `model.save` rewrites the file.

**`--cal-region full`** restores whole-screen calibration exactly (the
full-screen region reproduces the old target positions), for A/B comparison.
`calibrate.py` defaults to `full` because it ends at a whole-screen gaze demo.

## 6. Smoothing & fixation (`gaze/smoothing.py` — VERIFIED CORE)

- **OneEuroFilter** on predicted (x, y), per axis: `min_cutoff=1.0`,
  `beta=0.007`, `d_cutoff=1.0` (Casiez et al. 2012).
- **FixationDetector** (dispersion / I-DT): sliding window of **150 ms**. Note
  what the verified implementation actually thresholds: **`Δx + Δy`, the sum of
  the two per-axis ranges** inside the window, so the budget is shared between
  the axes.
- **The dispersion threshold is derived from the measured accuracy, not fixed.**
  The original spec named 60 px; residual wander while genuinely staring scales
  with calibration error, so the shipped rule is
  `dispersion = clamp(1.35 × validation_error, 60, 220)` — 110 px at the ~81 px
  this setup achieves, against a minimum key pitch of ~162 px. The ratio sits
  between two walls: too low and a real stare keeps breaking so dwell never
  completes; too high and a saccade reads as a fixation and keys fire in
  passing. Staying under 2× guarantees the latter cannot happen, because NFR-2
  sizes keys at ≥ 2× the error. Override with `fixation_dispersion_px`.
- **Invalid frames (blinks, face lost):** hold the last output for up to
  **300 ms** with the fixation verdict *frozen*, then mark the stream invalid.
  The UI shows "tracking lost"; the dwell timer freezes rather than resetting,
  so a blink mid-dwell costs time but not progress.
- Two distinct notions of validity travel on every sample: `valid` (this frame
  had a usable face — what the calibration session counts) and `stream_valid`
  (a usable position exists right now, fresh or held — what the dwell uses).

## 7. Interaction controller (`interaction/controller.py`)

State machine per focused key: `IDLE → FOCUS → DWELLING → ACTIVATED`.

- Dwell threshold: config, default **1.0 s** (allowed 0.5–2.0).
- Dwell advances **only while `is_fixating` is true** and gaze is inside the key.
- **Hysteresis — sticky focus.** Once a key is focused its hit region grows by
  25% on each side, and **the focused key is asked first**: it keeps the point
  anywhere inside that grown region, so a challenger has to win the point from
  *outside* it.

  > This is what the rule always meant, but not what it used to do. Hit regions
  > are gapless, so asking the challengers first meant one of them always owned
  > the point inside the board and the margin was measurably inert between keys
  > — ownership was byte-identical at margins 0.0, 0.25, 0.45 and 0.90. Asking
  > the incumbent first is the whole fix (NFR-7, adopted).

- **A key always owns its core — a product rule, not a tuning detail.**
  (Approved 2026-08-08; changing it needs the owner, per CLAUDE.md rule 9.)
  The core is the middle of a key left when a fixed fraction (`CORE_MARGIN`,
  0.25) is taken off each side, and it belongs to that key **whatever the
  hysteresis margin is set to** — including any future per-axis margin. The
  margin is a fraction of the *focused* key's size, and Space is 250 px against
  the 124 px Backspace beside it: unbounded, a focused Space reaches past
  Backspace's centre and Backspace becomes unselectable. The guarantee, stated
  plainly: **hysteresis can never cost the user a key they are looking straight
  at.** No stickiness rule may be added that weakens it.

- **Grace period — a dwell survives losing its key.** Accumulated dwell decays
  linearly over 200 ms rather than being thrown away, and coming back inside
  that window resumes it where it left off. This applies to **both** ways a key
  can be lost:
  - the gaze left every key (the original path), and
  - **a neighbour took focus.** The interrupted dwell is *carried*: the new key
    starts its own dwell at zero while the old key's decays in the background,
    and returning to it restores whatever is left. One jittered frame across a
    row boundary therefore costs one frame of decay instead of the whole
    second, which is the reported H↔N↔Y flapping.

  There is **one carry slot**: a second steal replaces it, so a wander across
  three keys keeps only the most recent — the one the gaze is most likely to
  return to. `grace_period_ms: 0` reproduces the old discard-on-steal rule
  exactly.

- **The counts say which of those happened.** A steal is recorded as
  `focus_stolen` only when its carried dwell expires; one that comes back in
  time is `focus_recovered` and costs nothing. Counting the steal at the moment
  it happened would report a loss that did not occur and hide the mechanism
  that prevented it.
- **Refractory period:** 400 ms after activation during which the same key
  cannot re-activate.
- **Extended dwell 2.0 s** on the modal and destructive keys:
  **PAUSE, FIX AIM, RECALIBRATE, LANG, QUIT.**
- **While PAUSED**, tracking and all visual feedback continue but nothing is
  injected. Four keys stay live regardless: **PAUSE, FIX AIM, RECALIBRATE,
  QUIT.** None of them types, and each is something a paused user still has to
  reach — unpausing, fixing the aim that made them pause, and leaving. A paused
  keyboard whose Quit key does nothing is a trap.
- An **invalid stream** freezes the dwell where it is: it neither advances nor
  decays.

## 8. Keyboard & injection

### 8.1 The default layout (`qwerty-tall`)

Two thirds of the screen height, docked at the bottom, near-black keys with
white labels — modelled on commercial gaze keyboards, everything reachable
without paging. Rows, top to bottom:

1. **4-slot suggestion bar** (not 3 — the original spec's figure). The slots
   are real dwell targets: dwelling one types the rest of the word plus a space.
2. **Typed-text preview line** — what has been typed so far, tail-anchored.
3. **10-column QWERTY**, all letters on one page, plus `' , . ?`.
4. **Control row: Space (2 columns), Backspace, Enter, Shift, 123, EN/HE,
   Pause, Fix aim, Recal., Quit.**

The control row runs on its own **11-column grid**, one denser than the letter
rows, so the M5 Fix aim key fits without shrinking Space (the most used key) or
displacing Quit (an exit route). Both grids span the full width, so the rows
stay gapless. At the default height the narrower 124 px control keys still
clear the 93 px the row height allows, so the binding NFR-2 axis is unchanged.

The **123** key swaps to a digits/symbols page (the control row and suggestion
bar persist across pages). A **live webcam self-view** sits at the top-right of
the board, spanning the suggestion and preview rows (~273×140 px on
1366×768) — at the *top* right because the bottom row is control keys and every
alternative there either shrank Space or displaced Quit.

### 8.2 The fallback layout (`--layout paged`)

An 8-column, two-page alphabetical board with much larger keys, for when a
10-column row cannot satisfy NFR-2. Alphabetical rather than QWERTY because a
paged split destroys QWERTY's muscle memory anyway, and scanning for a letter
is what actually costs time when selecting by gaze. It gains a 9th control
column carrying Fix aim **only if that column still clears the required key
size**; where it does not, Recal. does the same job more slowly and the drift
indicator says so.

`--layout auto` keeps the older accuracy-driven selection between full QWERTY
and the paged board. `qwerty-tall` is the default even where it misses NFR-2 —
the shortfall is reported, never hidden (Section 14, NFR-2).

### 8.3 Injection (`interaction/injector.py`)

`pynput`. For Hebrew characters, inject the unicode character directly
(`keyboard.type(char)`) rather than simulating physical key positions — this
avoids depending on the OS keyboard layout. Enter/Backspace/Space use key
codes. Shift latches for exactly one character.

The overlay window must be **non-focusable and click-through**
(`Qt.WindowDoesNotAcceptFocus | Qt.Tool | WindowStaysOnTopHint |
FramelessWindowHint`, `WA_TranslucentBackground`, `WA_ShowWithoutActivating`,
`WA_TransparentForMouseEvents`) so injected keys go to the app behind it and
clicks reach the desktop underneath. Without these, every injected keystroke
lands back in GazeKey.

### 8.4 Word prediction — done (pulled forward from M6)

Local, instant and offline (NFR-5): nothing about what is typed leaves the
machine, and nothing is looked up over a network. Ranking blends a **bundled
English frequency list** with a **personal count** of the words this user
actually picks, persisted to `~/.gazekey/user_words.json` — **words and counts
only, never sentences or key logs**. A word picked a few times climbs above
more common English, which is the point for someone typing the same names every
day. Typing `he` offers `her here hello hey`; dwelling one types the remainder
plus a space. Hebrew prediction is M6: the same mechanism with a Hebrew
frequency list if feasible, otherwise gracefully absent.

## 9. UI requirements (`ui/`)

- Keyboard overlay docked at the bottom, **~2/3 of screen height**
  (`keyboard_height_ratio`, default `2/3`; `--height-ratio` overrides). The
  original spec's 0.35 is superseded — at webcam accuracy a third of the screen
  cannot give keys of a usable height.
- The overlay window spans the whole display so the gaze cursor and status
  icons work everywhere, but it only *paints* the docked board.
- Focused key: highlighted + circular **progress ring** filling with dwell
  progress. Activation: brief flash + optional click sound.
- Optional gaze cursor (semi-transparent dot): blue while travelling, green
  while the fixation detector agrees, amber while a blink is being held. It
  disappears when tracking is lost rather than lying about a stale position.
- **Corner status panel**, capped at 40% of screen width so it never sits over
  the app being typed into. Lines it can show: paused, waiting for the camera,
  tracking lost, no keystroke injection, **`keys below spec: …`** (NFR-2, both
  axes), **`LOW ACCURACY: aim has drifted …`** (drift), transient outcome
  notices, and the current page. The two accuracy warnings are deliberately
  named differently: one wants a bigger keyboard, the other a corrected aim.
- Calibration screen: fullscreen, black, **calm non-animated** targets
  (Section 5.2), a "3 / 9" counter, and a validation result screen with the
  measured error, the verdict and the diagnostics map.
- **Target practice** is **opt-in** (`--practice`): the default flow goes from
  calibration straight to the keyboard, because a ten-target test between
  someone and the thing they opened the app to do is a tax on every launch.
  When asked for, it runs between calibration and the keyboard: 10 dots
  hold a fixated gaze inside the hit radius (`max(60, error)` px) for 0.8 s to
  pop each one, 12 s timeout, Esc skips. It reports hit rate, mean error while
  holding and mean time to hit. It deliberately mirrors the dwell rules
  (fixation-gated, grace decay, freeze on invalid stream) minus hysteresis, so
  it measures the same thing typing depends on, and its targets stay inside
  the calibration region. It reports **average aim error** (every sample of
  every attempt, hits and misses) as well as average error while holding —
  the first is the figure that compares two calibrations, since the second
  only sees gaze that had already landed inside the hit radius.
- Every question the app asks is answerable **by looking**: the Quit
  confirmation is a `Keyboard` of two enormous keys driven by the real dwell
  controller and painted by the real renderer — same timings, same look,
  nothing new to learn.

## 10. Config (`~/.gazekey/config.json`)

Load with defaults for anything missing or unreadable; save on change. Unknown
keys are ignored, so an old or partial file never breaks startup.

```json
{
  "dwell_time_s": 1.0,
  "extended_dwell_s": 2.0,
  "hysteresis_margin": 0.25,
  "grace_period_ms": 200,
  "refractory_ms": 400,
  "language": "en",
  "sound_feedback": true,
  "show_gaze_cursor": true,
  "camera_index": 0,
  "keyboard_height_ratio": 0.6666666666666666,
  "show_webcam_preview": true,
  "fixed_head": true,
  "setup_check_min_hy_span": 0.035,
  "calibration_settle_ms": 700,
  "calibration_collect_samples": 45,
  "calibration_collect_max_s": 4.0,
  "fixation_window_ms": 150,
  "fixation_dispersion_px": 110,
  "one_euro_min_cutoff": 1.0,
  "tracking_hold_ms": 300,
  "drift_offset_px": 0
}
```

`drift_offset_px: 0` means "derive the drift threshold from the measured
calibration error" (Section 5.5.1). `setup_check_min_hy_span` is the gate in
Section 5.1c; `--skip-setup-check` bypasses it for one run.

## 11. Build order — milestones

| | | |
|---|---|---|
| M1 | vision pipeline + live debug view | **done** |
| M2 | 9-point calibration + validation gate + persistence + gaze dot | **done** |
| M3 | One-Euro smoothing + fixation detection | **done** |
| M4 | keyboard overlay + dwell + keystroke injection | **done** |
| M5 | drift monitor + 1-point touch-up + in-place recalibration + PAUSE/RECALIBRATE keys | **done** |
| — | region-scoped calibration (Section 5.6), practice made opt-in | **done** |
| — | point-level calibration repair (Section 5.3) | **done** |
| — | typing stability: sticky focus + dwell survives a steal (NFR-7) | **done, awaiting the after-measurement** |
| — | the 5-second setup check before calibration (Section 5.1c) | **done** |
| M6 | **Hebrew layout + RTL only** | next |

Word prediction was pulled forward out of M6 and is done (Section 8.4), so M6
is Hebrew and RTL alone.

**After M6 — the two end-to-end scenarios**, both by gaze only:
1. Type "Hi" into WhatsApp Web.
2. Type "Dog" into a Google search box.

Each milestone must be runnable on its own, with a stated command and a stated
checkpoint, before the next one starts.

## 12. Tests (pytest, no camera, no visible window)

- `test_features.py` — synthetic landmarks → expected hx/hy; EAR blink gate.
- `test_calibration*.py` — synthetic data from a known polynomial + noise +
  injected outliers; the fitted model recovers ground truth within tolerance;
  validation-gate verdicts correct at the boundaries; a simulated realistic eye
  drives whole sessions.
- `test_controller.py` — scripted gaze streams verify dwell accumulation,
  hysteresis, grace decay, refractory, extended dwell, pause behaviour.
- `test_layouts.py` — NFR-2 sizing on both axes, layout selection, geometry.
- `test_drift.py` / `test_touchup.py` — the M5 core, driven from explicit
  timestamps so the 3 s and 60 s boundaries are tested at their real values.
- `test_typing_diagnostics.py` — the tuning-knob precedence (flag > config >
  default), the dwell-loss accounting, and what each knob now does: the margin
  changes ownership between keys, one jittered frame does not zero the dwell,
  the decay follows `grace-ms`, a key always owns its core, and the refractory
  is unaffected by a carried dwell.
- `test_setup_check.py` / `test_setup_check_ui.py` — the gate of Section 5.1c:
  the measured sittings land either side of the threshold, the targets sit on
  the calibration rows and inside the region, too few samples fail differently
  from a low span, a retry measures from scratch, and continuing anyway is
  recorded rather than hidden. Plus the three that pin the P0 fix: **a user
  reacting at human speed (up to 1.2 s) is measured correctly**, the lead-in
  feeds no target, a stale backlog cannot spend one, and **the painted dots
  land on the calibration grid's extreme rows in screen pixels** at full
  resolution.
- `test_region.py` — region geometry, target placement, persistence, and that
  every gaze-selectable target (keys, the Quit boxes, the Fix aim dot, the
  drill) lands inside the calibrated area.
- `test_calibration_session.py` — point-level repair: a glanced-away point is
  re-collected and the accuracy recovers, a clean session is left alone, three
  outliers are treated as systemic, the worse collection is discarded and the
  pass never loops.
- `test_overlay.py` / `test_drift_ui.py` / `test_touchup_ui.py` — window
  contract, sample→activation→injection wiring, the drift indicator.
- `test_m5_flow.py` / `test_main_flow.py` — application wiring: what is
  suspended, what comes back, what survives a recalibration.

The Qt "offscreen" platform has no font database, so UI tests check on-screen
copy through `text_lines()` / `status_lines()` (the same source the painter
draws from) and check geometry against painted pixels.

## 13. Definition of done

- **Calibration validation error is measured and displayed, never assumed**
  (≤ 80 px = PASS).
- **The user can type a sentence into a browser using only their eyes, with
  ≥ ~90% correct key selections** — measured on **the layout that satisfies
  NFR-2 for the calibration error actually achieved** (in practice the paged
  layout at ≤ ~81 px). Full QWERTY at webcam accuracy is expected to be usable
  but below this bar; that is an **accepted trade-off, not a defect**, and the
  app says so in its own status panel rather than pretending otherwise.
- ≥ 20 FPS on CPU, the UI never freezes, no crash when the camera disconnects.
- `pip install -r requirements.txt` + **one command** runs the app; the README
  explains everything.
- All four exit routes work (NFR-6).
- Type hints + docstrings on public APIs; each module independently importable.

## 14. Non-functional requirements

### NFR-1 — Performance
≥ 20 FPS end-to-end on CPU. The Qt thread never blocks on vision work; exactly
one screen consumes the gaze queue at a time (Section 3).

### NFR-2 — Key size versus measured accuracy

> **Compliant ⇔ `validation_error_px ≤ min(key_width, key_height) / 2`**
> for the **smallest selectable key**, measured over the **real key
> rectangles**, where `validation_error_px` is the **in-region error** of
> Section 5.6.

Equivalently: every selectable key must be at least twice the expected gaze
error **on both axes**, so a prediction landing one error away from where the
user was looking still falls inside the intended key.

**The error is measured over the interaction region, not the screen.** That is
the only error that means anything here: NFR-2 is a statement about whether
gaze lands in the right *key*, so the accuracy it consumes has to be the
accuracy where the keys are. A whole-screen figure would mix in error from
parts of the display nobody ever selects on. Comparing an in-region error
against a whole-screen one is meaningless in both directions — always say which
was measured, as the results screen and the console report do.

Four points that were previously got wrong and are now normative:

1. **Both axes, and the minimum of them.** Not the width, and not a nominal
   cell size. A key that is wide but short fails exactly as surely as a narrow
   one: an error of one radius in the short direction lands outside it either
   way.
2. **Measured over real rectangles**, of *selectable* keys only — a control row
   at a different column count, a spanning Space, or a non-selectable strip
   cannot hide behind an average.
3. **The warning must report both axes, name the binding one, and quote an
   accuracy target rounded DOWN.** "Recalibrate to about 47 px" when 46.5 is
   the real limit is advice that fails when followed.
4. **The error compared against is the in-region one** (Section 5.6), and every
   report says which region it was measured over.

Keys are never built below a **90 px floor**, however good a calibration
claims to be.

**Real numbers on the 1366×768 development screen** (keyboard at the default
2/3 height; error measured over the 1366×570 keyboard region):

| board | smallest selectable key | NFR-2 complies up to |
|---|---|---|
| `qwerty-tall` — QWERTY letters | 137 × 93 px | — |
| `qwerty-tall` — control keys (11 columns) | 124 × 93 px | — |
| **`qwerty-tall` overall** | **124 × 93 px** | **≤ 46 px — height binds** |
| `paged` (8 columns) at 81 px error | 171 × 162 px | **≤ 81 px** |
| `paged` — ceiling on this screen | 171 px wide max | ≤ 85 px (width binds) |

So on this screen the target to aim for with the default board is **~46 px**,
not the ~68 px that the letter width alone suggests (and not the ~62 px that
the *narrowest* column alone suggests). Raising `keyboard_height_ratio` lifts
the height limit (0.86 → 60 px, 0.88 → 62 px); beyond that the column width is
the ceiling at ~62 px however tall the board gets. The paged fallback is the
layout that actually complies at realistic webcam accuracy.

**Non-compliance is a warning, never a block.** The app runs, prints the key
size and the verdict, shows `keys below spec: …` in the corner with both axes,
and names what would fix it.

### NFR-7 — Typing stability (fixes 1 and 2 adopted; 3 deferred)

A dwell that never completes is as unusable as a bad calibration. The keyboard
must let a user hold a key through the residual wobble of their own gaze.

**What was measured.** With a 44.3 px in-region calibration and 93 px rows, a
single frame of vertical jitter stole focus and discarded the whole accumulated
dwell. Of the three tuning knobs, **two could not affect this at all**:
hysteresis was inert between keys and grace was never reached on a steal. Only
`--min-cutoff` bit, by reducing the wobble itself. A second session on a worse
sitting (117.5 px, hy span 0.024) recorded the extreme: **0 keys activated in
47 s across 40 dwell attempts** — 18 steals, 13 fixation drops, 9 grace
expiries, focus changing 3.7×/s, jitter x 34 y 152 px.

**Adopted (both are Section 7 conformance, not new behaviour — the spec always
said the focused key's grown region keeps ownership, and that dwell decays over
the grace rather than resetting):**

1. **Sticky focus** — the focused key is asked first, so its grown region
   actually decides ownership and `--hysteresis` is live. Bounded by
   `CORE_MARGIN` so a wide key can never swallow a narrow neighbour (Section 7).
2. **The dwell survives a steal** — it is carried and decays over the grace
   instead of being zeroed, making `--grace-ms` live. A steal counts as a loss
   only if the gaze does not come back in time; one that does is reported as
   `saved`/`recovered`.

Defaults are unchanged: margin 0.25, grace 200 ms.

**Deferred pending the after-measurement:**

3. **Per-axis hysteresis** — rows are 93 px and columns 124–137 px, and the
   vertical jitter is the larger, so a separate vertical margin targets the axis
   that actually fails. Not adopted: 1 and 2 may already cover it, and a second
   axis-specific knob is only worth its complexity if the measurement still
   shows steals dominating.

`--debug-typing` measures which of the failure modes is in play:

```
[typing]  5.0s  focus 12.3/s  resets: steal 25  fixdrop 0  grace 0  saved 3
                jitter x   6 y  45 px  spread x 201 y  69  fixating 100%  keys 0
```

`jitter` is the median per-axis deviation inside ~1/3 s buckets — deliberately
*not* the window-wide standard deviation, which is dominated by the user moving
between keys and reads ~350 px on x while the real wobble is 6 px. Read it
against the key size: jitter approaching a quarter of a key crosses boundaries
routinely.

**What a healthy session looks like after 1 and 2** — the bar the
after-measurement is read against:

| number | healthy | what it means if it is not |
|---|---|---|
| `keys` per dwell attempt | **≥ 90%** | dwells are completing |
| `steal` | low, and **fewer than `saved`** | the carry is doing its job; if steals still dominate, fix 3 |
| `fixdrop` | near zero | otherwise raise `fixation_dispersion_px` |
| `grace` | near zero | otherwise the gaze is leaving the board entirely — calibration, not interaction |
| `focus` changes | **< ~1.5/s** while typing | 3–4/s is flapping between neighbours |
| `jitter y` | **< ~23 px** (a quarter of a 93 px row) | above it, boundary crossings are routine |
| `fixating` duty | **> 60%** | below it the detector, not the keyboard, is the limit |

### NFR-3 — Robustness
No crash when the camera is unplugged or held by another app. The overlay never
steals keyboard focus and never swallows a click meant for the app behind it.

### NFR-4 — Privacy
Video frames and landmarks stay in memory and are **never written to disk**.
The only files GazeKey writes are `~/.gazekey/config.json`,
`~/.gazekey/calibration.json` (coefficients, not imagery),
`~/.gazekey/user_words.json` (words and counts only) and the cached MediaPipe
model bundle.

### NFR-5 — Local and offline
All processing is local. No network access at runtime beyond the one-time
MediaPipe model-bundle download. Word suggestions are instant and offline.

### NFR-6 — Four independent exit routes (permanent — never remove any)

A gaze user cannot alt-tab to a terminal, so there are four, deliberately
independent, so that no missing permission, absent mouse or mis-aimed dwell can
leave someone stuck in a keyboard they cannot close:

1. **Ctrl+Alt+Q** — global `pynput` hotkey, works from anywhere, even
   mid-typing. If it cannot register, the launch banner says **UNAVAILABLE**
   rather than pretending.
2. **The Quit key** on the keyboard — 2 s extended dwell, then a YES/NO gaze
   confirmation, so a mis-dwell cannot end the session.
3. **The clickable X** — a separate small always-on-top window
   (`WS_EX_NOACTIVATE`), because the overlay itself is click-through and
   nothing on it can receive a click.
4. **Ctrl+C** in the terminal.

All four are printed in a banner at every launch. Any screen that takes over
from the keyboard must restore the X on the way back — losing it silently costs
one of the four.

## 15. Environment facts (for sizing decisions)

Windows 11, screen **1366×768**, webcam **640×480**, Python 3.13, MediaPipe
**tasks** backend (the legacy `mp.solutions.face_mesh` API no longer ships in
current wheels, and in no Python 3.13 build; `vision/face_tracker.py` supports
both and picks automatically), OpenCV, PyQt5, pynput, numpy.

User's historical best calibration: **81 px**; typical **85–95 px**. **Vertical
error tends to exceed horizontal** — which is exactly why NFR-2 is evaluated on
the minimum key dimension and why the height axis is the one that binds on this
screen.

---

## Appendix — provenance

`SPEC_AMENDMENTS.md` (A1–A10) was merged into this document on 2026-08-08 and
deleted. Where it was already stale against the M5 code, the code won:

| amendment | landed in | note |
|---|---|---|
| A1 fixed-head default | 5.1b | free-head kept as an optional mode with the 590/10 px rationale |
| A2 calm targets + practice | 5.2, 9 | |
| A3 single flow, no startup question | 5.4, 11 | |
| A4 exit routes | NFR-6 | |
| A5 keyboard design | 8.1, 9, 10 | **updated:** control row now also carries Fix aim, on an 11-column grid |
| A6 NFR-2 | NFR-2 | **updated:** smallest selectable key is the 124×93 control key, not the 137×93 letter; the ≤46 px conclusion is unchanged |
| A7 word prediction / no pyenchant | 1, 8.4 | |
| A8 milestones | 11 | **updated:** M5 done; 520 tests, not 436 |
| A9 definition-of-done conditions | 13 | |
| A10 environment | 15 | |
