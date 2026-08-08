# GazeKey

A desktop virtual keyboard driven by eye gaze from a standard webcam, for users
with motor disabilities. Gaze is estimated from the webcam, keys are selected by
dwell time, and real OS keystrokes are injected into the focused application.

**Status: Milestones 1-5 complete, with word prediction (M6) pulled forward,
plus region-scoped calibration, point-level calibration repair, the NFR-7
typing-stability fixes and the pre-calibration setup check.**

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

That is the whole thing. `main.py` walks the user through everything by gaze —
no second command, no terminal step. Python 3.10+ (tested on 3.13); everything
runs locally on the CPU.

## Getting out (read this first)

A gaze user cannot alt-tab to a terminal, so there are **four independent ways
to quit**, printed as a banner every time the app starts:

| | |
|---|---|
| **Ctrl+Alt+Q** | global hotkey, works from anywhere, even mid-typing |
| **Quit key** | on the keyboard — hold 2 s, then confirm by looking at YES |
| **X button** | top-right corner of the keyboard, clickable with a mouse |
| **Ctrl+C** | in the terminal |

They are deliberately independent: no missing permission, absent mouse or
mis-aimed dwell can leave someone stuck in a keyboard they cannot close. If the
global hotkey cannot register, the banner says so rather than pretending.

## The flow

```
launch
  ├─ setup check    2 targets, ~5 s: can the camera see your eyes move up/down?
  ├─ calibration    9 targets over the KEYBOARD AREA, then 3 validation targets
  ├─ result         measured in-region error + verdict + diagnostics map
  └─ keyboard       type into whatever app you click into
                      ├─ Fix aim   →  1 target, ~2.5 s  →  back to typing
                      └─ Recal.    →  9 targets         →  back to typing
```

**The setup check comes first, and takes about five seconds.** A camera below
eye height is the single most common way a session is wasted: the eyelids crop
the iris, the vertical signal collapses, and the user only finds out forty
seconds later as a verdict they cannot interpret. Two dots — top and bottom of
the calibration area — measure how far the vertical iris ratio actually travels.
Measured on this machine: **0.051 sitting well, 0.024 with the camera too low**
(which calibrated to 117 px and typed nothing at all in 47 s). Below the
threshold (0.035) the screen says *"Camera looks too low — raise the laptop so
the camera is at eye height"*, and offers **Space** to check again, **Enter** to
calibrate anyway, **Esc** to quit. It never blocks, it always prints the number,
and `--skip-setup-check` turns it off. It runs at startup only: it is answered
from the keyboard (there is no calibration yet to aim with), and raising a
camera needs hands anyway.

**Every launch calibrates.** There is no "use the saved one?" question, because
for these users the honest answer is almost always *no*: the head is re-seated
on its support between sessions, and a calibration measured against yesterday's
head position is stale in a way the user cannot see — it just feels like the
keyboard got worse. The calibration is still saved to disk (the M5 touch-up and
recalibration write to it), but the file is a development convenience, not the
default path. `--use-saved` loads it and skips straight to the keyboard — for
working on the UI, not for real use.

**The aiming drill is opt-in** (`--practice`). It measures rather than teaches,
and a ten-target test between someone and the thing they opened the app to do
is a tax on every launch.

Every question is answerable by looking. `calibrate.py` still exists for working
on calibration alone; users never need it.

## The keyboard

Two thirds of the screen height, near-black keys with white labels, modelled on
commercial gaze keyboards:

* **suggestion bar** — four slots, filled as you type; dwell one to complete the
  word and add a space;
* **typed-text preview** — a line showing what you have typed so far;
* **full QWERTY**, ten columns, everything on one page;
* **control row** — Space, Backspace, Enter, Shift, 123 (digits and symbols),
  EN/HE, Pause, **Fix aim**, Recalibrate, Quit. It runs on its own 11-column
  grid, one denser than the letter rows, so Fix aim fits without shrinking
  Space or displacing Quit;
* **live self-view** in the top-right of the board, so you can see the tracker
  still has your face.

Focus lightens a key and fills a circular progress ring; activation flashes it.

> The self-view sits at the **top**-right of the board rather than the bottom
> right: the bottom row is control keys, and every alternative there either
> shrank Space or displaced Quit.

### Key size, honestly — and which axis NFR-2 actually measures

Spec NFR-2 wants keys at least twice the expected gaze error. **The check is on
the smallest selectable key's *minimum dimension*, measured over the real key
rectangles** — not the width, not a nominal cell size. A key that is wide but
short fails exactly as surely as a narrow one: an error of one radius in the
short direction lands outside it either way.

That distinction matters on this setup, and an earlier version of this README
got it wrong. On 1366×768 at the default 2/3 height:

| | | |
|---|---|---|
| letter keys | 137 px wide | 93 px tall |
| control keys | 124 px wide (11 columns) | 93 px tall |
| **NFR-2 clears up to** | ~62 px on width alone | **~46 px — height is what binds** |

So the number to aim for on this screen is about **46 px**, not the ~68 px the
width alone suggests. Raising `keyboard_height_ratio` lifts the height limit
(0.86 gets you to 60 px, 0.88 to 62 px); past that the 10-column row is the
ceiling at ~62 px however tall the board gets.

The app never pretends. Every "BELOW NFR-2" message names **both axes, the one
that binds, and an accuracy target rounded down** so that following it actually
clears the gate:

```
keys are 124 px wide x 93 px tall - width BELOW, height BELOW - but NFR-2 wants
190 px on BOTH axes for a 95 px calibration; height is what binds, and as built
this board is only good to 46 px. no keyboard height fits 190 px rows on a
768 px screen (it would need keyboard_height_ratio 1.36); the board's narrowest
column is 124 px on a 1366 px screen (10 letter columns, 11 control columns),
so this layout tops out at a 62 px calibration however tall it gets;
recalibrate to about 46 px, or use --layout paged (8 columns, up to 85 px)
```

The same two axes appear in the corner *"keys below spec"* note while the board
is under-sized. (That is a different indicator from the *"LOW ACCURACY"* drift
line below — one wants a bigger keyboard, the other a corrected aim.)
`--layout paged` swaps to the 8-column two-page board, whose bigger
keys clear NFR-2 at up to ~85 px of error on the same screen — slower to type
on, but far more forgiving. (That board has no room for a Fix aim key at 8
columns; its drift note points at Recalibrate instead.)

Options: `--layout qwerty-tall|paged|auto`, `--height-ratio`, `--dwell`,
`--no-cursor`, `--no-webcam`, `--camera N`.

## When typing will not select — `--debug-typing`

```bash
python main.py --debug-typing
```

Every 5 s (`--debug-interval`) it prints one line, and a summary on quit:

```
[typing]  5.0s  focus 12.3/s  resets: steal 25  fixdrop 0  grace 0  saved 3
                jitter x   6 y  45 px  spread x 201 y  69  fixating 100%  keys 0
```

| | |
|---|---|
| `focus /s` | how often the focused key changes. Healthy is under ~1.5/s; 10/s is flapping |
| `steal` | dwells lost because a neighbour took focus and the gaze did not come back in time |
| `saved` | steals where it *did* come back inside the grace, and the dwell carried on. Nothing was lost |
| `fixdrop` | dwells stalled because the fixation detector let go |
| `grace` | dwells that decayed away after the gaze left every key |
| `jitter x/y` | median wobble inside ~1/3 s buckets — the number that predicts steals. Compare against the key size |
| `spread x/y` | deviation across the whole window, **including moving between keys**. Large by design; not a fault |
| `fixating` | duty cycle of the I-DT detector |

Read `jitter` against the key size, not `spread`: a session moving across the
board reads ~350 px of spread on x while the actual wobble is 6 px, so the
window-wide figure names the wrong axis.

Three knobs, each falling back to config and then to the default, so an absent
flag changes nothing: `--hysteresis` (0.25), `--min-cutoff` (One-Euro, 1.0 —
lower is smoother and laggier), `--grace-ms` (200).

**All three are live.** Two of them used to be inert, which is what NFR-7 in the
spec measured and what the current rules fix:

* `--hysteresis` — the focused key is asked *first*, so its grown region really
  does decide ownership and a neighbour has to win the point from outside it.
  Bounded so that a key always owns its middle half: a focused Space (250 px)
  can never swallow the 124 px Backspace beside it, however wide the margin.
* `--grace-ms` — a dwell now survives *losing* its key, not just the board. The
  interrupted dwell is carried and decays over the grace, so one jittered frame
  across a row boundary costs a frame of decay rather than the whole second, and
  coming back resumes it. `--grace-ms 0` restores the old discard-on-steal rule.
* `--min-cutoff` — unchanged: it reduces the wobble itself rather than
  tolerating it.

A healthy session has `keys` ≈ 90% of dwell attempts, `steal` low and below
`saved`, `fixdrop` and `grace` near zero, `focus` under ~1.5/s, and `jitter y`
under about a quarter of the row height.

## Word prediction

Local, instant and offline (spec NFR-5) — nothing about what is typed leaves the
machine. Typing `he` offers `her here hello hey`; dwelling on one types the rest
of the word plus a space.

Ranking blends a bundled English frequency list with a **personal count** of the
words this user actually picks, saved to `~/.gazekey/user_words.json` (words and
counts only, never sentences). Pick a word a few times and it climbs above more
common English — which is the point for someone typing the same names every day.

`pyenchant` is deliberately not used: it is a spell checker, so it can say
whether a word exists but not which candidate is likelier, and its Windows
builds need external DLLs that often fail to resolve. Lookups take ~0.01 ms.

Hebrew stays in M6.

## Calibration

Calibration runs inside `main.py`; the pacing flags below work there too.

What happens, in order:

1. **Nine targets.** A steady dot appears. Nothing about it moves — no pulsing,
   no colour changes, no flashes, because anything animating near the fixation
   point makes it harder to hold a steady stare. The *only* feedback is a thin
   ring around the dot: **empty while it settles, filling while it measures**.
   A point that yields too little clean data repeats itself automatically (up
   to twice).
2. **Point-level repair.** After the first fit, any point that fits both 3×
   worse than the median *and* worse than 60 px is re-collected once and the
   model refitted — before validation, in every verdict band. One glanced-away
   target is enough to drag an otherwise 20-25 px session into MARGINAL, and
   waiting for the verdict to fail never catches that. The console says
   `point 2 fit poorly (283 px vs 77 px median) - re-collecting`. Three or more
   bad points is systemic rather than a glance, so it is reported instead of
   re-collected; whichever collection fits better is kept, and the pass runs at
   most once.
3. **Three fresh validation targets** at different positions.
4. **Results screen** with the measured error, the verdict, and the full
   diagnostics map (below).
5. **The keyboard.** (`--practice` inserts the aiming drill first; running
   `calibrate.py` instead ends at a free-form gaze-dot demo.)

Keys: `Space` continue/repeat · `R` calibrate again · `Esc` cancel.

### Where the dots go — the calibration region

A webcam has a roughly fixed angular error, so spreading nine dots over the
whole display spends most of that accuracy on parts of the screen nobody looks
at. **The nine dots and the three validation targets span the keyboard area**,
not the screen: on 1366×768 that is a 1366×570 region (the board plus 7.5% of
screen height above it), **74% of the display**.

What follows from that:

* **the reported error is an in-region error** — the accuracy that applies
  where the keys are, and the number NFR-2 is judged against;
* **everything you select by gaze moves inside it** — the YES/NO quit
  confirmation, the Fix aim dot and the practice targets are all placed in the
  region rather than at the centre of the screen. Startup warns if any key ends
  up outside;
* **outside the region the model extrapolates.** The gaze cursor is still drawn
  up there and is expected to be loose. Nothing is selected up there, so that
  is fine;
* the region is **saved with the calibration**, and the touch-up and in-place
  recalibration reuse it.

`--cal-region full` restores whole-screen calibration exactly, for comparison.
To A/B the two fairly, run both with `--practice` and compare **average aim
error** — the drill's targets are in-region either way, so that number is
measured identically in both. (The validation errors are *not* comparable: one
is measured over the keyboard, the other over the whole screen.)

> The margin is sized against the **dot hull**, not the region edge: dots sit
> at 10/90% of the region, so their hull is inset by a tenth of it. 7.5% lifts
> that hull to y 255, just above the top key row at 256 — at 5% the suggestion
> bar sat outside it and aiming there was extrapolated. The bottom row cannot
> be covered at any margin, since the board is flush with the screen edge:
> 57 px of overhang here, against 77 px for a whole-screen calibration.

### Head mode

The default assumes the user's head rests on a physical support, as it usually
does for the people this is built for. In that mode pose does not vary, so the
head-sweep stage is skipped entirely and the pose-compensation sensitivities
stay at zero — `compensate()` becomes the identity and calibration starts on
the first target.

| | |
|---|---|
| `--fixed-head` | head rests on a support — **the default** (`fixed_head` in config) |
| `--with-head-sweep` | head moves freely — run stage A first |

Use `--with-head-sweep` if the head is unsupported. That stage takes ~5 s of
slow head rotation while staring at a centre dot; if it is refused the screen
says why and `Space` repeats it. Without it, head movement wrecks accuracy
(the spec measures ~590 px of error against ~10 px), so it stays mandatory for
anyone whose head is not held still. Which mode ran is recorded in the
diagnostics line of every result.

### The setup check

| | |
|---|---|
| `--skip-setup-check` | skip the two-target camera check before calibration |
| `setup_check_min_hy_span` | the threshold it judges against (config, default **0.035**) |

Two dots at the top and bottom of the calibration area, ~5 s, measuring how far
the vertical iris ratio travels between them — the same `hy span` the
diagnostics print after a session, so the two numbers can be compared directly.
Below the threshold it says the camera looks too low and offers **Space** to
re-check, **Enter** to calibrate anyway and **Esc** to quit. The measurement is
printed either way:

```
[GazeKey] setup check PASS: hy span 0.051 (top 0.302, bottom 0.353, 30+30 samples, threshold 0.035)
```

### Pacing

| Flag | Default | |
|---|---|---|
| `--settle-ms` | 700 | countdown before a target starts measuring |
| `--collect-samples` | 45 | valid samples wanted per target |
| `--collect-max-s` | 4 | wall-clock cap per target, so a bad point cannot stall |
| `--slow` | — | preset: 1000 ms settle, 60 samples — first-time use and accessibility |

Collection ends as soon as it has enough *valid* samples, so blinks extend a
target rather than corrupting it. Defaults also live in
`~/.gazekey/config.json` (`calibration_settle_ms`, `calibration_collect_samples`,
`calibration_collect_max_s`); explicit flags beat `--slow`, which beats config.

### What a good result looks like

| Measured error | Verdict | What it means |
|---|---|---|
| ≤ 80 px | **PASS** — "Good" | Saved. The dot should sit inside the reference crosses. |
| 80–130 px | **MARGINAL** — "Usable" | Saved with a warning; plan on larger keys. |
| > 130 px | **FAIL** — "Too low" | Not saved. The two worst points are re-collected and revalidated once automatically before the verdict is final. |

On a 1920×1080 screen a good session lands around 30–60 px. In the demo the
dot should land within roughly a thumb's width of wherever you look, stay calm
while you stare, and keep up when you glance across the screen. Drift when you
move your head means the head sweep was too small — press `R` and give it a
wider rotation.

The measured number is printed to the console as well as shown on screen, e.g.
`[GazeKey] calibration PASS: 43.7 px  (saved to ...)`.

## Dwell and typing

Everything in spec Section 7, all configurable in `~/.gazekey/config.json`:

* dwell **1.0 s**, and it only advances while the fixation detector agrees the
  gaze is actually held — a travelling gaze never types;
* **2.0 s** extended dwell on Pause, Fix aim, Recalibrate, EN/HE and Quit;
* **hysteresis** 25% — the focused key keeps ownership over a region grown by a
  quarter on each side, and a neighbour has to win the point from outside it
  (bounded so every key always owns its own middle half);
* **grace** 200 ms — a dwell that loses its key decays over the grace instead of
  vanishing, whether the gaze left the board *or* a neighbour stole focus, and
  coming back inside that window resumes it;
* **refractory** 400 ms before the same key can repeat;
* a lost stream **freezes** the dwell rather than resetting it, so a blink
  mid-key costs a moment, not the whole selection;
* **Shift** latches for exactly one character;
* **Pause** keeps the tracking and all the visual feedback running but injects
  nothing. Pause, Fix aim, Recalibrate and Quit keep working while paused —
  none of them types, and a paused keyboard you cannot fix or close is a trap.

## Drift, and fixing it (M5)

Calibration decays during a session: the head settles deeper into its support,
the chair rolls back a centimetre, the light changes. The mapping is still the
right *shape*, it just sits a few dozen pixels off — so the fix is a
translation, not another nine points.

### The monitor — it reports, it never acts

Two signals, exactly as spec 5.5 describes:

* **every key activation is weak ground truth.** The user *was* looking at that
  key when it fired, so `key centre − gaze centroid` is one measurement of the
  offset; an exponentially-weighted mean of those (α = 0.3, over `char` keys
  only — Space is 250 px wide, its centre says nothing) is the estimate.
* **type-then-backspace within 3 s** counts a correction. Three inside 60 s is
  enough on its own.

Partial evidence from both adds up, and the flag is sticky between 0.7 and 1.0
so the corner indicator cannot blink on and off between keystrokes.

> **A limit worth stating.** A dwell only completes *inside* a key, so the
> measurable offset can never exceed half a key. Drift larger than that shows
> up as keys that stop activating at all — which is exactly why the correction
> count is the second signal and not a nicety.

When it trips, one line appears in the corner panel: *"LOW ACCURACY: aim has
drifted about 74 px — look at Fix aim to correct it."* That is the whole
intervention. **Nothing opens a screen, nothing recalibrates itself, and typing
is never interrupted** — deciding to stop mid-sentence is the user's call.

The offset that counts as a full unit of evidence defaults to the measured
calibration error (floor 50 px), which is the point at which being off starts
costing keystrokes; set `drift_offset_px` in config to pin it instead.

### Fix aim — the one-point touch-up

Dwell 2 s on **Fix aim** → one still dot in the middle of a dark screen →
**0.4 s settle + 2.0 s measure** → straight back to the keyboard with the
outcome in the corner. Under three seconds of interruption, well inside the ten
that decide whether the sentence survives.

The median of the collected gaze against the dot gives `(dx, dy)`, applied
through `CalibrationModel.apply_offset` — which touches **only `wx[0]` and
`wy[0]`**, the two constant terms. The curvature of the verified fit is never
altered. Two refusals, both reported rather than silently applied:

| | |
|---|---|
| fewer than 10 samples | *"could not see your eyes for long enough"* |
| offset > 25% of the screen diagonal | *"too far to be drift — recalibrate instead"* |

An accepted touch-up saves the corrected model, resets the smoothing filters
and clears the drift evidence (it was about the old model). Esc cancels and
changes nothing.

### Recalibrate — in place, not back to the start

**Recal.** runs the full 9-point session and returns **to the keyboard**, not to
any startup screen. The typed line, the word in progress, the language, the
page and the pause state are all carried across; the personal dictionary is
untouched. The board is rebuilt at whatever key size the *new* accuracy calls
for, so improving your calibration can change the layout under you — that is
the point. Esc mid-recalibration returns to typing on the old model rather than
ending the session.

### Smoothing and fixation (M3)

The gaze stream runs predictions through the verified `OneEuroFilter`
(`min_cutoff=1.0, beta=0.007`) and then the verified I-DT `FixationDetector`.
Every `GazeSample` carries `is_fixating`, plus two separate notions of validity:
`valid` (this frame had a usable face) and `stream_valid` (a usable gaze
position exists right now, fresh or held).

In the demo the dot is **blue while your gaze is travelling**, **green the
moment I-DT calls it a fixation** (with a faint ring at the dispersion
threshold), and **amber while a blink is being held**. When tracking is lost
the dot disappears rather than lying about a stale position.

**Invalid-frame policy (spec Section 6).** A blink or a lost face holds the last
position for up to `tracking_hold_ms` (300 ms) with `held=True` and the fixation
verdict **frozen** — so a blink mid-dwell freezes the dwell instead of throwing
it away. Past the hold the stream goes invalid: no position, no fixation, and
the UI says "tracking lost".

**Choosing the dispersion threshold.** The I-DT detector thresholds `Δx + Δy` —
the *sum* of both per-axis ranges in the window — so the budget is shared
between the axes. Residual wander while genuinely staring scales with
calibration accuracy, so the threshold is derived from the measured validation
error rather than fixed, at `DISPERSION_RATIO = 1.35` (`config.py`):

* too low → a real stare keeps breaking and dwell never completes;
* too high → a saccade reads as a fixation and keys fire in passing. Spec NFR-2
  sizes keys at ≥ 2× the expected error, so staying under 2× guarantees a
  deliberate move to a neighbouring key always breaks fixation.

At the **81 px** measured on this setup that gives the shipped default of
**110 px**, against a minimum key pitch of ~162 px. Override with
`--fixation-dispersion PX` or `fixation_dispersion_px` in config; the app prints
the active value and, if your accuracy has moved, what it would suggest instead.

### Target practice

**Opt-in**, on either entry point:

```bash
python main.py --practice
python main.py --practice --practice-targets 20
python calibrate.py --use-saved --practice
```

A dot appears somewhere **inside the calibration region** with a faint ring
around it — **that ring is
the hit radius**, so what counts as a hit is never a guess. Hold a fixated gaze
inside it for **0.8 s** and a bright arc fills around the ring exactly as the
dwell ring does on a key; when it completes the target pops with one short
expanding ring and an optional click, and the next one appears somewhere at
least a quarter of the screen diagonal away, so every round needs a real
saccade. Your live gaze dot stays visible throughout, in the same colours as
the demo. `Esc` skips at any time.

```
hit radius = max(60 px, validation error)
```

After ten targets the summary reports **hits**, **average aim error**,
**average error while holding** and **average time to hit**, on screen and in
the console.

The two error numbers answer different questions and it matters which you
quote:

| | |
|---|---|
| **average aim error** | every sample of every attempt, hits *and* misses, weighted by how long each took. This is the honest in-region accuracy, and the number to compare two calibrations with. |
| **average error while holding** | only gaze that had already landed inside the hit radius, over hits only. Always the smaller number — it says how steady a held stare is, not how well you aim. |

The drill follows the keyboard's rules on purpose — the hold only advances
while the fixation detector agrees, leaving the ring decays the hold over the
grace period rather than resetting it, and a lost stream freezes it — so what
you practise is exactly what typing depends on. The one difference is that
there is no hysteresis: the ring you see is exactly the region that counts,
because this screen is a measurement rather than a target to be helped along.

### Diagnostics (every run)

The results screen draws a miniature of your screen showing where every point
landed: the nine calibration points coloured by fit residual (green / amber /
red, ringed in amber if they had to repeat), and each validation target as a
hollow square with an **arrow to where the model predicted you were looking** —
the direction and length of that arrow is the error. Labels give the residual
in px and the `kept/collected` sample counts. Underneath: the hx/hy spans across
the nine points, the head-sweep pose range, and a one-line most-likely-cause.

The same evidence prints to the console as a table:

```
[GazeKey] calibration diagnostics
  head sweep: 90 samples, yaw 25.0 deg, pitch 16.5 deg
  calibration points   target        residual   kept/valid  retries
     1  ( 1728,  972)      7.2 px    40/45      0
     ...
  validation targets   target     ->  prediction      error
     1  (  576,  324)  ->  (   599,   310)     26.7 px
     ...
  feature spread: hx 0.409-0.592 (span 0.183)   hy 0.448-0.577 (span 0.129)
  verdict       : PASS 15.7 px
  most likely   : Consistent across the screen (best 4 px, worst 12 px) - no action needed.
```

The most-likely-cause line checks the evidence in order of how actionable it is:
too little head rotation in the sweep → too little iris movement between
targets (sit closer / camera at eye height) → samples being discarded or points
repeating (blinking, dim lighting) → a tight calibration fit but loose
validation (head shifted between stages) → one point far worse than the rest
(you looked away) → otherwise general noise.

## Run the Milestone 1 debug view

```bash
python debug_vision.py
```

Options: `--camera N` (default from config), `--width`, `--height`,
`--no-mirror`, `--backend auto|legacy|tasks`, `--startup-timeout SECONDS`.
`python main.py` currently launches the same view.

The window is created before the camera is polled, and startup aborts with an
explanatory message (exit code 2) if no frame arrives within the startup
timeout (default 5 s) — a busy or missing camera can never look like a hang.

Keys inside the window: `q`/`Esc` quit · `m` mirror · `l` landmark cloud ·
`r` reset the stability statistics.

### What you should see (M1 checkpoint)

* Your mirrored webcam image with landmarks drawn: green iris points, cyan eye
  corners and lids, magenta head-pose points, and an RGB axis gizmo at the nose.
* A HUD with `hx`, `hy`, `yaw`, `pitch`, `roll`, the EAR of each eye, the
  status flag (`VALID` / `BLINK` / `NO FACE`) and both frame rates.
* Three gauges at the bottom: `hx`, `hy`, and `hx (image axis)`.

To confirm the pipeline:

1. **Stability** — stare at one point. `hx`/`hy` should barely move; the `std`
   figures next to them should settle around `0.005` or less.
2. **Horizontal sweep** — look slowly left → right. The `hx` gauge and the
   `hx (image axis)` gauge slide across together, monotonically.
3. **Vertical sweep** — look up → down. The `hy` gauge slides down.
4. **Blink** — close your eyes; the status flips to `BLINK` and stays valid
   again when you open them.
5. **Head pose** — turn/nod your head; `yaw`/`pitch` change smoothly and the
   axis gizmo follows your face.
6. **Frame rate** — the pipeline figure should be ≥ 20 fps.
7. **Unplug the camera** — the window shows "camera disconnected —
   reconnecting…" and recovers when you plug it back in. It must not crash.

---

## Fixed — `hx` used to cancel horizontal gaze

As originally written (following the spec Section 4 formula), `gaze/features.py`
measured each eye between its *inner* and *outer* corner — along that eye's own
nasal→temporal axis — and averaged the two eyes.

Looking to one side abducts one eye (its ratio rises toward 1) and adducts the
other (its ratio falls toward 0). The two changes are almost equal and
opposite, so their average stayed nearly constant and `hx` carried virtually no
horizontal-gaze signal. Confirmed on a real face in the debug view, and exactly
zero on the symmetric synthetic eye in `tests/test_features.py`. `hy` was never
affected — both eyes move the same way vertically.

The fix (authorised as an explicit amendment to CLAUDE.md rule 2, and the only
change made to that file) measures the right eye along the image axis too:

```python
r = _eye_ratios(landmarks, R_IRIS, R_OUTER, R_INNER, R_UP, R_LO)   # was R_INNER, R_OUTER
```

Guarded by `tests/test_features.py`, which now asserts that a left→right sweep
moves `hx` monotonically across more than half its range, that both per-eye
ratios move together, and that `hx` matches an independently computed
image-axis ratio. The debug view keeps showing `per-eye hx R / L` and the
`hx (image axis)` gauge as a live cross-check.

---

## Layout

```
config.py              user config (~/.gazekey/config.json)
calibrate.py           M2 entry point: calibration session + gaze demo
debug_vision.py        M1 live debug view
main.py                entry point
vision/
  camera.py            threaded capture, reconnect on unplug
  face_tracker.py      MediaPipe -> FrameFeatures (both MediaPipe APIs)
  head_pose.py         solvePnP on a rigid 6-point subset -> yaw/pitch/roll
  model_assets.py      one-time model bundle download + checksum
  pipeline.py          worker thread -> queue of GazeSample
gaze/                  VERIFIED CORE — calibration math, do not rewrite
  features.py          FrameFeatures + iris ratios + blink gate
  calibration.py       two-stage calibration model
  smoothing.py         One-Euro filter, I-DT fixation detector
  calibration_session.py   session sequencing over that core (not math)
  drift.py             drift monitor + one-point touch-up (M5)
  region.py            the calibration region and its persistence
  setup_check.py       the 5 s camera check before calibration
ui/
  setup_check_screen.py    the two-target camera check
  calibration_screen.py    9 points / validation / results (+ head sweep)
  gaze_demo.py             M2 checkpoint gaze dot
  practice_screen.py       aiming drill + summary
  touchup_screen.py        the single-target Fix aim screen (M5)
  choice_screen.py         gaze questions (the Quit confirmation)
  keyboard_widget.py       key rendering, suggestions, typed preview, self-view
  overlay.py               always-on-top click-through keyboard window
  exit_button.py           the clickable X, in its own clickable window
interaction/
  layouts.py           key geometry, NFR-2 sizing, layout selection
  controller.py        dwell state machine (focus, hysteresis, grace, refractory)
  injector.py          pynput keystroke injection
  practice.py          aiming drill: targets, hold, hit/miss statistics
  prediction.py        offline word suggestions + personal frequency learning
  hotkey.py            global Ctrl+Alt+Q quit listener
tests/
```

## Decisions taken (spec ambiguities)

* **Flat package layout.** Spec Section 2 nests everything under `gazekey/`,
  but the verified core already lives at `gaze/` in the repository root and
  `tests/test_calibration_pipeline.py` imports it as `gaze.calibration`. The
  repository root *is* the package root; nothing was moved.
* **MediaPipe backend.** The spec names
  `mp.solutions.face_mesh(refine_landmarks=True)`. That legacy API no longer
  exists in current MediaPipe wheels (and in no Python 3.13 build) — only the
  Tasks API ships. `vision/face_tracker.py` therefore supports both and picks
  automatically:
  * `legacy` — `mp.solutions.face_mesh`, used when available;
  * `tasks` — `FaceLandmarker`, which needs the `face_landmarker.task` bundle.
    It embeds the same attention/iris mesh, so both produce 478 landmarks and
    feed the identical `extract_features`.
* **Model bundle.** The Tasks backend downloads `face_landmarker.task`
  (~3.7 MB) once into `~/.gazekey/models/` and verifies its SHA-256. It is the
  only thing GazeKey ever writes besides config and calibration.
* **Head pose.** `cv2.solvePnP` (SQPnP + LM refinement) on landmarks
  1/152/33/263/61/291, with `f ≈ image width` intrinsics. Sign conventions —
  yaw > 0 turning to the user's right, pitch > 0 looking up, roll > 0 tilting
  toward the user's left shoulder — are documented in `vision/head_pose.py`
  and pinned by tests, because changing them later would invalidate every
  saved calibration.
* **Mirroring.** Features and head pose are always computed on the raw
  (un-mirrored) frame; the debug view mirrors only what it draws.
* **Session sequencing lives outside the verified core.**
  `gaze/calibration_session.py` decides *which target is showing, when to
  collect and when to repeat*; every calculation it needs (pose compensation,
  outlier rejection, aggregation, the fit, the verdict, the worst-residual
  points) is called from `gaze/calibration.py`, which is untouched. Keeping the
  sequencing free of Qt and of the camera is what lets the whole session be
  tested against a synthetic eye.
* **Vision → UI handoff** uses the thread-safe queue option from spec Section
  3, not Qt signals: `vision/pipeline.py` has no Qt import, and the widgets
  drain it from a 16 ms `QTimer`. Same threading guarantee, testable headless.
* **Point collection window.** The spec says ~1 s / ~30 samples per target.
  Collection is driven by the *valid sample count* instead, with a wall-clock
  cap, so blinks extend a target rather than corrupting it — and the defaults
  are slower than the spec's (700 ms settle, 45 samples) because the spec pace
  felt rushed in practice. All three numbers are configurable, with a `--slow`
  preset for first-time and accessibility use.
* **Two kinds of "valid" on a sample.** `valid` is per-frame (face found, not
  blinking) and is what the calibration session counts; `stream_valid` is the
  spec's stream-level notion that survives a blink for 300 ms. Keeping them
  separate is what lets the hold policy freeze a dwell without the calibration
  session mistaking a held frame for a real measurement.
* **The gaze policy is a public method.** `GazePipeline.track(features)` does
  prediction → smoothing → fixation → hold and returns the sample; the worker
  thread just calls it. That makes every timing rule testable synchronously
  with no camera and no threads.
* **Paged alphabetical instead of QWERTY when QWERTY cannot fit.** The spec
  asks for QWERTY, and QWERTY is used whenever the geometry allows it. When a
  large error on a small screen makes a 10-column row impossible, an
  8-column alphabetical layout over two pages is used instead — alphabetical
  because a paged split destroys QWERTY's muscle memory anyway, and scanning
  for a letter is what actually costs time when selecting by gaze.
* **Hit regions are gapless.** Keys are drawn with a visual inset but their hit
  rectangles touch, so there are no dead zones between them. That is why the
  focused key is asked *first* in the hit test: asking the neighbours first
  handed them every boundary crossing and made the hysteresis margin inert
  between keys, which is what spec NFR-7 measured. Since the margin is a
  fraction of the *focused* key's own size, it is bounded by a fixed core — the
  middle half of every key belongs to that key whatever the margin is — so a
  wide Space cannot swallow the narrow Backspace beside it.
* **The suggestion bar is display-only until M6.** The three slots exist in the
  geometry per spec Section 8 but are not dwell targets, so they are exempt
  from NFR-2 sizing and drawn as a slim strip rather than eating a whole row.
* **The overlay covers the screen, not just the keyboard.** The window spans
  the display so the gaze cursor and status icons work everywhere, but it only
  *paints* the docked keyboard, and it is click-through and non-focusable, so
  the application behind stays both visible and usable.
* **The X lives in its own window.** Click-through is all-or-nothing per
  top-level window on Windows, so a child widget cannot opt back in. The exit
  button is therefore a separate tiny always-on-top window that accepts clicks
  but still refuses focus.
* **Gaze questions reuse the keyboard.** The quit confirmation is a `Keyboard`
  object of two enormous keys, driven by the real dwell controller and painted
  by the real key renderer — same timings, same look, nothing new to learn, and
  already covered by the controller's tests.
* **No startup question at all.** Asking "use the saved calibration?" put the
  hardest judgement call of the session on the user before they had any
  evidence to answer it with. Every launch calibrates instead; the saved file
  stays for the M5 touch-up and for `--use-saved` during development.
* **NFR-2 is judged on the minimum key dimension, over the real rectangles.**
  Not the width and not a nominal cell size — see *Key size, honestly* above.
  Height is what binds on a 1366×768 screen, and quoting only the width would
  send the user chasing a calibration target 16 px looser than the one that
  actually passes.
* **Drift evidence comes from `char` keys only.** Spec 5.5 says "every key
  activation", but Space is 250 px wide and the control keys are oddly shaped —
  their centres say almost nothing about where the user was looking, and
  including them would bias the estimate toward whatever those keys' geometry
  happens to be.
* **The drift indicator never acts.** No auto-recalibration, no dialog, no
  interruption: one line in the corner panel, and the user decides. Interrupting
  someone mid-sentence to offer help is worse than the drift.
* **The control row is one column denser than the letter rows.** Eleven slots
  instead of ten, so the M5 Fix aim key fits without shrinking Space (the most
  used key) or displacing Quit (an exit route). At the default height the
  narrower 124 px control keys still clear the 93 px the row height allows, so
  the binding axis is unchanged.
* **The default layout no longer auto-selects.** `--layout qwerty-tall` is the
  design asked for and is the default even where it misses NFR-2; `auto` keeps
  the older accuracy-driven selection between full QWERTY and the paged board.
* **Screen coordinates** are Qt logical pixels with high-DPI scaling enabled.
  Calibration targets, the saved model and the demo all use that same space,
  and the saved `screen_size` is what the reload check compares.

## Privacy

Video frames and landmarks stay in memory and are never written to disk. The
only files GazeKey writes are `~/.gazekey/config.json`,
`~/.gazekey/calibration.json` (coefficients, not imagery),
`~/.gazekey/user_words.json` (words and counts only, never sentences) and the
cached model bundle. No network access at runtime beyond the one-time model
download.

## Tests

```bash
pytest -q
```

No camera and no visible window required — the Qt screens are rendered
offscreen and the whole calibration session is driven by a synthetic eye. Note
that the Qt "offscreen" platform has no font database, so the UI tests check
on-screen copy through `text_lines()` / `status_lines()` (the same source the
painter draws from) and check geometry against the painted pixels.

## Milestones

| | | |
|---|---|---|
| M1 | vision pipeline + debug view | **done** |
| M2 | head-sweep + 9-point calibration + validation gate + persistence | **done** |
| M3 | One-Euro smoothing + fixation detection | **done** |
| M4 | keyboard overlay + dwell + keystroke injection | **done** |
| M5 | drift monitor + 1-point touch-up + in-place recalibration | **done** |
| — | region-scoped calibration + point-level repair | **done** |
| — | typing stability (NFR-7): sticky focus, dwell survives a steal | **done** |
| — | the 5-second setup check before calibration | **done** |
| M6 | Hebrew layout + RTL | next — English word prediction **done** early |
