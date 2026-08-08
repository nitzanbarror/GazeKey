# CLAUDE.md — Instructions for building GazeKey

## What this project is
GazeKey: a desktop virtual keyboard controlled by eye gaze via a standard webcam, for users with motor disabilities. Gaze is estimated from webcam video, keys are selected by dwell-time (staring ~1 s), and real OS keystrokes are injected into the focused application.

## Ground rules
1. **Build from scratch.** If any old code exists in this repo, ignore it — the previous implementation failed (calibration never worked). Do not copy or "fix" old code.
2. **The full technical spec is in `GAZEKEY_AGENT_SPEC.md`. It is the single source of truth.** `SPEC_AMENDMENTS.md` has been merged into it and deleted — there is no second document to reconcile against. Follow it exactly, especially Section 5 (Calibration).
   - The calibration math is ALREADY IMPLEMENTED AND TESTED in `gaze/features.py`, `gaze/calibration.py`, `gaze/smoothing.py` with passing tests in `tests/test_calibration_pipeline.py`. **Use these files as-is. Do not rewrite, simplify, or 'improve' them.** Everything else — the camera pipeline, the UI, the interaction layer, the injection, `gaze/calibration_session.py` and `gaze/drift.py` — is ordinary code wired around that verified core.
   - **Fixed-head mode is the default** (spec 5.1b): the user's head rests on a physical support, so the head-sweep step (Stage A) does not run and no pose compensation is applied. The earlier "head-sweep is mandatory, never skip it" rule applied to free-head use only. Free-head with the head sweep remains available behind `fixed_head: false` / `--with-head-sweep`, and must never be deleted — without it, head movement wrecks accuracy (~590 px of error versus ~10 px).
3. **Work milestone by milestone (Section 11 of the spec).** Complete one milestone, make its checkpoint runnable, then STOP and tell me how to test it before starting the next one.
4. **Every milestone must be runnable by me.** Give me the exact command to run and what I should see on screen.
5. **Write the tests in Section 12 as you go**, not at the end. Run `pytest` before declaring a milestone done.
6. **Fixed stack:** Python 3.10+, opencv-python, mediapipe, numpy, PyQt5, pynput. Do not add other heavy dependencies without asking. **No pyenchant, ever** — word prediction ships on a bundled offline frequency list plus personal counts (spec 8.4).
7. **Privacy:** never write video frames or landmarks to disk.
8. If something in the spec is ambiguous, choose the simplest option that satisfies the checkpoint and note the decision in the README — don't ask about every small detail, but DO ask before changing anything in Section 5.1 (the verified calibration core).
9. **When the spec and the code disagree, which one is wrong depends on what kind of disagreement it is.** Classify it before touching anything:
   - **Descriptive / implementation detail** — measured sizes and geometry, config defaults, timings, file and module structure, wording, counts. **The code wins**: fix the spec in the same change and tell me what you corrected. No need to ask first.
   - **Product decision** — user-facing flows and screens, default modes (e.g. fixed-head vs free-head), NFR thresholds and formulas, which keys exist and what they do, privacy rules. **Ask me first.** Do not change the code, and do not change the spec: bring me the conflict with your recommendation, and update both only after I approve. A product decision does not become correct by having been implemented.
   - **The verified core** (spec Section 5.1: `gaze/features.py`, `gaze/calibration.py`, `gaze/smoothing.py`) — untouchable either way, as in rules 2 and 8.

   If you are unsure which bucket something falls in, treat it as a product decision and ask.

## Milestones (spec Section 11)
- M1: Vision pipeline + live debug view — **done**
- M2: 9-point calibration UI over the verified core + validation gate + persistence + gaze-dot demo — **done**
- M3: One-Euro smoothing + fixation detection — **done**
- M4: Keyboard overlay + dwell state machine + keystroke injection — **done**
- M5: Drift monitor + 1-point touch-up ("Fix aim") + in-place recalibration + PAUSE/RECALIBRATE keys — **done**
- Region-scoped calibration (spec 5.6) + practice made opt-in — **done**. `python main.py` is calibrate → keyboard; the nine dots span the keyboard area rather than the whole screen, so the measured error is an in-region error and NFR-2 is judged against it. `--practice` runs the drill, `--cal-region full` restores whole-screen calibration for comparison.
- Typing stability (spec NFR-7) — **done, awaiting the after-measurement.** Sticky focus (the focused key's grown region really does decide ownership, bounded so a wide key cannot swallow a narrow neighbour) and a dwell that survives a steal by decaying over the grace instead of being zeroed. Both defaults unchanged (margin 0.25, grace 200 ms). Proposal 3 (per-axis hysteresis) is deferred until the measurement says whether it is still needed.
- The 5-second setup check before calibration (spec 5.1c) — **done**. Two dots measure the hy span; below 0.035 it says the camera is too low and offers a retry, calibrate-anyway, or quit. `--skip-setup-check` turns it off.
- M6: **Hebrew layout + RTL only** — next. (Word prediction was pulled forward and is already done, so it is not part of M6; Hebrew prediction is the same mechanism with a Hebrew frequency list if feasible, otherwise gracefully absent.)
- Then the two end-to-end scenarios, by gaze only: type "Hi" into WhatsApp Web, and "Dog" into a Google search box.

## Definition of done for the whole project
- Calibration validation error is measured and displayed, never assumed (≤ 80 px = pass).
- I can type a sentence into a browser using only my eyes, with ≥ ~90% correct key selections — **measured on the layout that satisfies NFR-2 for the calibration error actually achieved** (in practice the paged layout at ≤ ~81 px). Full QWERTY at webcam accuracy is expected to be usable but below this bar; that is an accepted trade-off, not a defect, and the app reports it rather than pretending.
- **NFR-2 (spec Section 14):** every selectable key is at least twice the measured error on **both** axes — `error ≤ min(key_width, key_height) / 2`, measured over the real key rectangles, against the **in-region** error (spec 5.6). Any shortfall is a warning that names both axes and the binding one, never a block.
- Every gaze-selectable target — keys, the Quit confirmation, the Fix aim dot, the practice drill — sits inside the calibrated region (spec 5.6).
- ≥ 20 FPS on CPU, UI never freezes, no crash when the camera disconnects.
- All four exit routes work (spec NFR-6): Ctrl+Alt+Q, the gaze Quit key, the clickable X, Ctrl+C.
- `pip install -r requirements.txt` + one command (`python main.py`) runs the app; README explains everything.
