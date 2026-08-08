# How to run this with Claude Code

## Setup (one time)
1. Create a new empty folder for the project, e.g. `gazekey/`.
2. Put these two files inside it:
   - `CLAUDE.md`  (the instructions — Claude Code reads this automatically)
   - `GAZEKEY_AGENT_SPEC.md`  (the full technical spec)
3. Open a terminal in that folder and run `claude`.

## Kickoff prompt (paste this as your first message)

```
Read CLAUDE.md and GAZEKEY_AGENT_SPEC.md fully before writing any code.

Then implement Milestone 1 only (vision pipeline + live debug view, spec Section 11).
Set up the repo structure from spec Section 2, requirements.txt, and the M1 code + its tests.
When done: run pytest, then give me the exact command to run the debug view and tell me
what I should check on screen to confirm M1 works (per the M1 checkpoint in the spec).
Do not start Milestone 2.
```

## After each milestone
Test the checkpoint yourself with your webcam. Then continue with:

```
M1 checkpoint confirmed / here is what I saw: <describe any problem>.
Proceed to Milestone 2 (calibration). Follow spec Section 5 exactly — 9-point session,
outlier rejection, ridge polynomial fit, and the validation gate with the 80/130 px thresholds.
When done, give me the run command and what a good validation result looks like.
```

Repeat the same pattern for M3–M6 (one milestone per request).

## If calibration accuracy is bad at M2
Paste this instead of moving on:

```
Validation error was <X> px, above the pass threshold. Do not proceed to M3.
Add a diagnostics mode to the calibration screen that shows, for each of the 9 points:
number of valid samples, number rejected as outliers, and the fit residual in px.
Also plot/print hx,hy per point so we can see if the features are stable.
Then suggest what's wrong (lighting? blink threshold? head pose?) and fix it.
```

## Tips
- Sit ~60 cm from the camera, face well lit from the front, camera at eye height. Bad lighting is the #1 cause of bad calibration — no code fixes that.
- Ask Claude Code to commit after every milestone: "commit with message 'M2: calibration passing, validation error 54px'".
- If it ever tries to simplify the calibration (fewer points, linear fit, skipping validation) — say no and point it back to spec Section 5.
