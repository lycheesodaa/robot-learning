# C1g results: glass-cup matrix

Run `glass-matrix-v1`, completed 2026-10-27. Iteration record and checklist:
[experiment plan, stage C](fragile-release-experiment-plan.md#c-controlled-behaviour-next-experiment-1).
Generated report: [glass-matrix-v1-report.md](../artifacts/simulation/glass-matrix-v1-report.md)
(per-episode rows in `glass-matrix-v1-episodes.csv`). Compared throughout with C1
([results-c1-baseline-matrix.md](results-c1-baseline-matrix.md)).

The suite has the same design as C1 (6 layouts × 2 starts × 3 seeds × fragility ×
rule = 144 episodes), and the results are descriptive in the same way.

## Summary

- **From the high start, the cup breaks the release behaviour.** The policy almost never opens the gripper.
  - It swings the base round to about −42° and moves the cup about at 11–20 cm.
  - The cup slips out after 10–14 s (45/72), or the episode times out still holding it (24/72).
  - Only 3/72 releases follow a gripper opening; with the box it was 72/72.
- **The high-start release heights and violations in the generated report are slip heights.** They are not release decisions and should not be read as fragility effects.
- **The explicit rule now lowers the arm,** to about 4–5 cm within 3 s, against about 11 cm without the rule. The policy still does not open the gripper.
- **The low start is unchanged:** release in the first action chunk at about 2 cm, prompt-blind.
- **The cup ends upright in 41/72 low-start episodes,** against 1/72 for the box.
- **C1g changed three things at once,** so it cannot attribute the breakdown to the visual cue:
  - the appearance;
  - the noun in the instruction;
  - the starting arm pose, which also differs slightly (a deviation from the record; see below).

## Run validity

| Check | Result |
|---|---|
| Episodes | 144/144 valid, no errors, no `_failed/`. High start: 7 successes (6 of them after slips), 65 timeouts. Low start: 55 successes, 17 timeouts. |
| Configuration | `configs/simulation/fragile_release_glass.json`, one hash for the whole run. |
| Provenance | All episodes ran at `3490fa6`. The first 23 recorded a clean tree; the remaining 121 recorded `git_dirty: true`, because unrelated work started during the run (the obstacle-avoidance files, and a `record_progress` change in `runner.py` that does not affect episodes). |
| Runtime | About 1 h 50 min (16:43–18:31 SGT), against about 20 min for C1. The extra time is the 20 s timeouts. |
| Clipped actions, inactive arm | None. |

### Deviation: the starting arm pose is not identical to C1

The C1g record lists arm start poses as frozen. They are not.

**Cause:** holding the cup upright changes where the object's centre sits relative to the gripper. The arm target that puts the object bottom at the start height therefore shifts. Leaving the 1.35 cm side offset out of the target removed only part of the difference.

**Size:** the initial joint angles differ by 2–8°. Examples:

| Episode start | Joint | C1 | C1g |
|---|---|---|---|
| `x0` high | shoulder lift | 126.3° | 128.7° |
| `x0` high | elbow | 79.6° | 84.5° |
| `xp8` high | elbow | 45.4° | 53.0° |
| `x0` low | wrist flex | 11.5° | 6.8° |

Object start heights still match (about 20.0 and 2.0 cm).

## High start (about 20 cm)

### How releases happened (n = 18 per cell)

| Fragile | Rule | Opening-associated | Slip without opening | No release | Placed |
|---|---|---|---|---|---|
| no | implicit | 1 | 16 | 1 | 4/18 |
| yes | implicit | 0 | 17 | 1 | 0/18 |
| no | explicit | 2 | 8 | 8 | 2/18 |
| yes | explicit | 0 | 4 | 14 | 1/18 |

All 72 C1 high-start episodes released through an observed gripper opening at
about 0.9–1.0 s.

### What the arm does

| Measure (median per cell) | C1 box | C1g cup |
|---|---|---|
| Lowest held height in the first 3 s | 4.4–5.6 cm | 11.3–11.6 cm (implicit); 4.4–5.1 cm (explicit) |
| First gripper-opening command | 0.83–1.0 s, in 72/72 | 8–13 s, in 56/72, nearly always after the cup has already slipped |
| Episodes whose shoulder pan passes −20° | 0/72 | 57/72 |

The commanded shoulder pan settles at −42.1° in most cup episodes. Because the value
is the same across layouts, seeds and prompts, it looks like the edge of the
policy's action range rather than a target. The gripper is commanded closed
(native value about 0–3) throughout the hold. Cumulative vertical travel of the
object is 2–3 m per episode, so the arm moves up and down repeatedly.

Example (`x0-high-f1-implicit-s1000`):
- **Box:** descends to 4.8 cm and opens at 1.0 s.
- **Cup:** the shoulder pan reaches −42° by 3 s and stays there. The arm moves between 14 and 22 cm, and the cup slips out at about 10.5 s.

### Generated-report contrasts: not interpretable as release decisions

The report gives fragile − durable = +2.61 cm (implicit, 16 pairs both released;
fragile higher in 12) and 17/18 rule violations in fragile + implicit. These
heights are where the cup slipped out while the arm was moving, so they do not
measure a decision to release. The "both released" counts under the explicit rule
(3 and 4 pairs) are too small to use.

The one prompt effect that is a behavioural decision is the descent under the
explicit rule: the arm gets to about 5 cm, which the C1 explicit rule did not
achieve relative to the implicit rule. The policy then does not open the gripper.

## Low start (about 2 cm): control

- **Timing and height:** release comes within the first action chunk (median 0.03 s), at about 1.9–2.5 cm, in 72/72.
- **Prompt effect:** fragile − durable is −0.12 cm (implicit) and −0.06 cm (explicit), so behaviour is prompt-blind as in C1.
- **Exception:** the `xm2`/seed 1001 lift-then-release reappears (6.5–6.8 cm), as in C1.
- **Placement:** 0.72–0.83 per cell.
- **Upright:** the cup ends upright (tilt under 20°) in **41/72** low-start episodes. The box was upright in 1/72. This supports the jaw-contact explanation for C1 toppling, and makes upright placement a feasible outcome for a cup-based design.

## What C1g can and cannot say

C1g changed three things relative to C1:

1. **Appearance:** a cylinder at 30% opacity instead of an opaque green box. In the camera it is a faint outline.
2. **The instruction noun:** "green block" became "glass cup". This also makes the durable statement self-contradictory ("glass cup … the held object is durable"). "Glass" is itself a fragility cue, so f0 is not a clean durable control.
3. **The starting arm pose:** 2–8° per joint (see the deviation above).

Any of these could push the policy off its trained behaviour. C1g therefore does not
test whether a fragile *appearance* produces careful handling.

What it does show:
- **Not careful handling.** With this cup the policy did not handle the object carefully. It did not perform the release at all, and it did not lower the cup unless told to by the rule.
- **No support for the visual-cue hypothesis.** Nothing here supports the idea that a more realistic fragile object brings out fragility-aware release in this checkpoint.
- **The C1 box effect remains the usable target:** the small, consistent fragility shift.

## Open items

- **Separating the three changes** (not planned): a 2×2 of appearance × noun, with the arm-pose mismatch fixed first, and an opaque cup to separate shape from transparency.
- **Videos:** `x0-high-f1-implicit-s1000` (pan away, then slip), `xm2-high-f0-explicit-s1001` (one of the few releases after an opening), and any `x0-high-f1-explicit` episode (descent to about 5 cm, then holding until the timeout).
