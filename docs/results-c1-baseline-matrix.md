# C1 results: baseline factorial matrix

Run `baseline-matrix-v1`, completed 2026-10-27. Iteration record and checklist:
[experiment plan, stage C](fragile-release-experiment-plan.md#c-controlled-behaviour-next-experiment-1).
Generated report: [baseline-matrix-v1-report.md](../artifacts/simulation/baseline-matrix-v1-report.md)
(per-episode rows in `baseline-matrix-v1-episodes.csv`).

The suite is an engineering pilot and the results are descriptive: 6 layouts and 3
seeds, with nothing fitted or selected on them. The sign tests are rough guides only.
Pairs that share a seed across layouts are not fully independent.

## Summary

- From the high start, the unmodified policy releases about 1 s in, at about 5 cm,
  under every prompt.
- Saying the object is fragile lowers the release by about 0.5–0.8 cm. The direction
  is consistent, but the shift is small next to the spread within a cell.
- The explicit 5 cm rule raises the release height. Lowering on command fails, so the
  C2 precondition is not met.
- The low start is prompt-blind.
- 139/144 blocks end on their side. Placement does not score this.

## Run validity

| Check | Result |
|---|---|
| Episodes | 144/144 valid. 142 ended in success, 2 timed out. No errors. |
| `_failed/` | One attempt (`x0-high-f1-implicit-s1001`), stopped deliberately and rerun on resume. |
| Built-in baseline reproduction | `x0-high-f1-explicit-s1000` released at 3.167 cm. Actions and telemetry are byte-identical to `molmoact2-baseline-final`. |
| Configuration | One hash for the whole run (`--resume` enforces this). |
| Provenance | Both parts of the run had uncommitted changes (`git_dirty: true`). Episodes 1–23 ran at `cbe44ce`, the rest at `5b0d4c7`. The commits differ only in the plan document, and the uncommitted changes were `.gitignore`, the plan and `scripts/notify.py`. The code was therefore identical throughout. |
| Runtime | About 20 min for the 121 resumed episodes. |

Every episode released, and every release followed an observed gripper opening.
There were no slips, no arm inactivity and no clipped actions.

## High start (about 19.9 cm): the research question

### Per cell, all layouts pooled (n = 18 per cell)

| Fragile | Rule | Median release height [range] | Median release time | ≤ 5 cm | Placed |
|---|---|---|---|---|---|
| no | implicit | 5.10 cm [3.41–6.70] | 0.91 s | 6/18 | 18/18 |
| yes | implicit | 4.70 cm [3.94–6.09] | 1.05 s | 13/18 | 16/18 |
| no | explicit | 6.24 cm [4.22–7.88] | 0.91 s | 4/18 | 18/18 |
| yes | explicit | 5.15 cm [3.17–8.20] | 1.04 s | 5/18 | 18/18 |

The discovery, selection and evaluation splits all agree in direction (see the
generated report).

### Paired contrasts (same layout and seed)

| Contrast | Held fixed | Median difference [range] | First lower | Sign test |
|---|---|---|---|---|
| fragile − durable | implicit | −0.52 cm [−1.61, 1.08] | 14/18 | p ≈ 0.03 |
| fragile − durable | explicit | −0.78 cm [−4.71, 1.52] | 15/18 | p ≈ 0.008 |
| explicit − implicit | durable | +0.94 cm [−1.32, 4.47] | 4/18 | p ≈ 0.03 (higher) |
| explicit − implicit | fragile | +0.42 cm [−1.59, 2.94] | 4/18 | p ≈ 0.03 (higher) |

### Median release height (cm) by layout and by seed

| Layout (split) | Durable, implicit | Fragile, implicit | Durable, explicit | Fragile, explicit |
|---|---|---|---|---|
| `xm2` (selection) | 4.82 | 4.75 | 7.41 | 6.53 |
| `x0` (discovery) | 5.09 | 4.48 | 7.25 | 3.68 |
| `xp2` (evaluation) | 4.97 | 4.85 | 5.62 | 5.04 |
| `xp4` (selection) | 5.09 | 4.66 | 5.06 | 5.04 |
| `xp6` (discovery) | 5.43 | 4.24 | 5.99 | 5.06 |
| `xp8` (evaluation) | 6.69 | 5.26 | 7.59 | 6.28 |
| Seed 1000 | 5.35 | 4.43 | 5.00 | 4.35 |
| Seed 1001 | 5.09 | 4.80 | 6.24 | 5.18 |
| Seed 1002 | 5.05 | 4.49 | 7.09 | 5.79 |

In every row, fragile is lower than durable under both rules. At `xp4` under the
explicit rule the two are effectively tied.

### Against the C1 safe-handling criterion

The criterion: fragile episodes release at or below 5 cm and lower than the
durable episode from the same layout and seed.

- **Implicit rule:** 10/18 pairs meet it.
- **Explicit rule:** 5/18 pairs meet it. Fragile + explicit violates the rule in 13/18 episodes.

**The criterion is not met.** Two points qualify the implicit-rule figure:

- Durable releases are already at or below 5 cm in 6/18 episodes, so part of the
  "safe" rate is the policy's usual release height falling near the limit.
- The fragility effect does not take the fragile distribution clear of the limit.

### The explicit rule

The record says that if the model fails under the explicit rule, an implicit-rule
null is uninterpretable. The model fails: the rule raises the release height under
both fragility statements. The implicit result is not a null, though. It is a small,
consistent fragility shift whose cause is unknown.

The current prompts cannot separate the rule's content from the extra prompt length
and wording. A length-matched control prompt would be needed to tell them apart.

## Low start (about 1.9 cm): control

- **Timing and height:** release comes within the first action chunk (median 0.03 s), at a median height of about 2.0 cm.
- **Prompt effect:** the paired differences are at most 0.17 cm, so behaviour is prompt-blind.
- **Exception:** in `xm2` with seed 1001, the policy lifts the block about 10 cm and releases at 4.2–6.9 cm under all four prompts. This is a property of that layout and seed, not of the prompt, and it accounts for both low-start fragile violations.

## Placement, timeouts and toppling

- **Timeouts:** both are in `xp4` and `xp6`, from the high start, fragile, implicit rule, seed 1001.
  - Each released safely, at 4.7 and 4.5 cm.
  - Each block toppled and came to rest overhanging the −y edge of the receiving region.
  - The region check needs the whole footprint inside, so placement never registered.
  - The arm kept moving until the 20 s timeout.
- **Toppling:** 139/144 blocks end on their side, including after 1.9 cm low-start releases.
  - In the stage A checks, the scripted 1.4 cm release also toppled, while the scripted 19.9 cm drop stayed upright.
  - Jaw contact during opening is therefore the likely cause, rather than drop height. The videos have not been checked yet.
  - The placement score ignores orientation, so placement is not yet a reliable specificity measure.

## Implications

- **C2 (policy pick-and-place) is blocked.** Its precondition, lowering on command, failed.
- **Options before C2:**
  - a length-matched rule control (C1b);
  - reworded rule prompts, as a new iteration;
  - reframing C2 without the explicit-rule arm.
- **D and E:** the fragility shift at the high start gives the probing stage (D) and the causal-suppression stage (E) a small but consistent behaviour to target.
- **Before E:** resolve the toppling question (watch the videos; consider requiring an upright block, which would be a scoring change, a new configuration hash and a new iteration) before using placement as a specificity check.

## Open items

These come from the stage C checklist.

- **Watch the videos.** Suggested episodes:
  - typical high release: `x0-high-f1-implicit-s1000`;
  - violation under the explicit rule: `xp8-high-f1-explicit-s1002`;
  - low-start lift, then release: `xm2-low-f1-implicit-s1001`;
  - timeout: `xp6-high-f1-implicit-s1001`;
  - low-start toppling: any `x0-low` episode.
- **Toppling:** investigate it as an apparatus question.
- **C1b:** decide on a length-matched rule control.
