# C1n results: word-swap control

Runs `swap-{fragile,heavy,rough,old,blue,zorpy}-v1`, completed 2026-10-28 (00:00 SGT).
Iteration record: [experiment plan, stage C](fragile-release-experiment-plan.md#c-controlled-behaviour-next-experiment-1).
Generated reports: `artifacts/simulation/swap-*-v1-report.md` (per-episode rows in
`swap-*-v1-episodes.csv`). Configs: `configs/simulation/prompt_swap/`.

## Question

C1 found fragile releases 0.52 cm lower than durable (implicit rule, high start;
14/18 pairs). Is that shift about the meaning of "fragile"? Or does any one-word
change to the object statement move the release by a similar amount?

## Design

- **Prompt:** C1's high start and implicit rule. Only the object statement changes: "Place the held green block on the receiving surface. The held object is **X**. Complete the placement."
- **Size:** 6 layouts × 3 seeds × 2 words = 36 episodes per pair.
- **Hardware:** each pair ran on one GPU (two chains of three suites).
- **Pairing:** the first word in each pair is labelled `f1` by the suite. For neutral pairs that label is arbitrary.

## Run validity

| Check | Result |
|---|---|
| Episodes | 216/216 valid, no errors. 213 releases, all after an observed gripper opening; one `rough` episode did not release. 13 timeouts in total. |
| Anchor reproduces C1 | All 36 fragile/durable episodes have release height, release time and termination identical to C1. 34/36 are byte-identical. In two, actions diverge only after the release (at 1.33 s and 7.2 s), most likely GPU non-determinism. |
| Provenance | All at `20cc6ff` with `git_dirty: true`. The uncommitted changes are the prompt-swap configs, docs and the unrelated obstacle work. |
| Runtime | About 22 min wall time for the six suites. |

## Result: fragile/durable is not special

### Paired contrasts (same layout and seed; first word − second word)

| Pair | Median difference [range] | First lower | Sign test | Absolute median |
|---|---|---|---|---|
| **fragile − durable** | **−0.52 cm [−1.61, +1.08]** | **14/18** | **p ≈ 0.03** | **0.52 cm** |
| heavy − light | −0.48 cm [−1.51, +1.09] | 16/18 | p ≈ 0.001 | 0.48 cm |
| rough − smooth | +1.02 cm [+0.30, +2.46] | 0/17 | p < 0.001 | 1.02 cm |
| old − new | +0.23 cm [−2.23, +2.27] | 6/18 | p ≈ 0.24 | 0.23 cm |
| blue − green | −0.76 cm [−2.43, +0.57] | 16/18 | p ≈ 0.001 | 0.76 cm |
| zorpy − blicky | +0.35 cm [−1.38, +1.33] | 7/18 | p ≈ 0.48 | 0.35 cm |

Three of the five neutral pairs shift the release as consistently as fragile/durable, or more so. Two of them shift it by a larger amount:
- **rough/smooth:** "smooth" is lower in 17/17 pairs, by a median of 1.02 cm;
- **blue/green:** "blue" is lower in 16/18, by 0.76 cm.

Fragile/durable sits in the middle of the neutral range on both size and consistency.

### Median release height by word (n = 18 each; 17 for "rough")

| Word | Median [range] | ≤ 5 cm |
|---|---|---|
| smooth | 4.28 cm [3.72–6.05] | 14/18 |
| heavy | 4.51 cm [3.71–6.62] | 13/18 |
| blicky (nonsense) | 4.57 cm [2.92–6.85] | 13/18 |
| **fragile** | **4.70 cm [3.94–6.09]** | **13/18** |
| zorpy (nonsense) | 4.83 cm [3.55–6.98] | 12/18 |
| blue | 4.84 cm [3.57–5.80] | 10/18 |
| light | 5.05 cm [4.06–7.42] | 7/18 |
| **durable** | **5.10 cm [3.41–6.70]** | **6/18** |
| new | 5.36 cm [3.58–7.17] | 6/18 |
| rough | 5.55 cm [4.02–7.25] | 2/17 |
| old | 5.55 cm [4.72–7.18] | 4/18 |
| green | 5.63 cm [4.24–7.60] | 5/18 |

- **The single word sets the release height** somewhere between about 4.3 and 5.6 cm, a spread of about 1.3 cm.
- **"Fragile" ranks fourth of twelve.** Both nonsense words and "heavy" release about as low, and "smooth" releases lower.
- **Release time is about 0.9–1.1 s for every word,** with no consistent pattern.

## Interpretation

- **C1's fragility shift is not evidence that the policy uses the meaning of "fragile".**
  - The release height depends on which word is in the object statement, consistently and reproducibly.
  - The size of that dependence is the same with or without fragility meaning.
  - In C1's paired design the fragile/durable shift looked meaningful because nothing measured how much an arbitrary swap moves the release.
- **This is a policy-level null for semantic fragility handling** in this checkpoint, this scene and this prompt template. The word changes the behaviour; its meaning, as far as these words can show, does not.
- **Limit of the design: one word per meaning.** Each meaning has a single word, so word identity and meaning cannot be separated within a pair. A test with several synonyms per meaning would be needed to rule out a small semantic component within the word-level noise:
  - fragile side: fragile, delicate, breakable, glass;
  - robust side: durable, sturdy, unbreakable, solid.
  - Treat word as a random effect and ask whether meaning explains variance beyond it.
- **Not tested here:** the explicit rule, the low start, other prompt templates and other objects.

## Implications for later stages

- **E/F (suppression, patching, steering on the fragile − durable difference):**
  - A fragile − durable activation difference would encode "a different word", not a fragility concept that drives release.
  - Amplifying it is likely to move the release by an amount comparable to any word direction.
  - Steering on it would be hard to interpret without word-swap directions as controls.
- **D (probing) is still informative,** and the question becomes sharper: is fragility linearly decodable from the activations even though the action does not use it? "Represented but not used" is a coherent and testable claim.
- **A behaviour to interpret now has to be created,** by targeted fine-tuning or LoRA, rather than found in the pretrained policy.
