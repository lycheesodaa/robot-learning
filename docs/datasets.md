# Dataset selection and training assessment

Sources inspected on 2026-10-05. Start with ordinary LIBERO competence and SafeLIBERO evaluation. Register other sources without automatically downloading their corpora.

| Resource | Training/evaluation role | Coverage and caveats |
| --- | --- | --- |
| [SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO) | Primary safety evaluation | VLSA/AEGIS benchmark: 16 tasks, 32 obstacle scenarios, 1,600 episodes. Released task definitions and initial states are not a labelled safe/unsafe demonstration corpus. Its published displacement-based collision proxy can miss stationary contacts; report that proxy separately from simulator contact events. |
| [LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety) / [demonstrations](https://huggingface.co/datasets/LIBERO-Safety/libero_safety) | First safety fine-tuning candidate; broader evaluation | Five suites: affordance, human interaction, tabletop avoidance, hand/object avoidance, semantic reasoning. Authors report 19.7K collision-free demonstrations. Inspect raw LeRobot metadata, actions, and episode counts; the Hub video viewer is not authoritative for trajectory counts. Safe-only demonstrations need unsafe rollout counterparts. |
| [MolmoAct2-LIBERO-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset) | Task adaptation and retention | LeRobot training data for the selected family. Ordinary success is not a safety label. The checkpoint has already seen LIBERO training tasks; report safety perturbation generalization rather than claiming all tasks are unseen. |
| [MolmoAct2 robotics collection](https://huggingface.co/collections/allenai/molmoact2-datasets) | SO-100/101 adaptation | Choose embodiment-matched demonstrations with compatible cameras, state and action conventions. Generic DROID/Bridge data do not directly match SO-101 joints. The SO checkpoint's sample comes from `Beegbrain/pick_lemon_and_drop_in_bowl`; this is an example, not a safety dataset. |
| [SafeVLA-Bench](https://safevla.org/) | Additional evaluation specifications | Excess contact, bystander disturbance, object instability and self-contact on LIBERO/RoboCasa. Check runnable artifact availability before depending on its evaluation wrappers; this is distinct from the NeurIPS SafeVLA project. |
| [Safety-CHORES / SafeVLA](https://safevla.github.io/) | Later generalization and constrained-learning study | Navigation plus manipulation, including fragile collections and dangerous equipment. Different embodiment and simulator mean substantial adaptation. |

## What training is needed?

These are engineering assessments, not empirically calibrated probabilities:

- **Training a foundation VLA from scratch: very unlikely to be needed.** Both selected embodiments have released checkpoints.
- **Fine-tuning before baseline LIBERO evaluation: low likelihood.** First establish ordinary task competence.
- **Probe fitting or estimating steering directions: expected.** This is small-scale fitting on activations, not necessarily updating policy weights.
- **SO-101 adaptation: moderate to high likelihood.** Your scene, cameras, calibration and manipulation tasks can differ from training.
- **Safety fine-tuning: conditional.** First test frozen-model steering. If you train safety into the model, keep that as a separate condition rather than calling it discovery of pre-existing mechanisms.
- **New labelled rollout collection: highly likely.** Existing demonstrations do not supply all matched safety contrasts and causal controls.

The [official MolmoAct2 training guide](https://huggingface.co/docs/lerobot/main/en/molmoact2) supports LoRA and frozen-VLM training; its action expert is trained in both cases. Prefer those routes before full fine-tuning. A negative finding about localization is a valid research outcome.

## Data construction protocol

1. First record frozen-policy ordinary task success. Poor manipulation must not be interpreted as a safety-specific deficit.
2. Pair controlled scenes/instructions by task, layout and initial simulator state, changing one safety-relevant variable. Keep instruction-only pairs distinct from pairs changing visual geometry.
3. Collect actual simulator contacts, prohibited-region crossings, and task success independently. Define fragile-object proxies explicitly (e.g. contact impulse, drop height); do not call these validated physical breakage measurements.
4. Store pair IDs, task/layout IDs, initial-state identifiers, controller settings, seeds, and all proposed/executed actions. Estimate proposed risk by counterfactual simulation when available; otherwise leave it unknown, not zero.
5. Assign whole task/layout groups and all members of a pair to discovery, selection or final evaluation. Fit neurons/directions on discovery only; tune intervention strength on selection only.
6. Compare four conditions with matched seeds. Include zero-strength, random layer/neuron, and ordinary-task controls. Log latency and activation selection alongside safety metrics.
7. Separate benchmark-specific collision proxies from physical contact events. Success and safety are separate outcomes; stopping every task is not a successful policy.

## Selective download examples

Run from the activated project environment. These commands are opt-in and do not download data during model setup. Resolve and retain the printed full dataset revision before downloading.

```bash
python scripts/download_dataset.py THURCSCT/SafeLIBERO --list
python scripts/download_dataset.py LIBERO-Safety/libero_safety --list
python scripts/download_dataset.py allenai/MolmoAct2-LIBERO-Dataset --list
# Copy a revision from the listing, then request only the required files:
python scripts/download_dataset.py THURCSCT/SafeLIBERO --revision FULL_SHA --include '*.bddl' '*.pruned_init'
```

Inspect dataset licenses and raw metadata in the listing before use. Patterns that match no files fail. Record which files were selected; full training corpora are deliberately not fetched by default.
