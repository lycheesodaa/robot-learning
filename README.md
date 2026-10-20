# MolmoAct2 safety mechanisms

Research scaffold for the accompanying proposal, covering MolmoAct2-LIBERO and MolmoAct2-SO100_101. Compare vanilla, internal steering, external constraints, and combined safety. Dataset recommendations and training expectations are in [docs/datasets.md](docs/datasets.md).

## Quick start

From this directory:

```bash
conda activate "$PWD/.envs/policy"
# Recreate if needed:
# CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/policy" -f configs/environment-policy.yml
python scripts/download_models.py
python scripts/download_models.py --verify-only
python scripts/diagnose.py
python scripts/check_model_metadata.py
pytest -q
# After CUDA is available:
python scripts/smoke_inference.py --model libero
python scripts/smoke_inference.py --model so101
# CPU sample validation (slower; this machine has enough host RAM):
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python scripts/smoke_inference.py --model libero --device cpu
# Lower-memory mode; record this precision choice in results:
python scripts/smoke_inference.py --model libero --dtype bfloat16
```

Complete checkpoint revisions are pinned in `configs/models.json`. Snapshots include weights, tokenizer, processors, custom Python model code, normalization metadata, and official sample images. Loading executes that pinned custom code with `trust_remote_code=True`. Downloads resume through Hugging Face's local cache; `--verify-only` checks every file size, weight SHA256, shard references and normalization tags without network access. A completion manifest is written only after verification.

The two snapshots require approximately 43.6 GB. Keep additional room for environments, recordings and activations. All generated large files and environments are ignored by Git. Snapshot manifests live beside the weights. The proposal PDF is preserved.

## SO-101 single input-output demonstration

Run from the project directory in a host terminal with GPU access:

```bash
.envs/policy/bin/python demo_so101.py --device cuda:0
```

Open [the generated report](artifacts/so101_demo/report.html) to see the two official sample camera images, language instruction, six-dimensional robot state, and all 30 predicted action steps. The script also prints the first five steps and saves exact inputs, full output/provenance (`prediction.json`), and `actions.csv`. It makes exactly one prediction call and never connects to a robot.

Use `--device cuda:1` for the other GPU, `--device cpu` without GPU access, or `--instruction "..."` to try another instruction with the same recorded scene. Outputs go to `artifacts/so101_demo/` by default; use `--output-dir` to preserve separate runs. Values use the checkpoint's native joint/gripper convention and are not calibrated motor commands.

**Host validation on 2026-10-19:** driver 580.178.04, two NVIDIA RTX 6000 Ada GPUs (49140 MiB each), PyTorch CUDA available. The FP32 GPU demo returned finite `[30, 6]` actions: 11.13 s model load, 1.06 s first inference without warm-up, and 21.81 GiB peak allocated GPU memory. These are one-run measurements, not a latency benchmark. A restricted sandbox still reports no CUDA devices; that does not indicate a failed host driver. Diagnostics are saved in `artifacts/diagnostics-host-2026-10-19.json`.

## Implemented and scaffolded components

| Component | Status |
| --- | --- |
| Immutable snapshot download and integrity verification | Implemented |
| Local-only official sample inference / recorded-observation predictions | Implemented; both CPU samples verified, SO-101 GPU demo verified on 2026-10-19 |
| Action/observation contracts and local HTTP inference server | Implemented |
| Scoped capture, ablation, scaling, steering, activation patching | Implemented for explicitly selected tensor-output modules |
| Rollout records, safety/task metrics, split-leakage checks | Implemented |
| Four experiment configurations | Provided; not a completed experimental runner |
| LIBERO/SafeLIBERO rollout adapters | Explicit research stubs |
| Workspace/obstacle/forbidden-region geometric filter | Explicit research stub; requires calibrated dynamics and kinematics |
| SO-101 physical actuation | Explicit stub; recorded-input inference only |
| Probe fitting, safety-neuron discovery, paired scene generation | Research work still to implement |

Geometric filtering never silently passes actions when unavailable. `checked_filter` raises `ConstraintFailure`; a future runner must stop execution on that exception. An initial geometric baseline is not a certified CBF controller or a reproduction of AEGIS.

## Research workflow

1. Run the SO-101 GPU demo above in a GPU-visible host process. LIBERO has CPU sample validation; its GPU sample remains to be checked.
2. Integrate one seeded ordinary LIBERO rollout with 50 settling steps; verify image orientation, state order, gripper convention and action/controller scaling against the pinned upstream recipe.
3. Integrate SafeLIBERO in its own environment, then collect controlled safe/unsafe contrasts. See [the data protocol](docs/datasets.md#data-construction-protocol).
4. Stream selected activations to disk; fit probes/directions using only the discovery split. Modules should be enumerated from the loaded checkpoint, not guessed. Run `python scripts/smoke_inference.py --model libero --list-modules`.
5. Tune interventions on selection tasks, then evaluate held-out tasks/layouts under the four conditions. Log policy proposals separately from filtered execution.
6. Integrate SO-101 calibration and recording before any physical execution. The checkpoint's native joint values are not assumed to be the units accepted by your motor controller.

Hooks run eagerly: compilation and CUDA graphs are disabled. Action-expert modules can fire multiple times during flow integration; the capture sink must retain invocation order. Capturing everything can exhaust memory. Safety-layer localization is a hypothesis, not a property assumed by this code.

## Separate simulator processes

See [docs/setup.md](docs/setup.md) for pinned upstream environments, GPU diagnostics, and the server transport. Direct inference does not require installing LeRobot or a simulator. Use the policy server to avoid forcing old simulator dependencies into the model environment.

## Validation completed on 2026-10-05

- Both full snapshots (about 43.6 GB total) assembled and SHA256-verified. Inventories and completion manifests are stored beside each snapshot in `artifacts/models/`.
- Official sample inference succeeded offline on CPU in FP32: LIBERO returned finite `[10, 7]` actions; SO-101 returned finite `[30, 6]` actions. Outputs and logs are in `artifacts/*-cpu-smoke.*`.
- All 11 unit tests passed. `pip check` found no broken requirements; source compilation passed. The test suite covers action conventions, fail-closed filters, hook cleanup and transformations, metrics/splits, wire transport, and download corruption/range handling.
- `artifacts/validation.json` summarizes results; `artifacts/diagnostics.json` records the machine check. GPU inference remains unvalidated because the NVIDIA driver is unavailable.
- The policy environment is installed. LIBERO/SafeLIBERO have separate Python foundations and pinned source checkouts; their simulator dependencies and rollout adapters are still integration work. No closed-loop safety results or physical-robot validation are claimed.
