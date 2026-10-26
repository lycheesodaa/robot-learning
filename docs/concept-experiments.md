# Concept experiments and offline workflow replay

For the task “understand when an object is fragile and avoid releasing it too high,”
see [the fragile-release profile](fragile-release.md), including actual-scene inputs,
the configurable height limit and trajectory-based outcome scoring.

Every run is an independent directory under `artifacts/concepts/<concept>/<run-id>/`.
Open its **`replay.html` directly in a browser** to inspect it without Python, a web
server, internet access, GPUs, or model weights. Use the event slider, previous/next
buttons, or play/pause to follow the saved workflow. Playback can follow recorded
timing or advance one event per second.

Replay displays saved computations. It does not rerun inference or calculate new
activations. Default recording retains **all captured channel means at every module
invocation**, including each action flow step. It does not retain all token-level
activations unless `capture_raw` is enabled. The HTML embeds the 64 channels with the
largest absolute mean per component/prediction for compact inspection; complete
channel vectors are preserved in the accompanying NPZ files. Large activation values
are not automatically concept-relevant.

## Quick start

From the repository root, create a small official-sample study (eight matched
prompts, one seed). The sample's original “drop it” wording is changed to “place it”
so the task itself does not demand a drop during gentle handling.

```bash
.envs/policy/bin/python scripts/concept_experiment.py prepare \
  --output artifacts/concept_inputs/sample --small

# Run in a GPU-visible host process; no robot connection is made.
.envs/policy/bin/python scripts/concept_experiment.py run \
  --config artifacts/concept_inputs/sample/config.json
```

The command prints the run directory after completion. Its report, arrays, source
snapshot and inputs remain available after the worker exits. Failed runs also retain
their committed data and an error record. Configuration validation errors occurring
before run creation have no archive.

Omit `--small` when preparing a new directory to generate 12 families × two matched
variants × four fragility/handling conditions: 96 inputs per scene and three seeds.
Review `prompt_bank.json`, then regenerate a manifest using the `expand_scenes`
Python function if you edit the bank; editing the bank alone does not change an
already generated manifest. The actual frozen instructions are the source of truth.
The example configuration in `configs/concepts/fragility.json` uses this dataset.

For a fast software demonstration without MolmoAct2 weights:

```bash
.envs/policy/bin/python scripts/concept_experiment.py prepare \
  --output artifacts/concept_inputs/synthetic --synthetic
.envs/policy/bin/python scripts/concept_experiment.py run \
  --config artifacts/concept_inputs/synthetic/config.json
```

Synthetic runs are prominently labelled. They have known concept-responsive
channels and test capture, analysis, interventions and replay; they are not evidence
about MolmoAct2 or prompt robustness.

## What is saved

| Artifact | Contents |
| --- | --- |
| `config.json`, `provenance.json` | Resolved experiment settings, model revision/manifest, Python/package versions, Git commit and dirty state |
| `source/` | Exact local Python implementation and project manifest, including uncommitted code |
| `inputs/` | Frozen example manifest, prompt bank and deduplicated RGB PNG inputs; original file hashes retained |
| `components.json` | Actual module names, hook sites, branches and component types |
| `predictions/*.json` | Exact instruction/state, labels, seed, action values/conventions, component shapes, timing and intervention metadata |
| `predictions/*.npz` | All captured channel means per invocation; optional raw tensors; exact action arrays |
| `analysis/<concept>/` | Ranked channels, bootstrap intervals, probe trials, held-out results and selected probe coefficients without pickle |
| `interventions/` | Action changes, matched-counterpart distance changes and control results |
| `events.jsonl`, `status.json` | Ordered workflow journal and completed/failed/cancelled status |
| `replay.html`, `checksums.json` | Self-contained replay and SHA256/size inventory |

Model weights and the Python environment are not copied into each run. They are
unnecessary for replay; new model computations still require the pinned checkpoint
and a compatible environment. Image snapshots preserve the decoded RGB pixels fed
to the policy, not the original JPEG compression bytes.

## Move, verify and recover a run

```bash
python scripts/concept_experiment.py verify artifacts/concepts/fragility/RUN_ID
python scripts/concept_experiment.py bundle artifacts/concepts/fragility/RUN_ID \
  --output artifacts/fragility-run.zip
```

The ZIP includes the whole archive, not just the HTML. Unzip it on another machine
and open `replay.html`. The verification and export commands need only the base
package dependencies (NumPy and Pillow); they never import torch or load weights.

Each completed prediction is committed before its event is appended and flushed.
After a hard kill, reconstruct a report from committed events:

```bash
python scripts/concept_experiment.py replay artifacts/concepts/fragility/RUN_ID
```

This writes a separate `RUN_ID-replay-export.html` beside the run and does not change
the original checksums. An incomplete final journal line is ignored; corruption in
earlier lines is reported. A hard-killed run may still have a saved `running` status;
this is its last recorded status, not a claim that a worker is currently alive.
Unsealed/interrupted runs cannot be exported as verified ZIP bundles. The last
in-flight prediction may be absent; automatic resumption is not implemented.

Request cancellation between predictions with:

```bash
python scripts/concept_experiment.py cancel artifacts/concepts/fragility/RUN_ID
```

## Inspect every saved activation

The portable HTML intentionally summarizes channels. For full numerical inspection:

```python
import json
from pathlib import Path
import numpy as np

run = Path("artifacts/concepts/fragility/RUN_ID")
detail = json.loads(next((run / "predictions").glob("*.json")).read_text())
with np.load(run / detail["arrays"]["path"], allow_pickle=False) as data:
    for component in detail["components"]:
        print(component["component"], component["invocation"], component["shape"])
        channel_means = data[component["key"]]
        if "raw_key" in component:
            raw_tensor = data[component["raw_key"]]
```

Set `capture_raw: true` before running if you need token-level tensors. The
`max_capture_mib` limit applies per prediction, before transfer to host memory;
exceeding it fails explicitly rather than silently omitting components. Total run
size is not capped, so choose a component subset for large/raw studies. Patching
records the selected donor's raw activations even when baseline capture is pooled.

## Configuring concepts and datasets

The supplied bank crosses fragility with gentle handling. Families cover adjectives,
pressure, breakage, impact, damage, force, structure, handling, cracking, negation,
conditionals and reordered descriptions. These are related aspects, not guaranteed
semantic equivalents. Inspect family-specific results. Referent challenges require
annotated scenes and are not automatically invented for the official sample.

Use `--concept gentle_handling` to analyse that existing label instead, or pass
multiple paths to `run --config` for a sequential experiment suite. For a new concept,
provide new instructions/labels and change the config's `concept.id`; changing the
name alone does not create examples. Remove `also_analyse` entries whose labels are
absent. A JSONL row has this structure:

```json
{"id":"example-1","images":["camera0.png","camera1.png"],"state":[0,0,0,0,0,0],"embodiment":"so101","instruction":"...","labels":{"fragility":1,"gentle_handling":0},"pair_id":"quartet-1","family_id":"pressure","group_id":"session-1","split":"discovery"}
```

Images resolve relative to the manifest. Both labels of each matched contrast are
required for ranking. Keep scene/session groups and prompt families disjoint across
primary discovery, selection and evaluation splits. Off-diagonal combinations are
marked `diagnostic` and excluded from fitting/evaluation. They are saved for later
inspection. A single scene is always exploratory.

Import explicitly selected SO-101 recording sessions with:

```bash
.envs/policy/bin/python scripts/concept_experiment.py prepare \
  --output artifacts/concept_inputs/recordings \
  --recordings artifacts/inference_runs/SESSION_ID --stride 30
```

This snapshots every 30th frame pair and creates instruction contrasts. It does not
label observed physical fragility. Session-hash splits may be too small or unbalanced;
review the manifest. Held-out probes are unavailable unless all three primary splits
contain both labels. Their numerical metrics are exploratory with very few groups.

## Analysis and intervention semantics

The runner uses batch size one and averages **all emitted positions** per module
invocation, including language/state/image tokens or action positions. This coarse
pooling does not isolate semantic tokens or remove padded vision patches. Full raw
capture supports subsequent mask-aware analysis. Every action flow invocation is
saved, although the initial ranking averages flow steps and repeated seeds.

Neuron scores are mean matched activation differences divided by discovery-split
activation standard deviation. Groups are within-layer top-ranked subsets. L2
logistic probes use a NumPy implementation with training-only standardization and
feature-width normalization; convergence status and selection trials are recorded.
Only the three highest discovery-ranked components are probed. Group size and
regularization are selected using the selection split; final scores use evaluation
only. The probe null flips labels within complete matched quartets.

Bootstrap intervals resample scene/session groups and prompt families, not individual
tokens or seeds. They are omitted when either grouping has fewer than two members,
and are descriptive rather than multiplicity-corrected claims. Rank overlap and
per-family held-out performance expose wording sensitivity; no automatic “robust”
badge is assigned.

Interventions target the top-ranked MLP component (or a residual component if none
was captured). Conditions are zero strength, top neuron, top group, a same-size random
group, whole component, and a shape-compatible matched patch. A `whole_component`
intervention at an MLP input suppresses that MLP's neurons, not the entire transformer
block. Target selection uses discovery rankings; confirmatory inputs use evaluation.
Without held-out data, interventions are labelled exploratory. Action changes do not
measure real damage, task success or safe physical control.

Capture hooks precede intervention hooks, so the modified site's saved activations
are **before intervention**; downstream components capture the resulting propagation.
Input sites on `ff_out`, `down_proj` and `w2` represent post-activation MLP neurons;
block outputs are labelled residual channels. CUDA graphs remain disabled.

## Interactive launcher

The replay HTML needs no installation. For a local job launcher and embedded replay:

```bash
.envs/policy/bin/python -m pip install -e '.[concept-ui]'
.envs/policy/bin/streamlit run scripts/concept_ui.py --server.address 127.0.0.1
```

The UI edits a prepared configuration, launches a background process, displays
progress, requests cancellation, verifies archives, and opens/downloads replays.
The worker retains one model between jobs and serializes inference. Discovery is a
background study, not video-rate concept identification. Use an SSH tunnel to access
a remote host. This UI never connects to the robot controller.
