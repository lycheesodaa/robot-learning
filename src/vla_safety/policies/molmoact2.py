"""Local-only checkpoint loader using the pinned model's custom inference code."""

from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
from vla_safety.types import ActionChunk, Observation


class MolmoAct2Policy:
    def __init__(self, config: dict, snapshot: Path, device="cuda", dtype="float32"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = config
        self.device = device
        self.dtype = getattr(torch, dtype)
        manifest = json.loads((snapshot / "snapshot-manifest.json").read_text())
        if (manifest["revision"], manifest["repo_id"]) != (
            config["revision"],
            config["repo_id"],
        ):
            raise ValueError(
                "Checkpoint provenance mismatch; run download verification"
            )
        stats = json.loads((snapshot / "norm_stats.json").read_text())
        if config["norm_tag"] not in stats.get("metadata_by_tag", {}):
            raise ValueError("Missing embodiment normalization tag")
        self.processor = AutoProcessor.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                snapshot,
                trust_remote_code=True,
                local_files_only=True,
                dtype=self.dtype,
            )
            .to(device)
            .eval()
        )
        # Always eager: hook experiments must run on every forward call.

    def predict(
        self, observation: Observation, *, seed: int | None = None, num_steps: int = 10
    ) -> ActionChunk:
        import torch

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        c = self.config
        if (
            observation.embodiment != c["embodiment"]
            or len(observation.state) != c["state_dim"]
        ):
            raise ValueError(
                "Observation does not match checkpoint embodiment/state dimension"
            )
        if len(observation.images) != 2:
            raise ValueError("This adapter expects the two checkpoint cameras")
        autocast = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.startswith("cuda") and self.dtype == torch.bfloat16
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.model.predict_action(
                processor=self.processor,
                images=list(observation.images),
                task=observation.instruction,
                state=observation.state.copy(),
                norm_tag=c["norm_tag"],
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                generator=generator,
                normalize_language=True,
                enable_cuda_graph=False,
            )
        values = output.actions
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu().numpy()
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != c["action_dim"]:
            raise ValueError(f"Unexpected model action shape: {values.shape}")
        return ActionChunk(
            values,
            c["embodiment"],
            c["control_mode"],
            c["frame"],
            tuple(c["units"]),
            f"{c['repo_id']}@{c['revision']}:{c['norm_tag']}",
        )
