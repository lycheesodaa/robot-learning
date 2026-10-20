from typing import Protocol
from vla_safety.types import Observation, ActionChunk


class Environment(Protocol):
    def reset(self, task: str, seed: int) -> Observation: ...
    def step(self, action: ActionChunk): ...
    def close(self): ...


class LiberoEnvironment:
    """Boundary for an independently installed/pinned LIBERO fork."""
    def reset(self, task, seed):
        raise NotImplementedError('Bind task BDDL and seeded initial state; preserve agent/wrist image order')

    def step(self, action):
        raise NotImplementedError('Validate controller scaling, step physics, and compute trajectory safety labels')

    def close(self):
        pass


class SO101Environment:
    def reset(self, task, seed):
        raise NotImplementedError('Supply calibrated robot state and camera configuration')

    def step(self, action):
        raise NotImplementedError('Live actuation is unavailable; use recorded-observation policy inference')

    def close(self):
        pass
