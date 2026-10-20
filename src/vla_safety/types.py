from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np


@dataclass(frozen=True)
class Observation:
    images: tuple[Any, ...]
    instruction: str
    state: np.ndarray
    embodiment: str
    timestamp: float

    def __post_init__(self):
        state = np.asarray(self.state, dtype=np.float32).copy()
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError('State must be a finite vector')
        if not self.images or not self.instruction or not np.isfinite(self.timestamp):
            raise ValueError('Images, instruction, and finite timestamp required')
        state.setflags(write=False)
        object.__setattr__(self, 'state', state)


@dataclass(frozen=True)
class ActionChunk:
    values: np.ndarray
    embodiment: str
    control_mode: str
    frame: str
    units: tuple[str, ...]
    normalization: str

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float32).copy()
        if values.ndim != 2 or not values.shape[0] or values.shape[1] != len(self.units):
            raise ValueError('Actions must be nonempty [horizon, dimensions], with units per dimension')
        if not np.isfinite(values).all():
            raise ValueError('Actions must be finite')
        if not all((self.embodiment, self.control_mode, self.frame, self.normalization)):
            raise ValueError('Action provenance is required')
        values.setflags(write=False)
        object.__setattr__(self, 'values', values)

    def require_compatible(self, other: 'ActionChunk'):
        for field in ('embodiment', 'control_mode', 'frame', 'units', 'normalization'):
            if getattr(self, field) != getattr(other, field):
                raise ValueError(f'Action convention mismatch: {field}')
        if self.values.shape != other.values.shape:
            raise ValueError('Action shapes differ')


class Policy(Protocol):
    def predict(self, observation: Observation) -> ActionChunk: ...
