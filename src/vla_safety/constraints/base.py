from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np
from vla_safety.types import ActionChunk


class ConstraintFailure(RuntimeError):
    """Execution must stop when a filter cannot establish a valid result."""


@dataclass(frozen=True)
class FilterResult:
    action: ActionChunk
    intervened: bool
    reason: str
    modification_l2: float


class ConstraintFilter(Protocol):
    def filter(self, proposed: ActionChunk, state: Any) -> FilterResult: ...


class IdentityFilter:
    def filter(self, proposed, state):
        return FilterResult(proposed, False, 'disabled', 0.0)


class GeometricFilter:
    """Research boundary: needs forward kinematics and calibrated controller dynamics."""
    def filter(self, proposed, state):
        raise NotImplementedError('Implement workspace, swept-path obstacle and forbidden-region constraints '
                                  'using embodiment-specific kinematics; no unverified action is returned')


def checked_filter(filter_impl, proposed, state):
    try:
        result = filter_impl.filter(proposed, state)
        proposed.require_compatible(result.action)
        delta = float(np.linalg.norm(result.action.values - proposed.values))
        if not np.isfinite(result.modification_l2) or not np.isclose(delta, result.modification_l2):
            raise ValueError('Incorrect modification magnitude')
        if delta > 0 and not result.intervened:
            raise ValueError('Changed action must be marked as intervention')
        return result
    except Exception as exc:
        raise ConstraintFailure('Filter failed; do not execute the proposed action') from exc
