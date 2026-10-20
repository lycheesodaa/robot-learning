"""Episode records separate native task success from trajectory safety."""
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import math

CONDITIONS = {'vanilla', 'internal', 'external', 'combined'}


@dataclass
class StepRecord:
    proposed: list[list[float]]
    executed: list[list[float]]
    proposed_risk: float | None
    violations: list[str]
    safety_cost: float
    intervened: bool
    modification_l2: float
    action_metadata: dict

    def __post_init__(self):
        for value in (self.safety_cost, self.modification_l2):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Costs and modification must be finite and nonnegative')
        if self.proposed_risk is not None and (not math.isfinite(self.proposed_risk) or self.proposed_risk < 0):
            raise ValueError('Invalid proposed risk')


@dataclass
class EpisodeRecord:
    task: str
    layout: str
    pair_id: str
    split: str
    seed: int
    model_revision: str
    condition: str
    success: bool
    steps: list[StepRecord] = field(default_factory=list)

    def __post_init__(self):
        if self.condition not in CONDITIONS or self.split not in ('discovery', 'selection', 'evaluation'):
            raise ValueError('Unknown experiment condition or split')


def append_episode(path: Path, episode: EpisodeRecord):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(asdict(episode), allow_nan=False) + '\n')


def summarize(episodes: list[EpisodeRecord]):
    if not episodes or any(not ep.steps for ep in episodes):
        raise ValueError('Metrics require nonempty episodes')
    steps = [step for ep in episodes for step in ep.steps]
    risks = [s.proposed_risk for s in steps if s.proposed_risk is not None]
    n = len(episodes)
    return {
        'episodes': n,
        'task_success_rate': sum(ep.success for ep in episodes)/n,
        'episode_violation_rate': sum(any(s.violations for s in ep.steps) for ep in episodes)/n,
        'mean_cumulative_safety_cost': sum(s.safety_cost for s in steps)/n,
        'intervention_frequency': sum(s.intervened for s in steps)/len(steps),
        'mean_action_modification_l2': sum(s.modification_l2 for s in steps)/len(steps),
        'mean_proposed_risk': sum(risks)/len(risks) if risks else None,
        'proposed_risk_coverage': len(risks)/len(steps),
    }


def validate_splits(episodes):
    seen = {}
    for ep in episodes:
        for key in (('pair', ep.pair_id), ('task_layout', ep.task, ep.layout)):
            if key in seen and seen[key] != ep.split:
                raise ValueError(f'Split leakage: {key}')
            seen[key] = ep.split


def summarize_by_condition(episodes):
    """Do not pool four experimental conditions into one reported result."""
    return {condition: summarize([ep for ep in episodes if ep.condition == condition])
            for condition in sorted({ep.condition for ep in episodes})}
