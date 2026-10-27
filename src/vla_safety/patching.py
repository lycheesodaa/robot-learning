"""Online activation patching from a donor instruction on the recipient's own observation.

At each prediction the donor instruction is run on the recipient's current
observation with the same inference seed. Its outputs at the selected components
are kept on the device, per invocation (the language model runs once per
prediction; the action expert runs once per flow step). The recipient run then
replaces its own outputs with the donor's, invocation by invocation, through
`HookSpec('patch')`. Mismatched shapes (for example prompts of different token
length) or invocation counts stop the episode rather than patching misaligned
positions.
"""
from contextlib import contextmanager
from dataclasses import replace

from vla_safety.capture import enumerate_components
from vla_safety.hooks import HookSpec


def select_components(model, group):
    """Residual-stream blocks of one branch, ordinals [start, stop) in model order."""
    blocks = [c for c in enumerate_components(model) if c.branch == group['branch'] and c.kind == 'residual_channels']
    start, stop = group['ordinals']
    if not 0 <= start < stop <= len(blocks):
        raise ValueError(f"Ordinals {group['ordinals']} outside the {len(blocks)} {group['branch']} blocks")
    return blocks[start:stop]


@contextmanager
def donor_capture(model, components):
    """Keep every invocation's full output (or input) tensor on its device."""
    store = {c.module: [] for c in components}
    modules = dict(model.named_modules())
    handles = []
    def keep(module_name, x):
        x = x[0] if isinstance(x, tuple) else x
        store[module_name].append(x.detach().clone())
    try:
        for c in components:
            if c.site == 'input':
                handles.append(modules[c.module].register_forward_pre_hook(lambda m, inputs, name=c.module: keep(name, inputs[0])))
            else:
                handles.append(modules[c.module].register_forward_hook(lambda m, inputs, output, name=c.module: keep(name, output)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


class Replay:
    """Patch value that returns the donor's tensor for the recipient's matching invocation."""
    def __init__(self, module, tensors):
        self.module, self.tensors, self.calls = module, tensors, 0

    def __call__(self, selected):
        if self.calls >= len(self.tensors):
            raise RuntimeError(f'{self.module}: recipient invoked more often than the donor ({len(self.tensors)})')
        value = self.tensors[self.calls]; self.calls += 1
        return value


def donor_pass(policy, observation, donor_instruction, components, *, seed, num_steps):
    donor = replace(observation, instruction=donor_instruction)
    with donor_capture(policy.model, components) as store:
        chunk = policy.predict(donor, seed=seed, num_steps=num_steps)
    if any(not tensors for tensors in store.values()):
        raise RuntimeError('A patched component did not run in the donor pass')
    return {name: Replay(name, tensors) for name, tensors in store.items()}, chunk


def replay_specs(components, replays, strength):
    return [HookSpec(c.module, 'patch', strength, (), value=replays[c.module], site=c.site) for c in components]


def check_replayed(replays, strength):
    """With strength > 0 every donor invocation must have been used exactly once."""
    if strength == 0:
        return
    for replay in replays.values():
        if replay.calls != len(replay.tensors):
            raise RuntimeError(f'{replay.module}: recipient used {replay.calls} of {len(replay.tensors)} donor invocations')
