"""Explicit module hooks, scoped to a context; no automatic safety-neuron claims."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HookSpec:
    module: str
    operation: str = 'capture'
    strength: float = 0.0
    neurons: tuple[int, ...] = ()
    value: object = None


@contextmanager
def intervention_hooks(model, specs: list[HookSpec], sink: Callable | None = None):
    """Sink(name, CPU tensor) is called on every invocation, including flow steps.

    Target tensor outputs or tuple outputs whose first item is a tensor. Capture
    streams to the caller; nothing accumulates implicitly on GPU or host memory.
    """
    import torch
    handles = []
    modules = dict(model.named_modules())
    try:
        for spec in specs:
            if spec.module not in modules:
                raise ValueError(f'Unknown module: {spec.module}')
            if spec.operation not in ('capture', 'ablate', 'scale', 'steer', 'patch'):
                raise ValueError(f'Unknown operation: {spec.operation}')
            if not 0 <= spec.strength <= 1 or len(set(spec.neurons)) != len(spec.neurons):
                raise ValueError('Strength must be in [0,1]; neuron indices must be unique')
            def hook(module, inputs, output, spec=spec):
                x = output[0] if isinstance(output, tuple) else output
                if not isinstance(x, torch.Tensor):
                    raise TypeError('Select a tensor-output module')
                if sink is not None:
                    sink(spec.module, x.detach().cpu().clone())
                if spec.operation == 'capture' or spec.strength == 0:
                    return output
                if any(i < 0 or i >= x.shape[-1] for i in spec.neurons):
                    raise ValueError('Neuron index out of bounds')
                selection = list(spec.neurons) if spec.neurons else slice(None)
                y = x.clone()
                selected = x[..., selection]
                if spec.operation in ('ablate', 'scale'):
                    y[..., selection] = selected * (1 - spec.strength)
                else:
                    if spec.value is None:
                        raise ValueError('Patch/steer requires a value')
                    value = torch.as_tensor(spec.value, dtype=x.dtype, device=x.device)
                    if spec.operation == 'patch' and value.shape != selected.shape:
                        raise ValueError('Patch tensor must match selected activation shape')
                    if spec.operation == 'steer' and value.shape != selected.shape[-1:]:
                        raise ValueError('Steering direction must match selected hidden dimensions')
                    y[..., selection] = (selected + spec.strength * value if spec.operation == 'steer'
                                         else (1-spec.strength)*selected + spec.strength*value)
                return (y, *output[1:]) if isinstance(output, tuple) else y
            handles.append(modules[spec.module].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
