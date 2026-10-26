"""Bounded, eager capture at semantic module sites in the pinned model."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fnmatch


@dataclass(frozen=True)
class Component:
    name: str
    module: str
    site: str
    branch: str
    kind: str


def enumerate_components(model, patterns=('*',)):
    result = []
    blocks = {'MolmoAct2DecoderLayer': 'language', 'MolmoAct2PostNormDecoderLayer': 'language',
              'MolmoAct2VisionBlock': 'vision', 'ActionExpertBlock': 'action'}
    mlps = {'LanguageModelMLP': ('ff_out', 'language'), 'ViTMLP': ('w2', 'vision'),
            'ActionExpertMLP': ('down_proj', 'action')}
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls in blocks:
            result.append(Component(name + ':output', name, 'output', blocks[cls], 'residual_channels'))
        elif cls in mlps:
            suffix, branch = mlps[cls]
            path = name + '.' + suffix
            result.append(Component(path + ':input', path, 'input', branch, 'mlp_neurons'))
    result = [c for c in result if any(fnmatch.fnmatchcase(c.name, p) for p in patterns)]
    if not result:
        raise ValueError('No supported components matched; enumerate modules before selecting patterns')
    return result


@contextmanager
def capture_components(model, components, *, raw=False, max_bytes=256 * 1024 * 1024):
    """Batch size one: mean every emitted position; no implicit token-mask claims.

    Retain every invocation, reducing on-device before host transfer unless raw
    tensors were requested. Fail explicitly rather than silently truncate.
    """
    import torch
    captures = {c.name: [] for c in components}
    handles = []
    modules = dict(model.named_modules())
    used = 0

    def record(c, tensor):
        nonlocal used
        x = tensor[0] if isinstance(tensor, tuple) else tensor
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise TypeError(f'{c.name} did not emit a tensor')
        x = x.detach()
        size = x.shape[-1] * 4 + (x.numel() * 4 if raw else 0)
        if used + size > max_bytes:
            raise MemoryError('Activation capture limit exceeded; select fewer components or disable raw capture')
        used += size
        # Accumulate in float32; mean includes all emitted tokens/crops/action positions.
        mean = x.float().reshape(-1, x.shape[-1]).mean(dim=0).cpu().numpy()
        item = {'shape': list(x.shape), 'mean': mean}
        if raw:
            item['raw'] = x.float().cpu().numpy().copy()
        captures[c.name].append(item)

    try:
        for c in components:
            if c.site == 'input':
                def pre(module, inputs, c=c):
                    record(c, inputs[0])
                handles.append(modules[c.module].register_forward_pre_hook(pre))
            else:
                def post(module, inputs, output, c=c):
                    record(c, output)
                handles.append(modules[c.module].register_forward_hook(post))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def inventory(components):
    return [asdict(c) for c in components]
