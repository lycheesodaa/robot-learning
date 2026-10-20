import pytest
import torch
from vla_safety.interventions.hooks import HookSpec, intervention_hooks


def test_zero_strength_capture_and_cleanup():
    model = torch.nn.Sequential(torch.nn.Linear(3,3,bias=False))
    x = torch.ones(1,3)
    baseline = model(x).detach().clone()
    captures = []
    with intervention_hooks(model, [HookSpec('0','ablate',0)], lambda n,t: captures.append(t)):
        assert torch.equal(model(x),baseline)
    assert len(captures) == 1
    assert not model[0]._forward_hooks
    with pytest.raises(RuntimeError):
        with intervention_hooks(model,[HookSpec('0')]):
            raise RuntimeError('abort')
    assert not model[0]._forward_hooks


def test_ablate_patch_and_steer():
    model = torch.nn.Sequential(torch.nn.Identity())
    x = torch.ones(2,3)
    with intervention_hooks(model,[HookSpec('0','ablate',1,(1,))]):
        result = model(x)
        assert torch.equal(result[:,1],torch.zeros(2))
        assert torch.equal(result[:,0],x[:,0])
    with intervention_hooks(model,[HookSpec('0','patch',1,value=torch.zeros(2,3))]):
        assert torch.equal(model(x),torch.zeros_like(x))
    with intervention_hooks(model,[HookSpec('0','steer',.5,value=torch.ones(3))]):
        assert torch.equal(model(x),torch.full_like(x,1.5))
    with pytest.raises(ValueError):
        with intervention_hooks(model,[HookSpec('0'),HookSpec('missing')]):
            pass
    assert not model[0]._forward_hooks


def test_tuple_preserves_other_outputs():
    class TupleLayer(torch.nn.Module):
        def forward(self,x):
            return x, 'cache'
    model = torch.nn.Sequential(TupleLayer())
    with intervention_hooks(model,[HookSpec('0','ablate',1)]):
        output, cache = model(torch.ones(2,3))
    assert cache == 'cache'
    assert torch.count_nonzero(output) == 0
