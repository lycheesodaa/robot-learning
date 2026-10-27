"""Known toy release mechanism for software tests, never MolmoAct2 evidence."""
import re

import numpy as np

from vla_safety.capture import Component


def synthetic_release_policy(settings):
    import torch
    from vla_safety.types import ActionChunk

    class Fixture:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.hidden = torch.nn.Identity()
            self.model.head = torch.nn.Linear(8, 2, bias=False)
            with torch.no_grad():
                self.model.head.weight.zero_()
                self.model.head.weight[0, 3] = 1.
                self.model.head.weight[0, 4] = .1
            self.components = [Component('hidden:output', 'hidden', 'output', 'synthetic', 'residual_channels'),
                               Component('head:input', 'head', 'input', 'synthetic', 'mlp_neurons')]

        def predict(self, observation, *, seed=None, num_steps=10):
            property_match = re.search(r'The target object is (fragile|durable)\.', observation.instruction)
            height_match = re.search(r'(\d+(?:\.\d+)?) cm', observation.instruction)
            if not property_match or not height_match:
                raise ValueError('Synthetic release fixture requires its explicit toy prompt bank')
            fragile = property_match.group(1) == 'fragile'
            height = float(height_match.group(1)) / 100
            high = height > settings['max_release_height_m']
            rule = 'before opening the gripper' in observation.instruction or 'keep the gripper closed' in observation.instruction
            x = np.zeros((1, 1, 8), dtype=np.float32)
            x[0, 0, :5] = [fragile, high, rule, fragile and high, fragile and high and rule]
            x[..., 5:] = np.random.default_rng(seed).normal(0, .01, (1, 1, 3))
            with torch.inference_mode():
                for _ in range(num_steps):
                    decision = self.model.head(self.model.hidden(torch.from_numpy(x)))
            release_height = settings['low_height_m'] if decision[0, 0, 0].item() > .5 else height
            values = [[height, 0.], [release_height, 1.], [release_height, 1.]]
            return ActionChunk(values, observation.embodiment, 'synthetic_release', 'object_bottom_above_receiving_surface',
                               ('metres', 'synthetic_open_flag'), 'synthetic-release-v1')
    return Fixture()
