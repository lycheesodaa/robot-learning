import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared_release(tmp_path):
    """Synthetic fragile-release inputs and config, prepared by the probe CLI."""
    spec = importlib.util.spec_from_file_location('probe_cli', ROOT/'scripts/probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(output=tmp_path/'inputs', max_release_height_m=.05, low_height_m=.02,
                           high_height_m=.2, synthetic=True, small=False, scene_manifest=None)
    config = json.loads(module.prepare(args).read_text())
    config['bootstrap'] = 5
    config['output_root'] = str(tmp_path / 'runs')
    return config
