import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from vla_safety.archive import RunArchive, read_events, verify_archive
from vla_safety.capture import Component, capture_components
from vla_safety.concepts.analysis import fit_probe, metrics, probe_scores
from vla_safety.concepts.manifest import load_manifest, write_manifest
from vla_safety.concepts.replay import export_replay
from vla_safety.concepts.runner import run_experiment
from vla_safety.release import expand_release_scenes, release_bank, release_settings

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_archive_and_offline_replay(prepared_release, tmp_path):
    folder = run_experiment(prepared_release, ROOT)
    assert verify_archive(folder)['ok']
    events = read_events(folder)
    assert events[-1]['status'] == 'completed'
    result = json.loads((folder / 'analysis/fragility/results.json').read_text())
    assert result['rankings'][0]['channel'] == 0
    assert result['probes'][0]['evaluation']['auroc'] == 1.
    assert any((folder / 'source').rglob('runner.py'))
    # Remove original inputs; copied inputs must still support replay after relocation.
    for image in (tmp_path / 'inputs').glob('*.png'):
        image.unlink()
    (tmp_path / 'inputs/examples.jsonl').unlink()
    moved = tmp_path / 'moved'
    folder.rename(moved)
    assert verify_archive(moved)['ok']
    output = tmp_path / 'offline.html'
    # Import blocker establishes that replay does not even import policy/torch.
    code = '''
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guard(name, *a, **kw):
    if name.split('.')[0] in ('torch', 'transformers', 'mujoco'):
        raise RuntimeError('Model/simulator dependency imported during replay')
    return original(name, *a, **kw)
builtins.__import__ = guard
from vla_safety.concepts.replay import export_replay
export_replay(sys.argv[2], sys.argv[3])
'''
    subprocess.run([sys.executable, '-c', code, str(ROOT/'src'), str(moved), str(output)], check=True)
    html = output.read_text()
    assert 'data:image/png;base64,' in html
    assert '<script src=' not in html and 'fetch(' not in html
    (moved / 'config.json').write_text('{}')
    assert not verify_archive(moved)['ok']


def test_failed_run_still_replays(prepared_release):
    prepared_release['manifest'] += '.missing'
    with pytest.raises(FileNotFoundError):
        run_experiment(prepared_release, ROOT)
    folder = next(Path(prepared_release['output_root']).glob('*/*'))
    assert json.loads((folder / 'status.json').read_text())['status'] == 'failed'
    assert (folder / 'replay.html').exists()
    assert verify_archive(folder)['ok']


def test_interrupted_tail_and_script_injection(tmp_path):
    archive = RunArchive(tmp_path/'run', {'name': '</script><script>alert(1)</script>'})
    with (archive.folder / 'events.jsonl').open('ab') as f:
        f.write(b'{"seq":1')
    assert len(read_events(archive.folder)) == 1
    html = export_replay(archive.folder).read_text()
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script' in html
    with (archive.folder / 'events.jsonl').open('ab') as f:
        f.write(b'\n')
    with pytest.raises(ValueError, match='Corrupt'):
        read_events(archive.folder)


def test_manifest_rejects_split_leakage(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(image)
    scene = {'id': 's', 'group_id': 'g', 'images': [str(image)]*2, 'state': [0]*6,
             'instruction': 'Place the object.', 'embodiment': 'so101', 'split': 'exploratory'}
    settings = release_settings()
    rows = expand_release_scenes([scene], release_bank(settings)[:1], settings)
    path = tmp_path / 'examples.jsonl'
    write_manifest(path, rows)
    assert len(load_manifest(path)) == 16
    rows[0]['split'] = 'discovery'
    rows[1]['split'] = 'evaluation'
    write_manifest(path, rows)
    with pytest.raises(ValueError, match='leakage'):
        load_manifest(path)


def test_input_capture_limit_and_cleanup():
    import torch
    from vla_safety.hooks import HookSpec, intervention_hooks
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    component = Component('0:input', '0', 'input', 'test', 'mlp_neurons')
    x = torch.ones(1, 4, 3)
    with capture_components(model, [component]) as captures:
        model(x)
        model(x*2)
    assert len(captures['0:input']) == 2
    assert np.all(captures['0:input'][1]['mean'] == 2)
    with intervention_hooks(model, [HookSpec('0', 'ablate', 1, site='input')]):
        assert torch.count_nonzero(model(x)) == 0
    with pytest.raises(MemoryError):
        with capture_components(model, [component], raw=True, max_bytes=1):
            model(x)
    assert not model[0]._forward_pre_hooks


def test_probe_generalization_and_keyword_shortcut():
    y = np.array([0, 1]*20)
    x = np.column_stack([y*2-1, np.linspace(-1, 1, len(y))])
    probe = fit_probe(x, y)
    assert metrics(y, probe_scores(probe, x))['auroc'] == 1
    # The sole discriminating feature disappears on keyword-free wording.
    novel = x.copy()
    novel[:, 0] = 0
    assert metrics(y, probe_scores(probe, novel))['auroc'] < .6
    assert metrics(y, np.full(len(y), .5))['auroc'] == .5


def test_archive_path_traversal(tmp_path):
    archive = RunArchive(tmp_path / 'run', {})
    with pytest.raises(ValueError, match='escapes'):
        archive.json('../outside.json', {})
