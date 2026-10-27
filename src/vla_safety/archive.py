"""Append-only workflow events with portable inputs and non-pickle arrays.

Replay means inspecting saved computations, not executing the model again.
The archive contains exactly the activations requested by the capture config.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def archive_path(folder, relative):
    """Only resolve paths inside this archive, including through symlinks."""
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if path == folder or not path.is_relative_to(folder):
        raise ValueError(f'Path escapes archive: {relative}')
    return path


def read_events(folder):
    """A killed writer may leave one unfinished final line; never hide interior corruption."""
    path = Path(folder) / 'events.jsonl'
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise ValueError(f'Corrupt event line {i + 1}') from None
        if event.get('seq') != len(result):
            raise ValueError(f'Non-sequential event line {i + 1}')
        result.append(event)
    return result


def provenance(root):
    packages = {}
    for name in ('numpy', 'Pillow', 'torch', 'transformers', 'scikit-learn', 'mujoco'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
    try:
        result['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


class RunArchive:
    def __init__(self, folder, config, metadata=None):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.seq = 0
        self.status = 'running'
        self.metadata = metadata or {}
        atomic_json(self.folder / 'config.json', config)
        atomic_json(self.folder / 'provenance.json', self.metadata)
        self._status()
        self.event('run_started', config='config.json', provenance='provenance.json')

    @staticmethod
    def new_path(output_root, concept):
        if not concept or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in concept):
            raise ValueError('Concept id must contain only letters, digits, _ or -')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        return Path(output_root) / concept / run_id

    def _status(self, **extra):
        atomic_json(self.folder / 'status.json', {
            'schema_version': 1, 'status': self.status,
            'updated_at': datetime.now(timezone.utc).isoformat(), **extra})

    def json(self, relative, data):
        path = archive_path(self.folder, relative)
        atomic_json(path, data)
        return {'path': relative, 'sha256': digest(path)}

    def arrays(self, relative, arrays):
        path = archive_path(self.folder, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        return {'path': relative, 'sha256': digest(path)}

    def image(self, image):
        from io import BytesIO
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buf = BytesIO()
        image.convert('RGB').save(buf, format='PNG')
        data = buf.getvalue()
        relative = 'inputs/images/' + hashlib.sha256(data).hexdigest() + '.png'
        path = archive_path(self.folder, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
        return relative

    def event(self, kind, **payload):
        event = {'seq': self.seq, 'kind': kind, 'elapsed_seconds': time.monotonic() - self.started,
                 'time': datetime.now(timezone.utc).isoformat(), **payload}
        with (self.folder / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.seq += 1
        return event

    def prediction(self, example, seed, action, captures, *, elapsed, intervention=None):
        """All channel means per invocation are saved; raw tensors are opt-in."""
        index = self.seq
        stem = f'predictions/{index:06d}'
        arrays = {'actions': np.asarray(action['values'], dtype=np.float32)}
        components = []
        for i, (name, records) in enumerate(captures.items()):
            for j, record in enumerate(records):
                key = f'c{i}_call{j}'
                arrays[key] = np.asarray(record['mean'], dtype=np.float32)
                entry = {'component': name, 'invocation': j, 'key': key, 'shape': record['shape']}
                if 'raw' in record:
                    arrays[key + '_raw'] = record['raw']
                    entry['raw_key'] = key + '_raw'
                components.append(entry)
        array_ref = self.arrays(stem + '.npz', arrays)
        detail = {'example_id': example['id'], 'seed': seed, 'input': example,
                  'action': action, 'components': components, 'arrays': array_ref,
                  'inference_seconds': elapsed, 'intervention': intervention}
        detail_ref = self.json(stem + '.json', detail)
        self.event('prediction', example_id=example['id'], seed=seed, detail=detail_ref)
        return detail

    def finish(self, status='completed', error=None):
        if status not in ('completed', 'failed', 'cancelled'):
            raise ValueError('Invalid terminal status')
        self.status = status
        self.event('run_finished', status=status, error=error)
        self._status(error=error)
        # A rendering failure must not destroy already committed experiment data.
        from vla_safety.concepts.replay import export_replay
        export_replay(self.folder)
        seal_archive(self.folder)


def seal_archive(folder):
    folder = Path(folder)
    entries = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'checksums.json' and not path.name.endswith('.tmp'):
            entries[path.relative_to(folder).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(folder / 'checksums.json', {'schema_version': 1, 'files': entries})


def verify_archive(folder):
    folder = Path(folder)
    manifest = folder / 'checksums.json'
    if not manifest.exists():
        return {'ok': False, 'errors': ['Archive is not sealed; it may be running or interrupted.']}
    errors = []
    for relative, expected in json.loads(manifest.read_text())['files'].items():
        path = archive_path(folder, relative)
        if not path.is_file():
            errors.append(f'Missing: {relative}')
        elif path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            errors.append(f'Changed: {relative}')
    return {'ok': not errors, 'errors': errors}
