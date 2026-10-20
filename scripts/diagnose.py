#!/usr/bin/env python3
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

report = {'python': platform.python_version(), 'disk_free_gb': shutil.disk_usage('.').free/1e9,
          'nvidia_device_nodes': [str(x) for x in Path('/dev').glob('nvidia*')]}
for command in (['nvidia-smi'], ['lspci']):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=15)
        report[command[0]] = {'exit_code': r.returncode, 'output': r.stdout+r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        report[command[0]] = str(e)
for name in ('torch', 'transformers', 'huggingface-hub'):
    try:
        report[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report[name] = 'not installed'
try:
    import torch
    report['cuda_available'] = torch.cuda.is_available()
    report['cuda_runtime'] = torch.version.cuda
    report['gpus'] = [{'name': torch.cuda.get_device_name(i),
                       'vram_gb': torch.cuda.get_device_properties(i).total_memory/1e9}
                      for i in range(torch.cuda.device_count())]
except ImportError:
    report['cuda_available'] = False
print(json.dumps(report, indent=2))
