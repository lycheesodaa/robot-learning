import importlib.util
import json
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('download', Path(__file__).parents[1]/'scripts/download_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_snapshot_detects_corruption_and_missing_shard(tmp_path):
    weights = tmp_path/'weights.safetensors'
    weights.write_bytes(b'fake')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':weights.name}}))
    (tmp_path/'norm_stats.json').write_text('{"metadata_by_tag": {"libero": {}}}')
    manifest = {'norm_tag':'libero','files':[{'path':weights.name,'size':4,'sha256':module.digest(weights)}]}
    module.verify(tmp_path,manifest)
    weights.write_bytes(b'bad!')
    with pytest.raises(RuntimeError,match='SHA256'):
        module.verify(tmp_path,manifest)
    weights.unlink()
    with pytest.raises(RuntimeError,match='Missing'):
        module.verify(tmp_path,manifest)


def test_missing_normalization_tag(tmp_path):
    (tmp_path/'model.safetensors.index.json').write_text('{"weight_map":{}}')
    (tmp_path/'norm_stats.json').write_text('{"metadata_by_tag":{}}')
    with pytest.raises(RuntimeError,match='Normalization'):
        module.verify(tmp_path, {'norm_tag':'libero','files':[]})
