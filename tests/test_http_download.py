import importlib.util
import io
from pathlib import Path
import sys
import pytest
scripts = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec = importlib.util.spec_from_file_location('http_download', scripts/'download_models_http.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_http_range_validation_and_resume(tmp_path,monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range':'bytes 2-5/10'}
    calls = []
    def request(req,timeout):
        calls.append(req.get_header('Range'))
        return Response(b'abcd')
    monkeypatch.setattr(module.urllib.request,'urlopen',request)
    target = tmp_path/'part'
    module.transfer_part('https://example.test/file',target,2,5,10)
    assert target.read_bytes() == b'abcd'
    module.transfer_part('https://example.test/file',target,2,5,10)
    assert calls == ['bytes=2-5']


def test_http_rejects_whole_file_response(tmp_path,monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**kw: Response(b'wrong'))
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='byte range'):
        module.transfer_part('https://example.test/file',tmp_path/'part',0,3,4)
    assert not (tmp_path/'part').exists()
