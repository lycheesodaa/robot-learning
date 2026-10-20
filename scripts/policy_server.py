#!/usr/bin/env python3
"""Single-request-at-a-time local policy server. POST /predict with wire v1 JSON."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
from vla_safety.policies.molmoact2 import MolmoAct2Policy
from vla_safety.policies.wire import decode_observation, encode_action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['libero', 'so101'], required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = p.parse_args()
    c = json.loads((ROOT/'configs/models.json').read_text())[args.model]
    policy = MolmoAct2Policy(c, ROOT/'artifacts/models'/args.model/c['revision'], dtype=args.dtype)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/predict':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 32*1024*1024:
                    raise ValueError('Body must be 1 byte to 32 MiB')
                obs = decode_observation(json.loads(self.rfile.read(size)))
            except Exception as exc:
                self.send_error(400, str(exc))
                return
            try:
                payload = json.dumps(encode_action(policy.predict(obs)), allow_nan=False).encode()
            except Exception as exc:
                self.send_error(500, str(exc))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
