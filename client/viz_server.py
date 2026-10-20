"""Run recorder and local web viewer for robot-client runs.

A run lives in runs/<run_id>/: events.jsonl (one JSON event per line) plus frames/*.jpg
(the camera images of every observation sent to the policy server). viz_client.py
records runs and serves this viewer live; this file also replays a finished run:

    python client/viz_server.py                 # latest run under ./runs
    python client/viz_server.py runs/20261020_163739 --port=8765
"""

import argparse
import json
import math
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

VIEWER_HTML = Path(__file__).with_name("viz.html")
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _clean(value):
    """Round floats and replace NaN/inf so the browser's JSON.parse accepts the event."""
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class RunRecorder:
    """Keeps every event in memory for the live viewer and writes events and frames to disk
    on a background thread, so the control loop never waits on file I/O or JPEG encoding."""

    live = True

    def __init__(self, run_dir: Path, meta: dict):
        self.dir = Path(run_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._file = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self._pending: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, name="viz-writer", daemon=True)
        self._writer.start()
        self.add({"type": "meta", **meta})

    def add(self, event: dict):
        event = _clean(event)
        with self._lock:
            self._events.append(event)
        self._pending.put(("event", event))

    def add_frame(self, name: str, rgb_image):
        self._pending.put(("frame", name, rgb_image))

    def events_since(self, index: int) -> tuple[int, list[dict]]:
        with self._lock:
            return len(self._events), self._events[index:]

    def close(self):
        self._pending.put(None)
        self._writer.join(timeout=10)
        self._file.close()

    def _write_loop(self):
        while (item := self._pending.get()) is not None:
            try:
                if item[0] == "event":
                    self._file.write(json.dumps(item[1]) + "\n")
                else:
                    _, name, rgb = item
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(self.dir / "frames" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if self._pending.empty():
                    self._file.flush()
            except Exception as e:  # never let a bad frame kill the writer
                print(f"[viz] failed to write {item[0]}: {e}")


class RunReplay:
    """Read-only store over a finished run directory."""

    live = False

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        with open(self.dir / "events.jsonl", encoding="utf-8") as f:
            self._events = [json.loads(line) for line in f if line.strip()]

    def events_since(self, index: int) -> tuple[int, list[dict]]:
        return len(self._events), self._events[index:]


def serve(store, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Serve the viewer for `store` (a RunRecorder or RunReplay) on a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/events":
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                    total, events = store.events_since(since)
                    body = json.dumps({"next": total, "live": store.live, "events": events})
                    self._send(body.encode(), "application/json")
                elif url.path.startswith("/frames/"):
                    frame = store.dir / "frames" / Path(url.path).name
                    if frame.is_file():
                        self._send(frame.read_bytes(), "image/jpeg")
                    else:
                        self._send(b"not found", "text/plain", 404)
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="viz-http", daemon=True).start()
    return server


def _resolve_run(path: Path) -> Path:
    if (path / "events.jsonl").is_file():
        return path
    runs = sorted(p for p in path.iterdir() if (p / "events.jsonl").is_file())
    if not runs:
        raise SystemExit(f"No runs (folders with events.jsonl) found in {path}")
    return runs[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded robot-client run in the browser.")
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUNS_DIR,
                        help="run folder, or a folder of runs (picks the latest)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve_run(args.run)
    server = serve(RunReplay(run_dir), args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Replaying {run_dir} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
