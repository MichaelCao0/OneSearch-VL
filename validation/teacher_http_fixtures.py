"""Local test server and explicit fixture replies; never production teacher data."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


def reply(text, *, finish="STOP", model="test-only-gemini"):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                              "totalTokenCount": 120}, "modelVersion": model}


@contextmanager
def fixture_server(replies):
    received = []
    pending = list(replies)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "payload": value})
            status, body = pending.pop(0) if pending else (500, {"error": "fixture exhausted"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1beta", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
