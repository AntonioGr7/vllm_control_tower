#!/usr/bin/env python3
"""Stand-in for `vllm serve` used by tests.

Answers /health, /metrics, /tokenize, streamed /v1/completions (a fixed 4 ms per token)
and /v1/embeddings on --port, and prints vLLM-style startup memory lines.
"""

import argparse
import json
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN_DELAY = 0.004

parser = argparse.ArgumentParser()
parser.add_argument("subcommand")
parser.add_argument("model")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()

START = time.time()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, body: bytes, status: int = 200, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(b"")
        elif self.path == "/metrics":
            tokens = int((time.time() - START) * 50)
            self._send(
                (
                    "# HELP vllm:num_requests_running running\n"
                    f'vllm:num_requests_running{{model_name="{args.model}"}} 2.0\n'
                    f'vllm:num_requests_waiting{{model_name="{args.model}"}} 0.0\n'
                    f'vllm:kv_cache_usage_perc{{model_name="{args.model}"}} 0.25\n'
                    f'vllm:generation_tokens_total{{model_name="{args.model}"}} {tokens}\n'
                    f'vllm:prompt_tokens_total{{model_name="{args.model}"}} {tokens * 2}\n'
                ).encode(),
                ctype="text/plain",
            )
        else:
            self._send(b"", 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/tokenize":
            words = body["prompt"].split()
            self._send(json.dumps({"tokens": list(range(1, len(words) + 1))}).encode())
        elif self.path == "/v1/embeddings":
            time.sleep(TOKEN_DELAY)
            self._send(json.dumps({"data": [{"embedding": [0.0] * 8}]}).encode())
        elif self.path == "/v1/completions":
            self._stream(body)
        else:
            self._send(b"", 404)

    def _stream(self, body: dict) -> None:
        n = int(body.get("max_tokens", 16))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for _ in range(n):
            time.sleep(TOKEN_DELAY)
            chunk = {"choices": [{"text": " tok", "index": 0}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        usage = {"choices": [], "usage": {"completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *a):
        pass


signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print("INFO [model_runner.py:428] Model loading took 0.10 GiB memory and 1.0 seconds", flush=True)
print("INFO [gpu_worker.py:640] Available KV cache memory: 1.50 GiB", flush=True)
print(
    "INFO [kv_cache_utils.py:2395] GPU KV cache size: 12,345 tokens, "
    "Maximum concurrency for 4,096 tokens per request: 3.01x",
    flush=True,
)
print(f"fake vllm serving {args.model} on {args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
