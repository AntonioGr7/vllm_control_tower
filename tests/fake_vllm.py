#!/usr/bin/env python3
"""Stand-in for `vllm serve` used by tests: answers /health and /metrics on --port."""

import argparse
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("subcommand")
parser.add_argument("model")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()

START = time.time()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b""
        elif self.path == "/metrics":
            tokens = int((time.time() - START) * 50)
            body = (
                "# HELP vllm:num_requests_running running\n"
                f'vllm:num_requests_running{{model_name="{args.model}"}} 2.0\n'
                f'vllm:num_requests_waiting{{model_name="{args.model}"}} 0.0\n'
                f'vllm:kv_cache_usage_perc{{model_name="{args.model}"}} 0.25\n'
                f'vllm:generation_tokens_total{{model_name="{args.model}"}} {tokens}\n'
                f'vllm:prompt_tokens_total{{model_name="{args.model}"}} {tokens * 2}\n'
            ).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print(f"fake vllm serving {args.model} on {args.port}", flush=True)
HTTPServer((args.host, args.port), Handler).serve_forever()
