#!/usr/bin/env python3
"""Stand-in for the docker CLI used by tests.

`run` execs fake_vllm.py with the arguments after the image, like a container would,
and records its PID under $FAKE_DOCKER_STATE/<name>.pid. `stop` and `rm -f` SIGTERM /
SIGKILL that PID. Every call is appended to $FAKE_DOCKER_STATE/calls.jsonl.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

FAKE_VLLM = Path(__file__).parent / "fake_vllm.py"
state = Path(os.environ["FAKE_DOCKER_STATE"])
state.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\n")


def kill(name: str, sig: signal.Signals) -> bool:
    pid_file = state / f"{name}.pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    pid_file.unlink()
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    # Like docker, return once the container is gone (a zombie holds no port any more).
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().split(")")[-1].split()[0] == "Z":
                    break
        except OSError:
            break
        time.sleep(0.05)
    return True


if args[0] == "run":
    name = args[args.index("--name") + 1]
    image_at = args.index("--entrypoint") + 2
    (state / f"{name}.pid").write_text(str(os.getpid()))
    os.execv(sys.executable, [sys.executable, str(FAKE_VLLM), *args[image_at + 1 :]])
elif args[0] in ("stop", "rm"):
    sig = signal.SIGTERM if args[0] == "stop" else signal.SIGKILL
    sys.exit(0 if kill(args[-1], sig) else 1)
sys.exit(f"fake docker: unsupported {args[0]!r}")
