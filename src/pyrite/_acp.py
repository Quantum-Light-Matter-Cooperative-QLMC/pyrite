"""Optional local ACP bridge lifecycle for Marimo developer sessions.

Lives at the package root rather than under ``apps/`` because nothing in
``apps/`` uses it: its callers are the app launchers
(:mod:`pyrite.cli.commands.app_analysis` and friends) and ``pyrite-dev
acp-up``/``acp-down``. Keeping it below both leaves ``cli`` free of ``apps``
imports, so ``apps`` stays outside the driver import cycle.
"""

import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

from .console.config import workspace_root
from .paths import atomic_write_text, state_dir

ROOT = workspace_root()
ACP_STATE_PATH = state_dir() / "acp-servers.json"
ACP_SERVERS = {
    "claude": ("@agentclientprotocol/claude-agent-acp", 3017),
    "codex": ("@agentclientprotocol/codex-acp", 3021),
}


def acp_bridge_command(adapter: str, port: int) -> list[str]:
    """Build the stdio-to-WebSocket bridge command for one ACP adapter."""
    return [
        "npx",
        "-y",
        "@rebornix/stdio-to-ws",
        f"npx -y {adapter}",
        "--port",
        str(port),
    ]


def _start_bridge(adapter: str, port: int) -> subprocess.Popen[bytes]:
    """Start one ACP bridge with platform-specific process-group handling."""
    command = acp_bridge_command(adapter, port)
    if os.name == "nt":
        return subprocess.Popen(
            command, cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    return subprocess.Popen(command, cwd=ROOT, start_new_session=True)


def _write_state(processes: dict[str, subprocess.Popen[bytes]]) -> None:
    state = {
        name: {"pid": process.pid, "port": ACP_SERVERS[name][1]}
        for name, process in processes.items()
    }
    atomic_write_text(ACP_STATE_PATH, json.dumps(state, indent=2) + "\n")


def terminate_process_tree(pid: int) -> None:
    """Terminate a bridge and the npx/adapter processes it spawned."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        os.killpg(pid, 15)


def start_acp_servers() -> list[subprocess.Popen[bytes]]:
    """Start both ACP bridges and save their process IDs for ``acp-down``."""
    read_path = ACP_STATE_PATH
    if read_path.exists():
        raise RuntimeError(f"ACP state already exists at {read_path}; run acp-down first.")

    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for name, (adapter, port) in ACP_SERVERS.items():
            processes[name] = _start_bridge(adapter, port)
        _write_state(processes)
    except BaseException:
        for process in processes.values():
            terminate_process_tree(process.pid)
        raise
    return list(processes.values())


def stop_acp_servers() -> None:
    """Stop bridges recorded by ``acp-up`` and clear their local state."""
    read_path = ACP_STATE_PATH
    if not read_path.exists():
        print("No ACP bridge state found.")
        return
    state = json.loads(read_path.read_text(encoding="utf-8"))
    try:
        for server in state.values():
            terminate_process_tree(server["pid"])
    finally:
        read_path.unlink(missing_ok=True)


@contextmanager
def running_acp() -> Iterator[None]:
    """Keep ACP bridges alive for the duration of a Marimo launch."""
    start_acp_servers()
    try:
        yield
    finally:
        stop_acp_servers()
