"""Optional local ACP bridge lifecycle for Marimo developer sessions."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACP_STATE_PATH = ROOT / ".cache" / "acp-servers.json"
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


def _popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _write_state(processes: dict[str, subprocess.Popen[object]]) -> None:
    ACP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: {"pid": process.pid, "port": ACP_SERVERS[name][1]}
        for name, process in processes.items()
    }
    ACP_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def terminate_process_tree(pid: int) -> None:
    """Terminate a bridge and the npx/adapter processes it spawned."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        os.killpg(pid, 15)


def start_acp_servers() -> list[subprocess.Popen[object]]:
    """Start both ACP bridges and save their process IDs for ``acp-down``."""
    if ACP_STATE_PATH.exists():
        raise RuntimeError(f"ACP state already exists at {ACP_STATE_PATH}; run acp-down first.")

    processes: dict[str, subprocess.Popen[object]] = {}
    try:
        for name, (adapter, port) in ACP_SERVERS.items():
            processes[name] = subprocess.Popen(
                acp_bridge_command(adapter, port), cwd=ROOT, **_popen_kwargs()
            )
        _write_state(processes)
    except BaseException:
        for process in processes.values():
            terminate_process_tree(process.pid)
        raise
    return list(processes.values())


def stop_acp_servers() -> None:
    """Stop bridges recorded by ``acp-up`` and clear their local state."""
    if not ACP_STATE_PATH.exists():
        print("No ACP bridge state found.")
        return
    state = json.loads(ACP_STATE_PATH.read_text(encoding="utf-8"))
    try:
        for server in state.values():
            terminate_process_tree(server["pid"])
    finally:
        ACP_STATE_PATH.unlink(missing_ok=True)


@contextmanager
def running_acp() -> Iterator[None]:
    """Keep ACP bridges alive for the duration of a Marimo launch."""
    start_acp_servers()
    try:
        yield
    finally:
        stop_acp_servers()
