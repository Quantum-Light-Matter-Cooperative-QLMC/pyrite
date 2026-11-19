"""Identity-pin registry and one-pass re-minting (#1316).

Pinned dataset/case identities (``parameter_sha256``, ``case_content_key``,
named stems) live in ``tests/data/identity-pins.json`` or, for the
physics-baseline fixtures, in their own JSON files. Tests compare through the
``identity_pin`` fixture::

    identity_pin("profile/standard/hopg/parameter_sha256", identity["parameter_sha256"])
    identity_pin.in_file(RESOLVED_JSON, ("legacy", "case_content_key"), actual)

Check mode (default) fails on a mismatch or a missing pin. ``pytest
--update-pins`` records the actual value instead, so one run of
``pyrite-dev test-suite identity-pins --update-pins`` re-mints every pin a
``*_MODEL`` or ``tables-*-N`` bump moved; review the JSON diff and record the
generation in ``tests/data/physics-baseline/README.md``. Update mode only
writes pins; every other assertion in a pin test still runs.

Fixture users are marked ``identity_pin`` at collection, which is how the
``identity-pins`` suite selects them.
"""

import contextlib
import hashlib
import json
import tempfile
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import pytest

REGISTRY_PATH = Path(__file__).parents[1] / "data" / "identity-pins.json"
UPDATE_OPTION = "--update-pins"
MARKER = "identity_pin"


@contextlib.contextmanager
def _locked(path: Path):
    """Serialize read-modify-write across xdist workers (POSIX advisory lock)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX runs are serial
        yield
        return
    # Outside the checkout, and never unlinked: removing a held lock file lets
    # a waiter and a newcomer lock different inodes.
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    lock = Path(tempfile.gettempdir()) / f"pyrite-identity-pin-{digest}.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_json(path: Path, data: Any, *, sort_keys: bool) -> None:
    text = json.dumps(data, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


class IdentityPins:
    """Compare or (``--update-pins``) re-mint pinned identity values."""

    def __init__(self, *, update: bool, registry: Path = REGISTRY_PATH) -> None:
        self.update = update
        self.registry = registry

    def __call__(self, pin_id: str, actual: Hashable) -> None:
        """Check ``actual`` against registry entry ``pin_id``."""
        self._pin(self.registry, (pin_id,), actual, sort_keys=True)

    def in_file(self, path: Path, keys: Sequence[str], actual: Hashable) -> None:
        """Check ``actual`` against the value at ``keys`` in JSON file ``path``.

        The file keeps its key order (``indent=2``); only that value is
        rewritten under ``--update-pins``.
        """
        self._pin(Path(path), tuple(keys), actual, sort_keys=False)

    def _pin(self, path: Path, keys: tuple[str, ...], actual: Hashable, *, sort_keys: bool):
        label = f"{path.name}:{'/'.join(keys)}"
        if self.update:
            with _locked(path):
                data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                node = data
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                if node.get(keys[-1]) != actual:
                    node[keys[-1]] = actual
                    _write_json(path, data, sort_keys=sort_keys)
            return
        node = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                pytest.fail(
                    f"identity pin {label} is missing; mint it with "
                    f"`pyrite-dev test-suite identity-pins {UPDATE_OPTION}`"
                )
            node = node[key]
        assert actual == node, (
            f"identity pin {label} moved: expected {node!r}, got {actual!r}. If a "
            f"*_MODEL/tables-* bump intends this, re-mint every pin with "
            f"`pyrite-dev test-suite identity-pins {UPDATE_OPTION}`"
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        UPDATE_OPTION,
        action="store_true",
        default=False,
        help="re-mint identity pins (tests/data/identity-pins.json and "
        "physics-baseline identity fields) instead of failing on a mismatch",
    )


def mark_identity_pin_tests(items: list[pytest.Item]) -> None:
    """Mark every test that requests the ``identity_pin`` fixture."""
    for item in items:
        if MARKER in getattr(item, "fixturenames", ()):
            item.add_marker(MARKER)


@pytest.fixture
def identity_pin(request: pytest.FixtureRequest) -> IdentityPins:
    return IdentityPins(update=request.config.getoption(UPDATE_OPTION))
