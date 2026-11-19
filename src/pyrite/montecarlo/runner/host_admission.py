"""Host admission for adaptive retained rows, their join and spectrum staging."""

import numpy as np

from ..._util.env import env_value
from ..transport import host_memory

#: Bytes of kept rows one streamed adaptive block targets after a route switch.
STREAMED_BLOCK_BYTES = 2 * 1024**3


class StreamedAdaptiveRoute(Exception):
    """The retained adaptive rows would not fit; stream instead (#1280, #1274).

    Raised by a retained adaptive transport whose kept-bytes projection at
    ``max_histories`` exceeds host headroom (``resource="host"``) or, for
    resident rows, device headroom (``resource="device"``), before it is
    refused or runs out of device memory. The spectrum phase then runs the
    case through streamed adaptive scoring in blocks of ``block_electrons``.
    """

    def __init__(self, block_electrons: int, required: int, available: int, resource: str = "host"):
        super().__init__(block_electrons, required, available, resource)
        self.block_electrons = int(block_electrons)
        self.required, self.available = int(required), int(available)
        self.resource = resource

    def __str__(self) -> str:
        gib = 1024**3
        return (
            f"retained adaptive rows project to {self.required / gib:.1f} GiB, above "
            f"{self.available / gib:.1f} GiB {self.resource} headroom"
        )


def streamed_route_enabled(case) -> bool:
    """Whether a retained adaptive ``case`` may switch to streamed scoring (#1280).

    ``PYRITE_STREAMED_ADAPTIVE=off`` keeps the retained route (and its #1279
    refusal); cases :func:`.streamed_scoring.streaming_refusal` refuses never switch.
    """
    if env_value("PYRITE_STREAMED_ADAPTIVE", "auto").lower() == "off":
        return False
    from .streamed_scoring import streaming_refusal

    return streaming_refusal(case, live=True) is None


def block_bytes(blocks, *, host_only=False):
    """Count array storage once per alias, including nested transport metadata."""
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(item) for item in value)
        if hasattr(value, "nbytes") and (not host_only or isinstance(value, np.ndarray)):
            return int(value.nbytes)
        return 0

    return size(blocks)


class AdaptiveHostAdmission:
    """Project observed bytes/history without changing histories or stopping.

    Three retained populations cover the blocks, concatenated result and a
    spectrum staging/finalization copy. Headroom excludes live host arrays, so
    charge only additional bytes. Recheck every growth and join against current
    system/cgroup headroom; a sampled projection is not a reservation.
    """

    def __init__(self, limit, *, stream=False):
        self.limit = int(limit)
        #: Raise :class:`StreamedAdaptiveRoute` instead of refusing (#1280).
        self.stream = bool(stream)

    def check(self, blocks, histories, *, force=False, phase="growth"):
        if not blocks or not histories:
            return
        total = block_bytes(blocks)
        host = block_bytes(blocks, host_only=True)
        host_rows = any(isinstance(block.get("L_ang"), np.ndarray) for block in blocks)
        projected = (total * self.limit + histories - 1) // histories
        required = max(0, 3 * max(total, projected) - host)
        if self.stream:
            # Device rows count too: at the cap they download for the join.
            block = max(1, STREAMED_BLOCK_BYTES // max(1, total // histories))
            available = host_memory.host_headroom_bytes()
            if required > available:
                raise StreamedAdaptiveRoute(block, required, available)
            device = self._device_shortfall(total - host, histories)
            if device is not None:
                raise StreamedAdaptiveRoute(block, *device, resource="device")
        if not force and not host_rows:
            return
        host_memory.admit_host_allocation(
            required,
            f"adaptive kept blocks {phase}; max_histories={self.limit}; "
            "lower max_histories or use streamed adaptive scoring (#1280)",
        )

    def _device_shortfall(self, device, histories):
        """``(required, available)`` device bytes when resident rows would not fit (#1274).

        The resident join holds the kept chunks and their concatenation, then
        the spectrum phase stages its chunks (``spec_budget_mb``) beside the
        joined rows. Both peaks are projected at ``max_histories``; kept
        device bytes already count as used pool memory, so only the growth is
        charged against :func:`.electron_blocks.device_headroom_bytes`.
        Returns ``None`` when the rows are on the host, off a CUDA device, or
        the projection fits.
        """
        if device <= 0:
            return None
        from .chunking import _RESOURCE_POLICY
        from .electron_blocks import device_headroom_bytes

        headroom = device_headroom_bytes()
        if headroom is None:
            return None
        projected = (device * self.limit + histories - 1) // histories
        reserve = int(_RESOURCE_POLICY.spec_budget_mb) << 20
        required = max(2 * projected, projected + reserve) - device
        return (required, headroom) if required > headroom else None
