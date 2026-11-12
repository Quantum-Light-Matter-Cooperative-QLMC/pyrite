"""Population-resolved line grids for live streamed scoring (#1001).

A ``line_grid_policy`` grid is resolved from the whole line population, but
live streaming scores each block as it finishes. :func:`resolve_streamed_line_grid`
therefore runs two transport passes over the same electron blocks. Primary
streams are addressed by ``(seed, electron, draw)``, so both passes see the
same histories. The first pass keeps only what policy resolution reads, as
compact per-row arrays (:class:`BlockGridEvidence`), and resolves the grid
with the joined path's own code; the second pass scores against it.

A speed-cache hit needs no trajectories, so it skips the first pass.

Equality with the joined grid: the sinc samples of consecutive whole-electron
blocks concatenate to the joined samples, so the sinc spacing is exact, and
``resonance-local`` seeds are integer coverage counts. The measured
``resonance-population`` stop sums line masses chunk by chunk, so its float
sums may differ in the last ulp from the joined path's; the stop is rounded up
to a whole 100 eV step, so the grid differs only if a tail ratio lands within
rounding of the truncation limit.

Windowed policies read whole-population seed providers and are refused.
"""

from typing import Any

import numpy as np

from ..._cxr_grid_policy import RESONANCE_BANDWIDTH_POLICY

#: Line-grid policy features live streaming cannot resolve from block evidence.
WINDOWED_POLICY_REFUSAL = "windowed line_grid_policy is not supported by live streamed scoring"


def policy_streaming_refusal(case) -> str | None:
    """Why the case's line-grid policy cannot be streamed, or ``None``."""
    payload = case.get("line_grid_policy")
    if payload is not None and payload.get("windows") is not None:
        return WINDOWED_POLICY_REFUSAL
    return None


class BlockGridEvidence:
    """Grid-resolution evidence accumulated over whole-electron blocks.

    Serves the two quantities :class:`.cxr_grid.SegmentGridEvidence` reads from
    a joined population, for the case's single direction ``n_hat``: the sinc
    ``(width, weight)`` samples of every valid line row, and -- for the
    measured ``resonance-population`` bandwidth -- every production line's
    ``(energy, width, weight)``. Memory is 16 bytes per valid line row plus,
    when measured, three backend reals per production line.
    """

    def __init__(self, case, payload, n_hat, Ne, abs_layers=None, groove=None):
        self.case, self.n_hat, self.Ne = case, np.asarray(n_hat, dtype=float), int(Ne)
        self.abs_layers, self.groove = abs_layers, groove
        bandwidth = payload["bandwidth"]
        self.band = (float(bandwidth["start_eV"]), float(bandwidth["stop_eV"]))
        self.measured = bandwidth["policy"] == RESONANCE_BANDWIDTH_POLICY
        self._width: list[np.ndarray] | None = []
        self._weight: list[np.ndarray] | None = []
        self._populations: list = []

    def add(self, segs) -> None:
        """Record one block; commits only after every block quantity succeeded."""
        from ..spectrum.diagnostics import sinc_feature_samples
        from .cxr_grid import _direction_populations

        width, weight = sinc_feature_samples(segs, self.n_hat, electron_limit=self.Ne)
        populations = []
        if self.measured and int(segs["L_ang"].shape[0]):
            populations = _direction_populations(
                self.case,
                segs,
                self.n_hat,
                self.Ne,
                self.abs_layers,
                self.groove,
                *self.band,
                None,
            )
        if self._width is None or self._weight is None:
            raise RuntimeError("streamed sinc evidence was already reduced")
        self._width.append(width)
        self._weight.append(weight)
        self._populations.extend(populations)

    @property
    def nbytes(self) -> int:
        """Host bytes the evidence holds."""
        arrays = [*(self._width or ()), *(self._weight or ())]
        arrays += [
            array
            for population in self._populations
            for array in (population.energy_eV, population.weight, population.width_eV)
        ]
        return sum(array.nbytes for array in arrays)

    def _check_direction(self, n_hat) -> None:
        if not np.array_equal(np.asarray(n_hat, dtype=float), self.n_hat):
            raise ValueError("streamed line-grid evidence was collected along another direction")

    def sinc_spacing(self, n_hat, aliased_weight_limit):
        """:func:`.diagnostics.sinc_feature_spacing` of the accumulated rows."""
        from ..spectrum.diagnostics import sinc_spacing_from_samples

        self._check_direction(n_hat)
        width, weight = self._width, self._weight
        if width is None or weight is None:
            raise RuntimeError("streamed sinc evidence was already reduced")
        samples = [_drain(width), _drain(weight)]
        del width, weight
        self._width = self._weight = None
        return sinc_spacing_from_samples(samples, aliased_weight_limit=aliased_weight_limit)

    def populations(self, n_hat, start, ceiling, profile):
        """Every accumulated production line over the policy band."""
        del profile  # stage profiling covers the joined resolver only
        self._check_direction(n_hat)
        if not self.measured or (float(start), float(ceiling)) != self.band:
            raise ValueError("streamed line-grid evidence holds no lines for this band")
        return list(self._populations)


def _drain(blocks: list[np.ndarray]) -> np.ndarray:
    """Join ``blocks`` in order into one array, releasing each as it is copied.

    Equal to ``np.concatenate(blocks)``; the peak is the result plus one block
    rather than twice the total (#1001).
    """
    size = sum(block.size for block in blocks)
    joined = np.empty(size, dtype=blocks[0].dtype if blocks else float)
    offset = 0
    blocks.reverse()
    while blocks:
        block = blocks.pop()
        joined[offset : offset + block.size] = block
        offset += block.size
    return joined


def resolve_streamed_line_grid(
    case, block_electrons, *, transport_core, n_hat, groove
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """The case's policy line grid, from the speed cache or a transport pass.

    Returns ``(E_grid, record, grid_pass)``; ``record`` equals the joined
    :func:`.cxr_grid.resolve_cxr_grid` record and ``grid_pass`` describes the
    first pass (``transported``, ``evidence_bytes``).
    """
    from . import _transport_case
    from .cxr_grid import _resolve_policy_grid, cached_policy_cxr_grid

    hit = cached_policy_cxr_grid(case)
    if hit is not None:
        return *hit, {"transported": False, "evidence_bytes": 0}
    payload = case["line_grid_policy"]
    abs_layers = case.get("abs_layers")
    evidence = BlockGridEvidence(case, payload, n_hat, case["Ne"], abs_layers, groove)
    _transport_case(
        case,
        transport_core=transport_core,
        block_electrons=block_electrons,
        block_consumer=evidence.add,
        keep_segments_on_device=True,
    )
    evidence_bytes = evidence.nbytes
    grid, record = _resolve_policy_grid(
        payload,
        case,
        None,
        (n_hat,),
        case["Ne"],
        abs_layers,
        groove,
        observation=False,
        evidence=evidence,
    )
    return grid, record, {"transported": True, "evidence_bytes": evidence_bytes}
