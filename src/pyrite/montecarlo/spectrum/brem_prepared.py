"""Internal reusable band quadrature for fully covered coupled BremsLib tracks.

Same DDCS, exact piecewise segment escape and primary grouping as the scalar
scorer. Only preparation/reduction is shared. Unsupported table coverage fails
closed; callers needing the EEDL fallback retain ``mc_coupled_brem_spectrum``.
No production defaults or scientific identities change.
"""

import numpy as np

from ..._backend import BACKEND, REAL, _to_cpu, xp
from ...materials.attenuation import _finite_mu_or_raise, _mu_total_inv_ang, _normalize_composition
from .brem import _brem_incident_state
from .brem_bremslib import (
    bremslib_segment_state,
    dielectric_scoring_factor,
    evaluate_bremslib,
    stage_bremslib_table,
)
from .brem_events import _check_transport_partition, _scoring_plasma_eV
from .cxr import _observation_direction
from .segment_escape import mean_transmission, segment_escape_paths


def _band_quadrature(grids, weights):
    if not grids or len(grids) != len(weights):
        raise ValueError("band grids and weights must have equal nonzero length")
    nodes, quadrature, offsets = [], [], [0]
    for grid, q in zip(grids, weights, strict=True):
        grid, q = np.asarray(grid, dtype=float), np.asarray(q, dtype=float)
        if (
            grid.ndim != 1
            or not grid.size
            or np.any(~np.isfinite(grid))
            or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError("each band grid must be finite, positive and increasing")
        if q.shape != grid.shape or np.any(~np.isfinite(q)) or np.any(q < 0):
            raise ValueError("band weights must match nodes and be finite/nonnegative")
        nodes.append(grid)
        quadrature.append(q)
        offsets.append(offsets[-1] + grid.size)
    return np.concatenate(nodes), np.concatenate(quadrature), np.asarray(offsets, dtype=np.uint32)


class _PreparedBremsLibBands:
    """Grid/material preparation reused across independent transported blocks.

    Nodes are photon energies [eV]; weights integrate physical energy [eV].
    Each band's output is photons per original primary per sr, without an
    ensemble normalization. Angular integration supplies solid-angle weights.
    Layers and finite footprints follow the scalar production scorer.
    """

    def __init__(
        self,
        grids,
        weights,
        *,
        composition,
        bremslib_tables,
        cutoff_eV,
        layers=None,
        row_chunk=65536,
    ):
        nodes, q, offsets = _band_quadrature(grids, weights)
        if row_chunk < 1 or not np.isfinite(cutoff_eV) or cutoff_eV <= 0:
            raise ValueError("row_chunk and cutoff_eV must be positive")
        self.composition = _normalize_composition(None, None, composition)
        self.tables = bremslib_tables
        if any(el not in bremslib_tables for el, _ in self.composition):
            raise ValueError("prepared scoring requires full BremsLib table coverage")
        self.staged = {el: stage_bremslib_table(bremslib_tables[el]) for el, _ in self.composition}
        self.energy = xp.asarray(nodes, dtype=REAL)
        self.weights = xp.asarray(q, dtype=REAL)
        self.offsets_host = offsets
        self.offsets = xp.asarray(offsets)
        self.bands = len(grids)
        self.layers = layers
        self.cutoff_eV = float(cutoff_eV)
        self.row_chunk = int(row_chunk)
        absorbers = [self.composition] if layers is None else [c for _, _, c in layers]
        self.mu = xp.ascontiguousarray(
            xp.stack(
                [
                    _finite_mu_or_raise(
                        _mu_total_inv_ang(c, self.energy), self.energy, "prepared BremsLib bands"
                    )
                    for c in absorbers
                ]
            ),
            dtype=REAL,
        )

    def bind(self, segments):
        """Bind one independent block; never reuse primary state across runs."""
        return _PreparedBandTracks(self, segments)


class _PreparedBandTracks:
    """Block state shared by every angular node and every output band."""

    def __init__(self, recipe, segments):
        self.recipe, self.segments = recipe, segments
        _check_transport_partition(segments, recipe.cutoff_eV, recipe.tables)
        self.plasma_eV = _scoring_plasma_eV(segments, {"composition": recipe.composition})
        self.primaries = int(segments["Ne"])
        if self.primaries < 1:
            raise ValueError("incident primary count must be positive")
        ids = xp.asarray(segments["elec_id"])
        self.index = xp.flatnonzero((ids >= 0) & (ids < self.primaries))
        self.ids = ids[self.index]
        field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
        self.energy = xp.asarray(segments[field], dtype=REAL)[self.index]
        self.length = xp.asarray(segments["L_ang"], dtype=REAL)[self.index]
        self.velocity = xp.asarray(segments["v_hat"], dtype=xp.float64)[self.index]
        if self.energy.size:
            low, high = float(_to_cpu(self.energy.min())), float(_to_cpu(self.energy.max()))
            for staged in recipe.staged.values():
                table = staged.table
                if (
                    not np.isfinite(low + high)
                    or low < table.minimum_incident_energy_keV
                    or high > table.maximum_incident_energy_keV
                ):
                    raise ValueError("prepared scoring requires full incident-energy coverage")
        self.momenta, _ = _brem_incident_state(self.energy)

    def integrate(self, directions, weights):
        """Return FP64 primary × band scores, with one final device download."""
        directions, weights = np.asarray(directions, dtype=float), np.asarray(weights, dtype=float)
        if (
            directions.ndim != 2
            or directions.shape[1] != 3
            or weights.shape != (len(directions),)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError("directions and nonnegative angular weights must align")
        out = xp.zeros((self.primaries, self.recipe.bands), dtype=xp.float64)
        for direction, weight in zip(directions, weights, strict=True):
            if weight:
                out += weight * self.score(direction)
        return _to_cpu(out)

    def score(self, direction):
        """Device primary × band result without downloading primary scores."""
        recipe = self.recipe
        direction = _observation_direction(0.0, direction)
        out = xp.zeros((self.primaries, recipe.bands), dtype=xp.float64)
        if not self.index.size:
            return out
        owner, fraction, start, end = segment_escape_paths(
            self.segments, self.index, direction, layers=recipe.layers, xp=xp
        )
        length = self.length[owner] * fraction
        energy = self.energy[owner]
        cosine = self.velocity[owner] @ xp.asarray(direction, dtype=xp.float64)
        ids = self.ids[owner]
        cuda = BACKEND.name == "cuda" and np.dtype(REAL) == np.dtype(np.float32)
        chunk = min(recipe.row_chunk, 65536 if cuda else max(1, (1 << 21) // recipe.energy.size))
        for first in range(0, int(owner.size), chunk):
            rows = slice(first, first + chunk)
            if cuda:
                self._cuda_rows(out, rows, owner, length, energy, cosine, ids, start, end)
            else:
                self._portable_rows(out, rows, length, energy, cosine, ids, start, end)
        return out

    def _cuda_rows(self, out, rows, owner, length, energy, cosine, ids, start, end):
        from .brem_jit_kernel import run_bremslib_element_reduction

        r = self.recipe
        paths = xp.ascontiguousarray(xp.stack((start[rows], end[rows]), axis=1), dtype=REAL)
        n_layers = int(r.mu.shape[0])
        # The existing reducer carries a 4pi prefactor; add its raw result to
        # a temporary and divide once to retain DDCS per-sr normalization.
        value = xp.zeros_like(out)
        for element, density in r.composition:
            staged = r.staged[element]
            run_bremslib_element_reduction(
                energy[rows],
                length[rows],
                self.momenta[owner[rows]],
                xp.zeros_like(energy[rows]),
                paths.reshape(-1),
                r.mu.reshape(-1),
                r.energy,
                staged,
                cosine[rows],
                None,
                Z=staged.table.atomic_number,
                number_density_ang3=density,
                n_layers=n_layers,
                out=value,
                plasma_eV=self.plasma_eV,
                band_weights=(r.weights, r.offsets),
                row_electron=ids[rows],
            )
        out += value / (4 * np.pi)

    def _portable_rows(self, out, rows, length, energy, cosine, ids, start, end):
        r = self.recipe
        transmission = mean_transmission(start[rows] @ r.mu, end[rows] @ r.mu, xp=xp)
        for element, density in r.composition:
            staged = r.staged[element]
            state = bremslib_segment_state(staged, energy[rows], cosine[rows])
            cells = evaluate_bremslib(staged, state, r.energy) * transmission
            if self.plasma_eV:
                cells *= dielectric_scoring_factor(energy[rows], r.energy, self.plasma_eV)
            scale = density * 1e16 * length[rows]
            for band in range(r.bands):
                lo, hi = r.offsets_host[band : band + 2]
                values = (cells[:, lo:hi] @ r.weights[lo:hi]) * scale
                xp.add.at(out[:, band], ids[rows], values.astype(xp.float64))
