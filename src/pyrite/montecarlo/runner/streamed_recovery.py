"""Bound scratch for the row-additive streamed characteristic/continuum slots."""

from dataclasses import replace

from ..spectrum.cxr._kernels import _SEG_ARRAYS


class RowScoringRecovery:
    """Learn a row ceiling per slot after allocator OOM, preserving normalization.

    Characteristic scoring adds independent segment yields. Continuum uses
    geometry-only recovery below to preserve its fused reduction order. CXR
    and TR require complete histories and must never use this row splitter.
    Slices retain global electron IDs, fixed electron limits, geometry and
    scalar metadata; no transport or already successful slice is replayed.
    """

    def __init__(self):
        self.row_limits: dict[str, int] = {}

    def evaluate(self, adapter, batch, slot):
        from .. import runner
        from .oom import _drop_traceback_locals

        rows = int(batch.segments["L_ang"].shape[0])
        limit = min(rows, self.row_limits.get(slot, rows))
        start, retries, total = 0, 0, None
        while start < rows:
            stop = min(start + limit, rows)
            segments = dict(batch.segments)
            for key in _SEG_ARRAYS:
                if segments.get(key) is not None:
                    segments[key] = segments[key][start:stop]
            try:
                values = adapter.evaluate(replace(batch, segments=segments), slot).values
            except Exception as error:
                if not runner._is_gpu_oom(error):
                    raise
                _drop_traceback_locals(error)
                if limit == 1 or retries >= runner._RESOURCE_POLICY.gpu_oom_retries:
                    error.add_note(f"Streamed {slot} allocation recovery exhausted at {limit} rows")
                    runner.BACKEND.release_memory()
                    raise
                # Finished scoring frames pin device scratch until cleared.
                error.__traceback__ = None
                limit = max(1, (stop - start) // 2)
                self.row_limits[slot] = limit
                retries += 1
            else:
                total = values if total is None else total + values
                start = stop
                continue
            # Outside the handler: the failed exception no longer owns scratch.
            runner.BACKEND.release_memory()
        if total is None:
            return adapter.evaluate(batch, slot).values
        return total


class EscapeScoringRecovery:
    """Bound geometry scratch while preserving the continuum reduction order."""

    def __init__(self):
        self.row_limit: int | None = None

    def __call__(self, segments, index, n_hat, *, layers=None, groove=None, xp):
        from .. import runner
        from ..spectrum.segment_escape import _segment_escape_paths
        from .oom import _drop_traceback_locals

        rows = int(index.size)
        limit = min(rows, self.row_limit or rows)
        start, retries, parts = 0, 0, []
        while start < rows:
            stop = min(start + limit, rows)
            view = None
            try:
                if start == 0 and stop == rows:
                    result = _segment_escape_paths(
                        segments, index, n_hat, layers=layers, groove=groove, xp=xp
                    )
                else:
                    # Gather before the float64 geometry cast, so even source
                    # arrays larger than this selection cannot exhaust scratch.
                    view = dict(segments)
                    for key in ("r_mid", "v_hat", "L_ang"):
                        view[key] = xp.asarray(segments[key])[index[start:stop]]
                    result = _segment_escape_paths(
                        view, xp.arange(stop - start), n_hat, layers=layers, groove=groove, xp=xp
                    )
            except Exception as error:
                if not runner._is_gpu_oom(error):
                    raise
                _drop_traceback_locals(error)
                view = None
                if stop - start == 1 or retries >= runner._RESOURCE_POLICY.gpu_oom_retries:
                    error.add_note("Streamed escape-geometry allocation recovery exhausted")
                    runner.BACKEND.release_memory()
                    raise
                error.__traceback__ = None
                limit = max(1, (stop - start) // 2)
                self.row_limit = limit
                retries += 1
            else:
                if start == 0 and stop == rows:
                    return result
                owner, fraction, path0, path1 = result
                parts.append((owner + start, fraction, path0, path1))
                start = stop
                continue
            runner.BACKEND.release_memory()
        if not parts:
            view = {
                **segments,
                **{key: xp.asarray(segments[key])[:0] for key in ("r_mid", "v_hat", "L_ang")},
            }
            return _segment_escape_paths(view, index, n_hat, layers=layers, groove=groove, xp=xp)
        if len(parts) == 1:
            return parts[0]
        return tuple(xp.concatenate(items) for items in zip(*parts, strict=True))


_CONTINUUM_ROWS = {
    "r_mid",
    "v_hat",
    "L_ang",
    "E_keV",
    "E_start_keV",
    "E_end_keV",
    "E_repr_keV",
    "elec_id",
    "electron_id",
    "layer",
}


def stage_continuum_rows(segments, electron_limit):
    """Stage only continuum consumers' fields and global background population."""
    import numpy as np

    from .. import runner

    ids = segments["elec_id"]
    xp = np if isinstance(ids, np.ndarray) else runner.BACKEND.xp
    index = xp.flatnonzero(ids < electron_limit)
    view = {
        key: value
        for key, value in segments.items()
        if key not in _SEG_ARRAYS or key in _CONTINUUM_ROWS
    }
    for key in _CONTINUUM_ROWS:
        if view.get(key) is not None:
            view[key] = view[key][index]
    return runner._segments_on_device(view)
