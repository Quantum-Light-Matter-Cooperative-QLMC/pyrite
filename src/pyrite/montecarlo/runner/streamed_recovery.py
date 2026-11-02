"""Bound scratch for the row-additive streamed characteristic/continuum slots."""

from dataclasses import replace

from ..spectrum.cxr._kernels import _SEG_ARRAYS


class RowScoringRecovery:
    """Learn a row ceiling per slot after allocator OOM, preserving normalization.

    These two built-in estimators add independent segment yields. CXR and TR
    require complete histories and must never pass through this row splitter.
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
