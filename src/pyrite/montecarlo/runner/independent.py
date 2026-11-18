"""Independent brem-prefix stopping on the shared electron transport (#366)."""

from dataclasses import replace

from .adaptive import StoppingMonitor, case_measure


class IndependentStoppingMonitor:
    """Apply the scalar rule separately, freezing only the brem prefix.

    CXR moments continue while brem is unfinished: its population must contain
    the entire brem prefix. At a CXR maximum the transport stops regardless.
    Each scalar uses the Chan/RSE estimators and guards of StoppingMonitor on
    its own decision points; the two grids are visited in electron order.
    Electrons past a brem freeze run to the line cutoff, so a freeze at
    ``d`` inside a chunk sets ``cut_at = d``: the driver keeps ``[0, d)`` and
    transports the rest again. Equal fixed bounds recover the shared rule;
    equal targets alone need not.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self, settings, cxr_measure, brem_measure=None):
        self.settings = settings
        cxr = replace(
            settings,
            observables=("cxr",),
            brem_target_rse=None,
            brem_min_histories=None,
            brem_max_histories=None,
        )
        self.cxr = StoppingMonitor(cxr, cxr_measure)
        self.brem = StoppingMonitor(settings.brem_policy(), brem_measure or cxr_measure)
        self.replayed_transport = False
        self.realized = None
        self.stop_reason = None
        self.cut_at = None

    @property
    def brem_count(self):
        return self.brem.realized

    @property
    def measured(self):
        return self.cxr.measured

    def on_block(self, start, stop, block):
        if start == 0:
            self.brem._reset()
            self.realized = self.stop_reason = self.cut_at = None
        self.cxr.on_block(start, stop, block)
        if self.brem_count is None:
            self.brem.on_block(start, stop, block)

    def truncate(self, n):
        self.cxr.truncate(n)
        self.cut_at = None

    def next_point(self):
        points = [self.cxr.next_point()]
        if self.brem_count is None:
            points.append(self.brem.next_point())
        points = [point for point in points if point is not None]
        return min(points) if points else None

    @staticmethod
    def _visit(monitor, point, recheck_at=None):
        if monitor.next_point() != point:
            return False
        return monitor.should_stop(point, recheck_at)

    def should_stop(self, n, recheck_at=None):
        """Visit both grids in electron order up to ``n``; ``True`` on the shared stop.

        With ``recheck_at`` (realized-range replay) brem still decides at each
        of its points, since its freeze sets later cutoffs, while CXR folds and
        decides only at ``recheck_at``.
        """
        self.cut_at = None
        limit = min(n, self.measured)
        while (point := self.next_point()) is not None and point <= limit:
            froze = self.brem_count is None and self._visit(self.brem, point)
            cxr_done = self._visit(self.cxr, point, recheck_at)
            if cxr_done and self.brem_count is not None:
                self.realized = point
                self.stop_reason = (
                    "converged"
                    if self.cxr.stop_reason == self.brem.stop_reason == "converged"
                    else "max_electrons"
                )
                return True
            if froze and point < n:
                self.cut_at = point
                return False
        return False

    def record(self):
        record = self.cxr.record()
        record.update(
            precision=self.settings.to_dict(),
            observables=list(self.settings.observables),
            realized_electrons=self.realized,
            realized_electrons_brem=self.brem_count,
            stop_reason=self.stop_reason,
            statistics_limited=self.stop_reason != "converged",
            replayed_transport=self.replayed_transport,
        )
        record["statistics"]["brem"] = self.brem.observable_status("brem")
        for name, monitor in (("cxr", self.cxr), ("brem", self.brem)):
            record["statistics"][name].update(
                target_rse=monitor.settings.target_rse,
                min_electrons=monitor.settings.min_histories,
                max_electrons=monitor.settings.max_histories,
                stop_reason=monitor.stop_reason,
                checks=monitor.checks,
                pilot=monitor.pilot,
            )
        return record


def case_stopping_monitor(case, settings):
    """Build only the active population's emission monitor for each block.

    Validation: adaptive-sample-size-stopping
    """
    if settings.brem_target_rse is None:
        return StoppingMonitor(settings, case_measure(case, settings))
    cxr = replace(
        settings,
        observables=("cxr",),
        brem_target_rse=None,
        brem_min_histories=None,
        brem_max_histories=None,
    )
    return IndependentStoppingMonitor(
        settings, case_measure(case, cxr), case_measure(case, settings.brem_policy())
    )
