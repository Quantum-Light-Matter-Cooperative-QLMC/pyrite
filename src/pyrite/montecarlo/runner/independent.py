"""Independent brem-prefix stopping on the shared electron transport (#366)."""

from dataclasses import replace

from .adaptive import StoppingMonitor, case_measure


class IndependentStoppingMonitor:
    """Apply the scalar rule separately, freezing only the brem prefix.

    CXR moments continue while brem is unfinished: its population must contain
    the entire brem prefix. At a CXR maximum the transport stops regardless.
    Each scalar uses the Chan/RSE estimators and guards of StoppingMonitor.
    Equal fixed bounds recover the shared rule; equal targets alone need not.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self, settings, cxr_measure, brem_measure=None):
        self.settings = settings
        cxr = replace(
            settings,
            observables=("cxr",),
            brem_target_rse=None,
            brem_min_electrons=None,
            brem_max_electrons=None,
        )
        self.cxr = StoppingMonitor(cxr, cxr_measure)
        self.brem = StoppingMonitor(settings.brem_policy(), brem_measure or cxr_measure)
        self.replayed_transport = False
        self.realized = None
        self.stop_reason = None

    @property
    def brem_count(self):
        return self.brem.realized

    def on_block(self, start, stop, block):
        if start == 0:
            self.brem._reset()
            self.realized = self.stop_reason = None
        self.cxr.on_block(start, stop, block)
        if self.brem_count is None:
            self.brem.on_block(start, stop, block)

    def should_stop(self, n):
        if self.brem_count is None:
            self.brem.should_stop(n)
        cxr_done = self.cxr.should_stop(n)
        if cxr_done and self.brem_count is not None:
            self.realized = n
            self.stop_reason = (
                "converged"
                if self.cxr.stop_reason == self.brem.stop_reason == "converged"
                else "max_electrons"
            )
            return True
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
                min_electrons=monitor.settings.min_electrons,
                max_electrons=monitor.settings.max_electrons,
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
        brem_min_electrons=None,
        brem_max_electrons=None,
    )
    return IndependentStoppingMonitor(
        settings, case_measure(case, cxr), case_measure(case, settings.brem_policy())
    )
