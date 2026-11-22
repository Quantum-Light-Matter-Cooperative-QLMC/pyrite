"""Opt-in, flushed worker evidence that survives an interrupted pytest session."""

import faulthandler
import json
import os
import time
from pathlib import Path


def configure_test_diagnostics(config):
    directory = os.environ.get("PYRITE_TEST_DIAGNOSTICS")
    if not directory:
        return
    worker = getattr(config, "workerinput", {}).get("workerid")
    if worker is None and config.getoption("numprocesses", default=0):
        return  # Workers own the tests; the controller only waits for reports.
    config.pluginmanager.register(WorkerDiagnostics(Path(directory), worker or "serial"))


class WorkerDiagnostics:
    def __init__(self, directory, worker, *, stack_interval=120):
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{worker}-{os.getpid()}"
        self.events = (directory / f"{stem}.jsonl").open("a", buffering=1)
        self.stacks = (directory / f"{stem}-stacks.log").open("a", buffering=1)
        self.stack_interval = stack_interval

    def pytest_sessionstart(self):
        self._arm_stacks()

    def _arm_stacks(self):
        faulthandler.dump_traceback_later(self.stack_interval, repeat=True, file=self.stacks)

    def _record(self, **event):
        self.events.write(json.dumps({"time": time.time(), **event}) + "\n")

    def pytest_runtest_logstart(self, nodeid, location):
        self._record(event="start", nodeid=nodeid)
        self._arm_stacks()

    def pytest_runtest_logreport(self, report):
        self._record(
            event=report.when,
            nodeid=report.nodeid,
            outcome=report.outcome,
            duration=report.duration,
        )
        # pytest's interactive-exception hook can cancel the global watchdog.
        if report.outcome == "failed":
            self._arm_stacks()

    def pytest_runtest_logfinish(self, nodeid, location):
        self._record(event="finish", nodeid=nodeid)

    def pytest_unconfigure(self):
        faulthandler.cancel_dump_traceback_later()
        self.events.close()
        self.stacks.close()
