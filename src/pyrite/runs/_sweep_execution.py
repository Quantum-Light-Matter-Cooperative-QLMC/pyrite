"""Completion accounting, the execution budget, and worker-output resolution."""

from collections import defaultdict
from contextlib import contextmanager

from ..execution.artifacts import (
    CaseArtifactRef,
    DirectionalArtifactRef,
    load_directional_result,
)


class SweepBudget:
    """Soft wall-clock stop; an absolute ``deadline`` wins over ``max_seconds``.

    A ``deadline`` is already running when the sweep is called, so resume and
    cache replay spend it. ``max_seconds`` starts at :meth:`start`, immediately
    before dispatch, so they do not. Both are read through ``time_fn``.
    """

    def __init__(self, deadline, max_seconds, time_fn):
        self.deadline = deadline
        self.max_seconds = max_seconds
        self.time_fn = time_fn

    @property
    def should_stop(self):
        """The ``run_cases`` hook, or None for an unbounded sweep."""
        bounded = self.deadline is not None or self.max_seconds is not None
        return self.expired if bounded else None

    def start(self):
        if self.deadline is None and self.max_seconds is not None:
            self.deadline = self.time_fn() + self.max_seconds

    def expired(self):
        return self.time_fn() >= self.deadline


@contextmanager
def terminal_activity(on_activity):
    """Report no in-flight work however the sweep exits, without masking a failure."""
    if on_activity is None:
        yield
        return
    idle = {"phase": "handoff", "case": None, "in_flight_case_count": 0}
    try:
        yield
    except BaseException as error:
        try:
            on_activity(idle)
        except Exception as callback_error:
            error.add_note(f"on_activity also failed during sweep failure: {callback_error!r}")
        raise
    on_activity(idle)


def resolve_worker_output(case, out, *, artifact_store, content_key, observation_plans, material):
    """Verify a worker-published reference and load it; other output passes through."""
    if isinstance(out, DirectionalArtifactRef):
        if artifact_store is None:
            raise ValueError("executor returned a directional artifact without artifact_store")
        key = content_key(case)
        if key is None:
            raise ValueError("cannot resolve key for worker-published observation")
        plan = observation_plans.get((case["name"], case["E0_keV"]))
        if plan is None:
            raise ValueError("executor returned unexpected directional artifact")
        return load_directional_result(
            artifact_store,
            out,
            material=material,
            case_key=key,
            directions=plan.directions_sample,
        )
    if isinstance(out, CaseArtifactRef):
        if artifact_store is None:
            raise ValueError("executor returned a case artifact without artifact_store")
        key = content_key(case)
        if key is None:
            raise ValueError("cannot resolve content key for worker-published case")
        return artifact_store.load(
            out,
            expected_material=material,
            expected_content_key=key,
        )
    return out


class SweepCompletion:
    """Observation-only work advances progress without completing scalar groups.

    A config must be persisted before its group callback can fire. Progress
    advances after that callback, retaining the sweep's historical event order.
    """

    def __init__(self, cases, todo, group_key, case_cost_fn, on_cost, on_progress, on_chunk):
        self.cases = cases
        self.cached_cases = len(cases) - len(todo)
        self.group_key = group_key
        self.case_cost_fn = case_cost_fn
        self.on_cost = on_cost
        self.on_progress = on_progress
        self.on_chunk = on_chunk
        self.completed_new_cases = 0
        self.total_cost = sum(case_cost_fn(c) for c in cases) if case_cost_fn is not None else None
        self.done_cost = (
            self.total_cost - sum(case_cost_fn(c) for c in todo)
            if case_cost_fn is not None
            else None
        )

    def prepare_groups(self, todo, observation_only):
        group_key = self.group_key
        self.group_names = defaultdict(list)
        seen = set()
        for case in self.cases:
            if case["name"] not in seen:
                seen.add(case["name"])
                self.group_names[group_key(case)].append(case["name"])
        self.group_remaining = defaultdict(int)
        self.energies_remaining = {}
        for case in todo:
            if (case["name"], case["E0_keV"]) in observation_only:
                continue
            name = case["name"]
            if name not in self.energies_remaining:
                self.group_remaining[group_key(case)] += 1
            self.energies_remaining[name] = self.energies_remaining.get(name, 0) + 1
        self.n_groups = len(self.group_remaining)
        self.groups_done = 0

    def report(self):
        if self.on_cost is not None and self.case_cost_fn is not None:
            self.on_cost(self.done_cost, self.total_cost)
        if self.on_progress is not None:
            self.on_progress(self.completed_new_cases, len(self.cases), self.cached_cases)

    def case_finished(self, case):
        self.completed_new_cases += 1
        if self.case_cost_fn is not None:
            self.done_cost += self.case_cost_fn(case)
        self.report()

    def scalar_finished(self, case, checkpoint):
        name = case["name"]
        self.energies_remaining[name] -= 1
        if self.energies_remaining[name] != 0:
            return
        checkpoint.config_finished(name)
        group = self.group_key(case)
        self.group_remaining[group] -= 1
        if self.group_remaining[group] == 0 and self.on_chunk is not None:
            self.groups_done += 1
            print(
                f"\n=== {case['thickness_ang'] / 1e4:g} um, tilt "
                f"{case['tilt_deg']:g} deg done "
                f"-- {self.groups_done}/{self.n_groups} tilt-groups ==="
            )
            self.on_chunk(self.group_names[group])

    def replay_cached_groups(self, results):
        if self.on_chunk is None:
            return
        # Group keys promise hashability, not ordering: replay in request order.
        for group in self.group_names:
            if self.group_remaining.get(group, 0) != 0:
                continue
            rep = next(iter(results[self.group_names[group][0]].values()))["case"]
            print(
                f"\n=== cached: {rep['thickness_ang'] / 1e4:g} um, "
                f"tilt {rep['tilt_deg']:g} deg (already computed) ==="
            )
            self.on_chunk(self.group_names[group])
