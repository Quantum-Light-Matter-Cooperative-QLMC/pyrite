"""Completion accounting for scalar configs, groups, and all execution work."""

from collections import defaultdict


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
        for group in sorted(g for g in self.group_names if self.group_remaining.get(g, 0) == 0):
            rep = next(iter(results[self.group_names[group][0]].values()))["case"]
            print(
                f"\n=== cached: {rep['thickness_ang'] / 1e4:g} um, "
                f"tilt {rep['tilt_deg']:g} deg (already computed) ==="
            )
            self.on_chunk(self.group_names[group])
