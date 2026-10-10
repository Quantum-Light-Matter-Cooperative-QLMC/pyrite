"""Conservative cost/observable gate for already certified phase reductions.

This module does not derive a physics certificate. A reduction owner supplies
certified retained coordinates using its existing accuracy-budget share.
Cost evidence must describe that exact reduction scope; inconclusive decisions
retain all work. No stochastic or ensemble certificate is inferred here.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseRetentionScope:
    """One proposed reduction, with counts and its accuracy reference.

    ``limit`` is the existing dimensionless omission share against the grouped
    per-electron floor at evaluated nodes. It is not relative error against
    the full spectrum, nor a new budget allocation. Segment count describes
    the prepared spectrum state; row masks may cover several rows together.
    ``row_vectors_inv_ang`` identifies those rows by their reciprocal vectors
    in inverse angstroms; physical population is separate from sampled count.
    """

    sector: str
    observable: str
    rule: str
    route: str
    backend: str
    precision: str
    energies: int
    retained: int
    rows: int
    segments: int
    electrons: int
    limit: float
    reference: str = "grouped-per-electron-floor"
    row_vectors_inv_ang: tuple[tuple[float, float, float], ...] = ()
    physical_electrons: float | None = None


@dataclass(frozen=True)
class PhaseRetentionCost:
    """Scoped timing estimates in seconds, including decision overhead.

    ``full_lower_s`` and ``reduced_upper_s`` are conservative estimates from
    paired measurements, not assumed hardware rates. ``overhead_upper_s``
    covers certification, host/device transfers, mask construction, this gate,
    the estimator and reporting. ``evidence`` identifies the measurement and
    workload applicability; matching scope alone cannot prove applicability.
    """

    scope: PhaseRetentionScope
    full_lower_s: float
    reduced_upper_s: float
    overhead_upper_s: float
    evidence: str


@dataclass(frozen=True)
class PhaseRetentionDecision:
    """Retained/skipped work and the reason for accepting or refusing it."""

    scope: PhaseRetentionScope
    simplify: bool
    reason: str
    cost: PhaseRetentionCost | None = None

    @property
    def retained(self) -> int:
        return self.scope.retained if self.simplify else self.scope.energies

    @property
    def skipped(self) -> int:
        return self.scope.energies - self.retained


@dataclass(frozen=True)
class PhaseRetentionPolicy:
    """Opt-in decision policy; absence of usable evidence keeps full phase.

    The estimator must return evidence for the supplied scope, including route,
    backend, precision and mask size. It must reject workloads outside its
    measured applicability. ``enabled=False`` explicitly disables simplification.
    ``report`` observes decisions; it must not alter the input or run transport.
    """

    estimate: Callable[[PhaseRetentionScope], PhaseRetentionCost | None] | None = None
    enabled: bool = True
    observable: str = "evaluated-spectrum"
    report: Callable[[PhaseRetentionDecision], None] | None = None

    def decide(self, scope: PhaseRetentionScope) -> PhaseRetentionDecision:
        """Accept a certified mask only with a strictly positive net saving."""
        cost = None
        if not self.enabled:
            reason = "disabled"
        elif (
            scope.sector != "inter-electron"
            or scope.observable != "evaluated-spectrum"
            or self.observable != scope.observable
            or scope.rule != "certified-flat-omission"
            or scope.reference != "grouped-per-electron-floor"
        ):
            reason = "unsupported-scope"
        elif not math.isfinite(scope.limit) or scope.limit <= 0:
            reason = "no-accuracy-share"
        elif self.estimate is None:
            # The owner can refuse before constructing a candidate mask.
            reason = "missing-cost-evidence"
        elif not 0 <= scope.retained < scope.energies:
            reason = "no-certified-work-to-skip"
        elif (cost := self.estimate(scope)) is None:
            reason = "missing-cost-evidence"
        elif cost.scope != scope:
            reason = "cost-scope-mismatch"
        elif not cost.evidence or not all(
            math.isfinite(value) and value >= 0
            for value in (cost.full_lower_s, cost.reduced_upper_s, cost.overhead_upper_s)
        ):
            reason = "invalid-cost-evidence"
        elif cost.full_lower_s <= cost.reduced_upper_s + cost.overhead_upper_s:
            reason = "no-net-saving"
        else:
            reason = "certified-and-profitable"
        decision = PhaseRetentionDecision(scope, reason == "certified-and-profitable", reason, cost)
        if self.report is not None:
            self.report(decision)
        return decision
