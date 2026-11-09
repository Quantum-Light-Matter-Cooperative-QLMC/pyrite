r"""Measured axial photon-source depth calibrations from paired batches (#993).

Research producer for the #884 contract. It turns chunk-paired fibre-source
arms (``checks/axial_fiber_source.py`` with ``checkpoints_ang``) and #883
unit-weight fate samples (``checks/axial_fiber_fates.py``) into an
:class:`.AxialPhotonDepthEvidence` for **yield at crystal thickness** in
intervals $(c-s, c]$ ending at checkpoint thicknesses $c$. Constructing the
result is not admission; #994 owns the repository lookup.

Statistics. Every arm shares seed and incident chunks, so per-chunk means
$x_b$ (counts $n_b$, $B$ chunks) of any paired linear combination are
independent across chunks. A bound uses the count-weighted mean $\bar x$ and
batch standard error $\hat\sigma=\sqrt{\sum_b n_b(x_b-\bar x)^2/((B-1)N)}$
with a Student-$t_{B-1}$ quantile; the error rate of each interval is split
evenly (Bonferroni) over all its one-sided bounds, an absolute value
counting two. **Assumption:** batch means are close to normal; the record
keeps a batch-doubling ratio as a check.

Per interval and observable bin, with $Y_a(z)$ the arm-$a$ yield at slab
thickness $z$ and $D_a=Y_a-Y_\mathrm{ctl}$:

* reference $r=\bar Y_\mathrm{ctl}(c)$, statistical lower allowance
  $t\hat\sigma$, drift allowance $\kappa\max_z[\overline{Y(c)-Y(z)}+t\hat\sigma]^+$
  over recorded $z\in[c-s,c)$;
* probe $a$: candidate $\bar Y_a(c)$, paired statistical bound on $D_a(c)$
  and drift $\kappa\max_z(|\overline{D_a(z)-D_a(c)}|+t\hat\sigma)$;
* interaction (all-on minus the sum of single probes),
  $|\bar I|+t\hat\sigma$ plus its drift, added to the ``source_model`` probe's
  statistical bound (the #884 "allowance inside one of the probe bounds").

$\kappa\ge1$ (default 2) covers variation between recorded thicknesses; the
grid step is below a quarter of the main-line formation length
$2\gamma^2\hbar c/E$.

Fate caps (mean-response caps, not worst-case): an outgoing electron's
photon error is its omitted return emission, measured as
$C_o=\max_c \mathrm{UCB}|D_\mathrm{all}(c)|/\mathrm{LCB}[F^\mathrm{ctl}_o(c)-F^\mathrm{all}_o(c)]$.
An unresolved electron omits at most the mean emission of an electron
retained to $c$, $C_u=\max_c\mathrm{UCB}\,Y_\mathrm{ctl}(c)/\mathrm{LCB}\,F_r(c)$,
assuming group-conditional mean emission is nonnegative. Fate fractions use
one-sided Hoeffding bounds; $Y$ bounds use the batch-$t$ rule. The cap failure
probability is the Bonferroni total of these bounds.

Units: photons per incident electron per sr in each integrated band; depth Å.

Validation: cr-axial-depth-calibration
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from scipy.stats import t as student_t

from .axial_depth_budget import AxialPhotonDepthEvidence
from .axial_depth_measurements import (
    AxialPhotonPairedProbe,
    AxialPhotonSourceIntervalMeasurement,
    compile_axial_photon_depth_evidence,
)
from .axial_fate_source_bounds import AxialFatePhotonResponseCaps, add_physical_fate_source_bounds
from .axial_outgoing_launch import AxialHandoffRecord

PROBES = ("return_flux", "unresolved", "matching_surface", "grid_supercell", "source_model")


@dataclass(frozen=True)
class ArmBatches:
    """Chunk means of one arm: ``yields[b, t, j]`` at ``thickness_ang[t]`` in band ``j``.

    ``counts[b]`` primaries per chunk; ``starts[b]`` the first incident ID,
    shared with every paired arm.

    Validation: cr-axial-depth-calibration
    """

    thickness_ang: tuple[float, ...]
    starts: tuple[int, ...]
    counts: np.ndarray
    yields: np.ndarray

    def at(self, thickness: float) -> np.ndarray:
        """Chunk means ``(B, bands)`` at one recorded thickness."""
        hit = [i for i, z in enumerate(self.thickness_ang) if abs(z - thickness) < 1e-6]
        if len(hit) != 1:
            raise ValueError(f"thickness {thickness} A is not recorded by this arm")
        return self.yields[:, hit[0]]


def batch_mean_se(values: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count-weighted mean and batch standard error over axis 0.

    Validation: cr-axial-depth-calibration
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or counts.size < 2 or np.any(counts <= 0):
        raise ValueError("at least two chunks with positive counts are required")
    w = counts.reshape((-1,) + (1,) * (np.ndim(values) - 1))
    total = counts.sum()
    mean = np.sum(w * values, axis=0) / total
    var = np.sum(w * (values - mean) ** 2, axis=0) / ((counts.size - 1) * total)
    return mean, np.sqrt(var)


def _t(alpha: float, batches: int) -> float:
    return float(student_t.ppf(1.0 - alpha, batches - 1))


def _hoeffding(n: int, alpha: float) -> float:
    return sqrt(log(1.0 / alpha) / (2 * n))


def _common_shift(arms: Sequence[ArmBatches], checkpoint: float, span: float) -> list[float]:
    """Recorded thicknesses in ``[c - span, c)`` shared by every arm."""
    shared = set.intersection(
        *(
            {round(z, 6) for z in a.thickness_ang if checkpoint - span - 1e-6 <= z < checkpoint}
            for a in arms
        )
    )
    if not shared:
        raise ValueError(f"no shared recorded thickness in [{checkpoint - span}, {checkpoint})")
    return sorted(shared)


def _check_paired(arms: Mapping[str, ArmBatches]) -> int:
    first = next(iter(arms.values()))
    for name, arm in arms.items():
        if arm.starts != first.starts or not np.array_equal(arm.counts, first.counts):
            raise ValueError(f"arm {name!r} is not chunk-paired with the control")
    return len(first.starts)


def interval_measurement(
    arms: Mapping[str, ArmBatches],
    *,
    checkpoint_ang: float,
    span_ang: float,
    alpha: float,
    drift_factor: float = 2.0,
    evidence_prefix: str,
) -> tuple[AxialPhotonSourceIntervalMeasurement, dict]:
    """One interval ``(c - span, c]`` at joint error rate ``alpha``.

    ``arms`` holds ``control``, ``all_on`` and one arm per name in
    :data:`PROBES`. Returns the #884 measurement and an audit dictionary.

    Validation: cr-axial-depth-calibration
    """
    names = ("control", "all_on", *PROBES)
    if set(arms) != set(names):
        raise ValueError(f"arms must be exactly {names}")
    if drift_factor < 1:
        raise ValueError("drift_factor must be at least 1")
    batches = _check_paired(arms)
    counts = arms["control"].counts
    shifts = _common_shift(list(arms.values()), checkpoint_ang, span_ang)
    bands = arms["control"].yields.shape[-1]
    # 1 one-sided reference bound per checkpoint and per shift, two-sided paired
    # bounds for 5 probes + interaction at c and at each shift.
    n_bounds = bands * (1 + len(shifts) + 2 * 6 * (1 + len(shifts)))
    tq = _t(alpha / n_bounds, batches)  # two-sided bounds were counted twice above

    def upper_abs(values: np.ndarray) -> np.ndarray:
        mean, se = batch_mean_se(values, counts)
        return np.abs(mean) + tq * se

    ctl_c = arms["control"].at(checkpoint_ang)
    ref, ref_se = batch_mean_se(ctl_c, counts)
    ref_drift = np.zeros(bands)
    for z in shifts:
        mean, se = batch_mean_se(ctl_c - arms["control"].at(z), counts)
        ref_drift = np.maximum(ref_drift, mean + tq * se)
    ref_drift = drift_factor * np.maximum(ref_drift, 0.0)

    def difference(name: str, z: float) -> np.ndarray:
        if name == "interaction":
            single = sum(difference(p, z) for p in PROBES)
            return difference("all_on", z) - single
        return arms[name].at(z) - arms["control"].at(z)

    def drift(name: str) -> np.ndarray:
        d_c = difference(name, checkpoint_ang)
        worst = np.zeros(bands)
        for z in shifts:
            worst = np.maximum(worst, upper_abs(difference(name, z) - d_c))
        return drift_factor * worst

    probes, audit = {}, {"shifts_ang": shifts, "t_quantile": tq, "bounds": n_bounds}
    interaction = upper_abs(difference("interaction", checkpoint_ang)) + drift("interaction")
    for name in PROBES:
        d_mean, d_se = batch_mean_se(difference(name, checkpoint_ang), counts)
        stat = tq * d_se + (interaction if name == "source_model" else 0.0)
        probes[name] = AxialPhotonPairedProbe(
            candidate_yield_bins=(ref + d_mean).tolist(),
            difference_stat_error_upper_bins=stat.tolist(),
            difference_depth_drift_upper_bins=drift(name).tolist(),
            evidence_id=f"{evidence_prefix}:{name}",
        )
        audit[name] = {"difference": d_mean.tolist(), "se": d_se.tolist()}
    audit["interaction_upper"] = interaction.tolist()
    measurement = AxialPhotonSourceIntervalMeasurement(
        start_depth_ang=checkpoint_ang - span_ang,
        stop_depth_ang=checkpoint_ang,
        reference_yield_bins=ref.tolist(),
        reference_stat_error_upper_bins=(tq * ref_se).tolist(),
        reference_depth_drift_upper_bins=ref_drift.tolist(),
        return_flux=probes["return_flux"],
        unresolved=probes["unresolved"],
        matching_surface=probes["matching_surface"],
        grid_supercell=probes["grid_supercell"],
        source_model=probes["source_model"],
        evidence_id=f"{evidence_prefix}:reference",
        simultaneous_confidence=1.0 - alpha,
    )
    audit["reference"] = {"mean": ref.tolist(), "se": ref_se.tolist()}
    return measurement, audit


def _fractions(records: Sequence[AxialHandoffRecord], depth: float) -> dict[str, float]:
    n = len(records)
    out = sum(1 for r in records if r.fate == "physical_outgoing" and r.depth_ang <= depth)
    unresolved = sum(1 for r in records if r.fate == "unresolved" and r.depth_ang <= depth)
    return {
        "outgoing": out / n,
        "unresolved": unresolved / n,
        "retained": 1 - (out + unresolved) / n,
    }


def response_caps(
    arms: Mapping[str, ArmBatches],
    control_fates: Sequence[AxialHandoffRecord],
    all_on_fates: Sequence[AxialHandoffRecord],
    *,
    checkpoints_ang: Sequence[float],
    failure_probability: float,
    case_key: str,
    observable_key: str,
    model_id: str,
    evidence_id: str,
) -> tuple[AxialFatePhotonResponseCaps, dict]:
    """Measured mean-response caps ``C_o`` and ``C_u`` over all checkpoints.

    Per checkpoint: a two-sided yield bound and two Hoeffding fractions for
    ``C_o``; a one-sided yield bound and one Hoeffding fraction for ``C_u``.
    A nonpositive lower bound on the returned outgoing fraction or on the
    retained fraction refuses (no cap, hence no budget).

    Validation: cr-axial-depth-calibration
    """
    batches = _check_paired(arms)
    counts = arms["control"].counts
    n_bounds = len(checkpoints_ang) * (3 * arms["control"].yields.shape[-1] + 3)
    alpha = failure_probability / n_bounds
    tq = _t(alpha, batches)
    c_out = np.zeros(arms["control"].yields.shape[-1])
    c_unres = np.zeros_like(c_out)
    audit = []
    h_ctl = _hoeffding(len(control_fates), alpha)
    h_all = _hoeffding(len(all_on_fates), alpha)
    for c in checkpoints_ang:
        f_ctl, f_all = _fractions(control_fates, c), _fractions(all_on_fates, c)
        returned = f_ctl["outgoing"] - h_ctl - (f_all["outgoing"] + h_all)
        retained = f_ctl["retained"] - h_ctl
        if returned <= 0 or retained <= 0:
            raise ValueError(f"no positive lower fate bound at {c} A: response cap undefined")
        mean, se = batch_mean_se(arms["all_on"].at(c) - arms["control"].at(c), counts)
        ctl, ctl_se = batch_mean_se(arms["control"].at(c), counts)
        c_out = np.maximum(c_out, (np.abs(mean) + tq * se) / returned)
        c_unres = np.maximum(c_unres, (ctl + tq * ctl_se) / retained)
        audit.append({"checkpoint_ang": c, "control": f_ctl, "all_on": f_all})
    caps = AxialFatePhotonResponseCaps(
        case_key=case_key,
        observable_key=observable_key,
        model_id=model_id,
        evidence_id=evidence_id,
        outgoing_return_error_upper_bins=c_out.tolist(),
        unresolved_error_upper_bins=c_unres.tolist(),
        failure_probability_upper=failure_probability,
    )
    return caps, {"fractions": audit, "hoeffding": [h_ctl, h_all], "t_quantile": tq}


def calibrate_axial_depth(
    arms: Mapping[str, ArmBatches],
    control_fates: Sequence[AxialHandoffRecord],
    all_on_fates: Sequence[AxialHandoffRecord],
    *,
    checkpoints_ang: Sequence[float],
    span_ang: float,
    sampled_max_depth_ang: float,
    case_key: str,
    observable_key: str,
    model_id: str,
    evidence_id: str,
    source_risk: float = 0.03,
    sampling_risk: float = 0.01,
    cap_risk: float = 0.01,
    drift_factor: float = 2.0,
) -> tuple[AxialPhotonDepthEvidence, dict]:
    """Compile the depth budget; joint confidence ``1 - source - sampling - cap`` risk.

    Validation: cr-axial-depth-calibration
    """
    if not checkpoints_ang:
        raise ValueError("at least one checkpoint thickness is required")
    alpha = source_risk / len(checkpoints_ang)
    rows, audits = [], []
    for c in checkpoints_ang:
        row, audit = interval_measurement(
            arms,
            checkpoint_ang=float(c),
            span_ang=span_ang,
            alpha=alpha,
            drift_factor=drift_factor,
            evidence_prefix=f"{evidence_id}@{c:g}",
        )
        rows.append(row)
        audits.append(audit)
    source = compile_axial_photon_depth_evidence(
        case_key=case_key,
        observable_key=observable_key,
        model_id=model_id,
        evidence_id=f"{evidence_id}:source",
        measurements=rows,
    )
    caps, cap_audit = response_caps(
        arms,
        control_fates,
        all_on_fates,
        checkpoints_ang=checkpoints_ang,
        failure_probability=cap_risk,
        case_key=case_key,
        observable_key=observable_key,
        model_id=model_id,
        evidence_id=f"{evidence_id}:caps",
    )
    evidence = add_physical_fate_source_bounds(
        source,
        control_fates,
        sampled_max_depth_ang=sampled_max_depth_ang,
        caps=caps,
        sample_evidence_id=f"{evidence_id}:fates",
        sampling_failure_probability=sampling_risk,
    )
    report = {
        "intervals": [
            {
                "start_ang": i.start_depth_ang,
                "stop_ang": i.stop_depth_ang,
                "reference_lower": i.reference_lower,
                "return_upper": i.return_upper,
                "unresolved_upper": i.unresolved_upper,
                "grid_supercell_upper": i.grid_supercell_upper,
                "source_model_upper": i.source_model_upper,
                "relative_error_upper": i.relative_error_upper,
                "audit": a,
            }
            for i, a in zip(evidence.intervals, audits, strict=True)
        ],
        "simultaneous_confidence": evidence.simultaneous_confidence,
        "caps": {
            "outgoing": caps.outgoing_return_error_upper_bins,
            "unresolved": caps.unresolved_error_upper_bins,
            **cap_audit,
        },
    }
    return evidence, report
