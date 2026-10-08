"""Opt-in full-axis coherent yield/centroid checks on the production capture.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Mapping
from functools import partial
from time import perf_counter

import numpy as np

from ..._backend import _to_cpu
from ..._line_grid_policy import LineGridToleranceError
from ..spectrum.coherent_dispersion import CoherentDispersionLaw
from ..spectrum.coherent_form_factor import gaussian_form_factor_bounds
from ..spectrum.coherent_spectrum_audit import (
    FullAxisCentroidAudit,
    audit_full_axis_spectrum_centroid,
    audit_full_axis_spectrum_yield,
)
from ..spectrum.coherent_windows import CoherentRowCollector
from ..transport.kinematics import C_ANG_PER_FS
from .line_grid import longitudinal_rms_fs


class CoherentGridAudit:
    """Capture and check one spectrum without a second transport or axis change.

    Developer instrumentation is enabled with case['_coherent_yield_audit']
    containing relative_tolerance, initial_samples and/or max_evaluations.
    centroid_relative_tolerance enables the first-moment check; integration_bins
    controls its complete-axis partition without changing production samples.
    It refuses unsupported capture scopes and a failed or exhausted audit.
    No default runtime certificate is asserted by this opt-in check.

    Validation: coherent-line-grid-windowed-resolution
    """

    def __init__(self, case):
        options = case["_coherent_yield_audit"]
        if not isinstance(options, Mapping) or set(options) - {
            "relative_tolerance",
            "initial_samples",
            "max_evaluations",
            "integration_bins",
            "centroid_relative_tolerance",
        }:
            raise ValueError("coherent yield audit needs a mapping of tolerance/sample budgets")
        if not case.get("coherent_emission"):
            raise ValueError("coherent yield audit requires coherent_emission")
        if case.get("layer_radiators") is not None or case.get("sinc_cutoff") is not None:
            raise ValueError("coherent yield audit supports one radiator without sinc_cutoff")
        self.options = dict(options)
        self.case = case
        self.collector = CoherentRowCollector()
        self.factor_bounds = (0.0, 1.0)

    def capture(self, st, idx, coefs, good, lines):
        """Use the production setup's actual coherence sector, then capture rows."""
        if not st.decoherence_active:
            self.factor_bounds = (1.0, 1.0)
        elif st.finite_footprint_now:
            self.factor_bounds = partial(
                gaussian_form_factor_bounds,
                sigma_z_ang=longitudinal_rms_fs(self.case) * C_ANG_PER_FS,
            )
        # An infinite slab's row-dependent empirical factor remains [0,1].
        # A budget can refuse that loose enclosure; it cannot license F=0.
        self.collector(st, idx, coefs, good, lines)

    def check(self, energy_eV, source, electron_count):
        """Return scalar provenance only after every requested comparison passes."""
        started = perf_counter()
        energy = np.asarray(_to_cpu(energy_eV), dtype=float)
        want_centroid = "centroid_relative_tolerance" in self.options
        check = (
            audit_full_axis_spectrum_centroid if want_centroid else audit_full_axis_spectrum_yield
        )
        result = check(
            self.collector.rows,
            CoherentDispersionLaw(self.case["crystal"], float(energy[0]), float(energy[-1])),
            energy,
            np.asarray(_to_cpu(source), dtype=float),
            electron_count=electron_count,
            form_factor_bounds=self.factor_bounds,
            **self.options,
        )
        audit = result.yield_audit if isinstance(result, FullAxisCentroidAudit) else result
        if not audit.within_tolerance:
            raise LineGridToleranceError(
                "coherent full-axis yield audit failed: relative error upper "
                f"{audit.relative_error_upper:.6g} exceeds {audit.spectrum.relative_tolerance:g} "
                f"on {audit.points} coordinates over [{audit.start_eV:g}, {audit.stop_eV:g}] eV; "
                "refine the explicit grid or window policy; the grid is not coarsened"
            )
        if isinstance(result, FullAxisCentroidAudit) and not result.within_tolerance:
            raise LineGridToleranceError(
                "coherent full-axis centroid audit failed: relative error upper "
                f"{result.relative_error_upper:.6g} exceeds {result.relative_tolerance:g} "
                f"on {audit.points} coordinates over [{audit.start_eV:g}, {audit.stop_eV:g}] eV; "
                "refine the spectrum grid or integration partition; no acceptance is returned"
            )
        record = {
            "scope": "stored-input full finite-axis yield",
            "start_eV": audit.start_eV,
            "stop_eV": audit.stop_eV,
            "points": audit.points,
            "rows": len(self.collector.rows),
            "evaluations": audit.spectrum.evaluations,
            "yield_lower": audit.spectrum.lower,
            "yield_upper": audit.spectrum.upper,
            "quadrature_lower": audit.quadrature_lower,
            "quadrature_upper": audit.quadrature_upper,
            "relative_error_upper": audit.relative_error_upper,
            "relative_tolerance": audit.spectrum.relative_tolerance,
            "wall_s": perf_counter() - started,
        }
        if isinstance(result, FullAxisCentroidAudit):
            record.update(
                scope="stored-input full finite-axis yield and centroid",
                centroid_lower_eV=result.centroid_lower_eV,
                centroid_upper_eV=result.centroid_upper_eV,
                quadrature_centroid_lower_eV=result.quadrature_centroid_lower_eV,
                quadrature_centroid_upper_eV=result.quadrature_centroid_upper_eV,
                centroid_relative_error_upper=result.relative_error_upper,
                centroid_relative_tolerance=result.relative_tolerance,
            )
        return record
