"""Opt-in HOPG fiber CXR and matched isotropic CBS continuum subtraction.

A fiber roll is an intensity average, not an amplitude average. The same
finite reflection set and thermal tensor are used for both emission routes.
No change to catalog texture or the historical basal-only default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...._fiber_average import FiberAverage
from ...._numerics import CxrReflections, Numerics
from ....materials.channeling.thermal import UniaxialThermalSite, thermal_site
from ....materials.crystal import CRYSTALS, _reflection_families, reciprocal_g_vector
from ..brem_lattice_share import _lattice_share_tables, _reflection_shell_weights
from ._spectrum import mc_cxr_spectrum

FIBER_CXR_MODEL = "hopg-fiber-intensity-tensor-isotropic-share-v1"


@dataclass(frozen=True, slots=True)
class FiberCXRModel:
    """Explicit finite-domain ideal fiber model for Bernal HOPG.

    ``numerics`` is the user-facing :class:`pyrite.config.FiberAverage`
    (reciprocal cutoff [1/Angstrom], uniform roll count, reflection batch
    size); ``temperature_K`` must be covered by the measured carbon tensor.
    Neither cutoff nor quadrature convergence is inferred from these inputs.
    The c axis is the positive sample normal before mosaic tilt and miscut.
    """

    numerics: FiberAverage
    temperature_K: float
    crystal: str = "hopg"

    @classmethod
    def from_numerics(
        cls,
        source: Numerics | CxrReflections | FiberAverage,
        *,
        temperature_K: float | None = None,
        crystal: str = "hopg",
    ) -> FiberCXRModel:
        """Build from ``Numerics``/``CxrReflections.fiber`` or a ``FiberAverage``.

        ``temperature_K`` defaults to the measured tensor's own temperature.
        """
        if isinstance(source, Numerics):
            source = source.cxr_reflections
        if isinstance(source, CxrReflections):
            if source.fiber is None:
                raise ValueError("CxrReflections.fiber is None (basal-only domain)")
            source = source.fiber
        if temperature_K is None:
            site = thermal_site(crystal, "C")
            if not isinstance(site, UniaxialThermalSite):
                raise ValueError("a measured uniaxial carbon tensor is required")
            temperature_K = site.temperature_K
        return cls(source, float(temperature_K), crystal)

    @property
    def cutoff_invang(self) -> float:
        return float(self.numerics.cutoff_invang)

    @property
    def azimuth_count(self) -> int:
        return int(self.numerics.azimuth_count)

    @property
    def reflection_batch_size(self) -> int:
        return int(self.numerics.reflection_batch_size)

    def __post_init__(self):
        from ...runner.reflection_cutoff import F0_RANGE_INVANG

        if not isinstance(self.numerics, FiberAverage):
            raise TypeError("numerics must be a FiberAverage")
        if self.crystal != "hopg" or CRYSTALS[self.crystal].get("texture") != "fiber":
            raise ValueError("full fiber CXR currently requires HOPG and its measured tensor")
        if self.cutoff_invang > F0_RANGE_INVANG:
            raise ValueError(f"cutoff_invang must lie in (0, {F0_RANGE_INVANG}]")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0:
            raise ValueError("temperature_K must be positive and finite")
        self._site().b_factor_along(1.0, self.temperature_K)

    def _site(self) -> UniaxialThermalSite:
        site = thermal_site(self.crystal, "C")
        if not isinstance(site, UniaxialThermalSite):
            raise ValueError("a measured uniaxial carbon tensor is required")
        return site

    def prepare(self) -> FiberReflectionSet:
        """Enumerate all non-extinct families, including all Friedel mates.

        Use zero scalar B for candidate enumeration; the measured tensor is
        applied to actual couplings, not used as a premature selection gate.

        Validation: cxr-fiber-intensity-average
        """
        families = _reflection_families(
            self.crystal, 1000.0, 0.0, False, self.cutoff_invang, full_fiber=True
        )
        strongest = max((metric for _, metric in families), default=0.0)
        hkls = tuple(
            sorted(
                {
                    hkl
                    for (_, metric), members in families.items()
                    if metric > 0.0 and metric >= strongest * 1e-9
                    for hkl in members
                }
            )
        )
        if not hkls:
            raise ValueError("cutoff contains no allowed reflections")
        lattice = CRYSTALS[self.crystal]["lattice"]
        axis, norm = reciprocal_g_vector((0, 0, 1), lattice)
        axis = axis / norm
        site = self._site()
        b_values = []
        for hkl in hkls:
            g, norm = reciprocal_g_vector(hkl, lattice)
            cos2 = min(1.0, max(0.0, round(float((g @ axis / norm) ** 2), 14)))
            b_values.append(float(site.b_factor_along(cos2, self.temperature_K)))
        return FiberReflectionSet(self, hkls, tuple(b_values))


@dataclass(frozen=True, slots=True)
class FiberReflectionSet:
    """Prepared domain shared by CXR and continuum; no mutable catalog state."""

    model: FiberCXRModel
    hkls: tuple[tuple[int, int, int], ...]
    b_ang2: tuple[float, ...]

    def __post_init__(self):
        if not self.hkls or len(self.hkls) != len(self.b_ang2):
            raise ValueError("reflection and thermal rows must be nonempty and aligned")
        if len(set(self.hkls)) != len(self.hkls) or (0, 0, 0) in self.hkls:
            raise ValueError("reflection set contains duplicate or zero vectors")
        values = dict(zip(self.hkls, self.b_ang2, strict=True))

        def same(hkl, b):
            # Projected B of symmetry mates agrees only to rounding.
            other = values.get(hkl)
            return other is not None and math.isclose(other, b, rel_tol=1e-12, abs_tol=1e-15)

        for hkl, b in values.items():
            if not math.isfinite(b) or b < 0 or not same(tuple(-i for i in hkl), b):
                raise ValueError("thermal rows must be finite, nonnegative and Friedel-even")
            # A roll by pi/3 about c maps hexagonal (h, k, l) to (-k, h + k, l);
            # the sector quadrature needs the set closed under it.
            if not same((-hkl[1], hkl[0] + hkl[1], hkl[2]), b):
                raise ValueError("reflection set must be closed under 60-degree rolls about c")

    def _thermal_lookup(self):
        return dict(zip(self.hkls, self.b_ang2, strict=True))

    def record(self):
        """Serializable numerical/physical identity for a caller's run record."""
        site = self.model._site()
        return dict(
            model=FIBER_CXR_MODEL,
            crystal=self.model.crystal,
            cutoff_invang=float(self.model.cutoff_invang),
            azimuth_count=int(self.model.azimuth_count),
            temperature_K=float(self.model.temperature_K),
            thermal_source=site.source,
            reflection_batch_size=int(self.model.reflection_batch_size),
            thermal_doi=site.doi,
            hkls=self.hkls,
            B_ang2=self.b_ang2,
        )

    def spectrum(self, segments, E_grid_eV, *, azimuth_rad=0.0, **kwargs):
        """Per-electron CXR density on a fixed grid using existing recoil kernels.

        Basal rows are evaluated once; all other reflections are scored in a
        batch per roll. PXR/CBS interference within each row is preserved.
        ``components=True`` is supported. No inter-domain coherence is modeled.

        Validation: cxr-fiber-intensity-average
        """
        _validate_scoring(azimuth_rad, kwargs)
        options = dict(kwargs)
        options.setdefault("line_quadrature", "bin-mean")
        if options.get("beam_uvw") is None and options.get("surface_hkl") is None:
            options["surface_hkl"] = (0, 0, 1)
        E_grid_eV = np.asarray(E_grid_eV, dtype=float)
        if (
            E_grid_eV.ndim != 1
            or E_grid_eV.size < 2
            or not np.isfinite(E_grid_eV).all()
            or E_grid_eV[0] < 10.0
            or np.any(np.diff(E_grid_eV) <= 0.0)
        ):
            raise ValueError("expected a finite increasing photon grid starting at 10 eV or above")
        lookup = self._thermal_lookup()

        def score(hkls, phi):
            return mc_cxr_spectrum(
                segments,
                E_grid_eV,
                self.model.crystal,
                hkls,
                B_ang2=lookup.__getitem__,
                azimuth_rad=phi,
                **options,
            )

        shape = (3, len(E_grid_eV)) if options.get("components") else (len(E_grid_eV),)
        total = self._average(score, shape, azimuth_rad)
        return tuple(total) if options.get("components") else total

    def _average(self, score, shape, azimuth_rad):
        """Uniform roll average: midpoint nodes on one 60-degree sector.

        The full-circle average of a sixfold-closed set equals its average over
        one 60-degree sector (each roll by pi/3 permutes the set), so ``N``
        sector midpoints are ``N`` distinct orientations. Equally spaced
        full-circle nodes alias whenever ``gcd(N, 6) > 1`` (``N = 6`` is one
        orientation). Closure is verified in ``__post_init__``. Basal rows are
        roll invariant and scored once. Batches bound table allocations only.

        Validation: cxr-fiber-intensity-average
        """
        basal = [h for h in self.hkls if h[:2] == (0, 0)]
        other = [h for h in self.hkls if h[:2] != (0, 0)]
        total = np.zeros(shape, dtype=np.float64)
        batch = int(self.model.reflection_batch_size)
        count = int(self.model.azimuth_count)
        for rows, rolls in ((basal, 1), (other, count)):
            for i in range(rolls):
                phi = (
                    azimuth_rad + (math.pi / 3.0) * (i + 0.5) / rolls
                    if rows is other
                    else azimuth_rad
                )
                for start in range(0, len(rows), batch):
                    value = np.asarray(score(rows[start : start + batch], phi), dtype=np.float64)
                    if value.shape != total.shape or not np.isfinite(value).all():
                        raise RuntimeError("invalid fiber spectrum from CXR scorer")
                    total += value / rolls
        return total

    def shell_weights(self, composition):
        """Orientation-invariant shell weights using exactly the scored tensor rows.

        Validation: cxr-fiber-intensity-average
        """
        composition = _carbon_composition(composition)
        return _reflection_shell_weights(
            self.model.crystal, self.hkls, self._thermal_lookup().__getitem__, composition
        )

    def corrected_bremslib_tables(self, tables, composition, *, E_cut_keV, max_energy_keV):
        """Emission-only BremsLib correction; sum the full share before clamping.

        The released universal shell kernel is isotropically orientation
        averaged, not a texture-conditioned diffuse model. The caller must
        score these tables on the same geometry and line-active population.

        Validation: cxr-fiber-intensity-average
        """
        from ...runner.lattice_share import _kernel

        composition = _carbon_composition(composition)
        if tables is None or "C" not in tables:
            raise ValueError("full fiber CXR requires a carbon BremsLib table")
        kernel = _kernel()
        g, weights = self.shell_weights(composition)
        if g[0] < kernel.g_invang[0] or g[-1] > kernel.g_invang[-1]:
            raise ValueError("fiber reflection domain exceeds the released subtraction kernel")
        if not (
            math.isfinite(E_cut_keV)
            and math.isfinite(max_energy_keV)
            and kernel.T_keV[0] <= E_cut_keV <= max_energy_keV <= kernel.T_keV[-1]
        ):
            raise ValueError("line-active energy interval exceeds the released subtraction kernel")
        entries = [(tables[el], el, n, weights[el]) for el, n in composition]
        results = _lattice_share_tables(entries, kernel, g, T_floor_keV=E_cut_keV)
        out = dict(tables)
        records = []
        for (_, el, _, _), (table, info) in zip(entries, results, strict=True):
            out[el] = table
            records.append(
                dict(
                    element=el,
                    removed_fraction=info.removed_fraction,
                    clamped_nodes=info.clamped_nodes,
                )
            )
        return out, dict(
            model=FIBER_CXR_MODEL,
            continuum="isotropic shell kernel",
            shells=int(g.size),
            elements=records,
        )


def _carbon_composition(composition):
    rows = [(str(el), float(n)) for el, n in composition]
    if len(rows) != 1 or rows[0][0] != "C" or not math.isfinite(rows[0][1]) or rows[0][1] <= 0:
        raise ValueError("full fiber CXR requires a single positive carbon composition row")
    return rows


def _validate_scoring(azimuth_rad, options):
    """Refuse options the fiber average does not support (route, then geometry)."""
    if not math.isfinite(azimuth_rad):
        raise ValueError("azimuth_rad must be finite")
    _validate_route(options)
    _validate_geometry(options)


def _validate_route(options):
    """Incoherent, deterministic, uncached scoring; the domain owns hkl and B."""
    if "B_ang2" in options or "hkl_list" in options or "crystal" in options:
        raise ValueError("the prepared fiber reflection set owns the domain and thermal factors")
    if any(
        options.get(key) is not None
        for key in ("truncation_audit", "temporal", "coefficient_capture")
    ):
        raise ValueError(
            "fiber-weighted audits, temporal and coefficient capture are not supported"
        )
    if options.get("coherent") or options.get("physical_electrons") is not None:
        raise ValueError("full fiber CXR supports incoherent emission only")
    if options.get("_table_cache") is not None:
        raise ValueError("cross-roll table cache is not supported")
    if options.get("mosaic_sampling", "quadrature") != "quadrature":
        raise ValueError("full fiber CXR currently supports deterministic mosaic quadrature only")


def _validate_geometry(options):
    """Positive c axis on the sample normal; one carbon slab."""
    for key in ("beam_uvw", "surface_hkl"):
        axis = options.get(key)
        if axis is not None and (len(axis) != 3 or axis[0] != 0 or axis[1] != 0 or axis[2] <= 0):
            raise ValueError("fiber CXR requires the positive c axis along the sample normal")
    if options.get("beam_uvw") is not None and options.get("surface_hkl") is not None:
        raise ValueError("beam_uvw and surface_hkl are mutually exclusive")
    if options.get("composition") is not None:
        _carbon_composition(options["composition"])
    layers = options.get("layers")
    if layers is not None and len(layers) != 1:
        raise ValueError("full fiber CXR currently supports a single slab only")
    if options.get("absorber_element", "C") != "C":
        raise ValueError("full fiber CXR requires carbon absorption")
