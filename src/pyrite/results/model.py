"""Returned value for the public single-shot simulation API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..instrument import PlanarDetector
from ..montecarlo import Case
from .provenance import Provenance
from .spatial import (
    COMPONENTS,
    PixelMetadata,
    PixelRayMap,
    SpatialResult,
    SpectralFactors,
)

FluxUnits = Literal["detector", "nA", "sr", "electron"]

#: Units of each :meth:`Result.spectral_flux` ``units`` choice.
FLUX_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "detector": "photons / (s eV)",
        "nA": "photons / (s nA eV)",
        "sr": "photons / (s eV sr)",
        "electron": "photons / (electron eV sr)",
    }
)

if TYPE_CHECKING:
    import polars as pl

    from .._numerics import Numerics
    from ..campaign.model import Scene
    from ..instrument import AcquisitionBatch


@dataclass(frozen=True)
class SpectralDensity:
    """One radiation component paired with its own photon-energy grid.

    photon_energy_eV
        One-dimensional photon-energy coordinates in eV.
    density
        Density on exactly that grid, in photons / incident electron / eV / sr.
    normalization
        ``source`` means response-free source density without acceptance or
        current scaling. ``observation-average`` means filter-attenuated,
        solid-angle-weighted physical-detector average, still per sr.
    """

    photon_energy_eV: np.ndarray
    """Photon-energy coordinates in eV; this component's own grid."""
    density: np.ndarray
    """Photons / incident electron / eV / sr on :attr:`photon_energy_eV`."""
    normalization: str = "source"
    """``"source"`` (scalar detector) or ``"observation-average"`` (planar detector)."""

    def __post_init__(self) -> None:
        energy = np.asarray(self.photon_energy_eV)
        density = np.asarray(self.density)
        if energy.ndim != 1 or density.shape != energy.shape:
            raise ValueError("density must match its one-dimensional photon_energy_eV grid")
        if self.normalization not in {"source", "observation-average"}:
            raise ValueError("normalization must be 'source' or 'observation-average'")
        object.__setattr__(self, "photon_energy_eV", energy)
        object.__setattr__(self, "density", density)


@dataclass(frozen=True)
class CxrSpectra:
    """PXR/CBS densities in photons / incident electron / eV / sr.

    incoherent
        Incoherent CXR density paired with its photon-energy grid in eV.
    coherent
        Optional coherent CXR density on the same photon-energy grid; ``None``
        when not calculated. Neither component includes atomic characteristic
        radiation. Both carry their source/observation normalization.
    """

    incoherent: SpectralDensity
    """Incoherent PXR/CBS density; always present."""
    coherent: SpectralDensity | None = None
    """Coherent PXR/CBS density on the same grid, or ``None`` if not requested."""

    def __post_init__(self) -> None:
        if not isinstance(self.incoherent, SpectralDensity):
            raise TypeError("incoherent must be a SpectralDensity")
        if self.coherent is not None:
            if not isinstance(self.coherent, SpectralDensity):
                raise TypeError("coherent must be a SpectralDensity or None")
            if (
                not np.array_equal(self.incoherent.photon_energy_eV, self.coherent.photon_energy_eV)
                or self.incoherent.normalization != self.coherent.normalization
            ):
                raise ValueError("CXR modes must share their photon-energy grid and normalization")


@dataclass(frozen=True, init=False)
class Result:
    """One simulated scene: spectra, the inputs behind them, and how they were made.

    Layout::

        result
        ├── cxr.incoherent      SpectralDensity  PXR/CBS, CXR grid
        ├── cxr.coherent        SpectralDensity | None
        ├── characteristic      SpectralDensity | None   atomic lines, CXR grid
        ├── brem                SpectralDensity  continuum, own wider grid
        ├── scene, numerics     what you asked for (shortcuts into provenance)
        ├── case                low-level resolved transport input (Case)
        ├── provenance          Provenance: identity, models, diagnostics, env
        └── spatial             SpatialResult | None  per-pixel factors

    Every :class:`SpectralDensity` holds photons / incident electron / eV / sr
    on its own grid; no family includes another. Derived views:
    :meth:`to_dataframe` (long, native grids), :meth:`spectral_flux` (wide,
    common grid, beam/acceptance-scaled, with a ``total`` column), and
    :meth:`combined_cxr` (CXR plus characteristic). Displaying a result in a
    notebook renders this layout with grid sizes and ranges.
    """

    cxr: CxrSpectra
    """PXR/CBS emission: ``incoherent`` always, ``coherent`` when requested."""
    brem: SpectralDensity
    """Bremsstrahlung continuum on its own (usually wider, coarser) grid."""
    characteristic: SpectralDensity | None
    """Atomic characteristic lines on the CXR grid, excluded from both CXR modes."""
    case: Case
    """Fully resolved low-level transport input; prefer :attr:`scene` for inputs."""
    provenance: Provenance
    """Inputs, identity digests, physics-model markers, diagnostics, environment."""
    spatial: SpatialResult | None
    """Factorized per-pixel data for pixelated planar detectors, else ``None``."""

    def __init__(
        self,
        *,
        cxr: CxrSpectra,
        brem: SpectralDensity,
        characteristic: SpectralDensity | None = None,
        case: Case,
        provenance: Mapping[str, Any],
        spatial: SpatialResult | None = None,
    ) -> None:
        if not isinstance(cxr, CxrSpectra) or not isinstance(brem, SpectralDensity):
            raise TypeError("Result requires cxr=CxrSpectra and brem=SpectralDensity")
        if characteristic is not None:
            if not isinstance(characteristic, SpectralDensity):
                raise TypeError("characteristic must be a SpectralDensity or None")
            if not np.array_equal(characteristic.photon_energy_eV, cxr.incoherent.photon_energy_eV):
                raise ValueError("characteristic must use the CXR photon-energy grid")
            if characteristic.normalization != cxr.incoherent.normalization:
                raise ValueError("characteristic and CXR normalization must match")
        if spatial is not None and not isinstance(spatial, SpatialResult):
            raise TypeError("spatial must be a SpatialResult or None")
        if case is None or provenance is None:
            raise TypeError("Result requires case and provenance")
        for name, value in (
            ("cxr", cxr),
            ("brem", brem),
            ("characteristic", characteristic),
            ("case", case),
            (
                "provenance",
                provenance if isinstance(provenance, Provenance) else Provenance(provenance),
            ),
            ("spatial", spatial),
        ):
            object.__setattr__(self, name, value)

    def combined_cxr(
        self, *, mode: str = "incoherent", include_characteristic: bool = True
    ) -> SpectralDensity:
        """Return the selected CXR mode plus optional atomic emission on its grid.

        The returned density is photons / incident electron / eV / sr, with
        the selected mode's source/observation normalization. Brem is excluded
        because it may use another photon-energy grid.
        """
        if mode not in {"incoherent", "coherent"}:
            raise ValueError("mode must be 'incoherent' or 'coherent'")
        component = getattr(self.cxr, mode)
        if component is None:
            raise ValueError("coherent spectrum is not available")
        density = component.density
        if include_characteristic and self.characteristic is not None:
            density = density + self.characteristic.density
        return SpectralDensity(component.photon_energy_eV, density, component.normalization)

    @property
    def scene(self) -> Scene | None:
        """Resolved beam, target, detector, filters, emission, and acquisition."""
        return self.provenance.scene

    @property
    def numerics(self) -> Numerics | None:
        """Resolved sampling and execution controls."""
        return self.provenance.numerics

    def _components(self, mode: str | None = None) -> list[tuple[str, str | None, SpectralDensity]]:
        """``(component, mode, density)`` families; ``mode=None`` keeps both CXR modes."""
        if mode not in {None, "incoherent", "coherent"}:
            raise ValueError("mode must be 'incoherent' or 'coherent'")
        parts: list[tuple[str, str | None, SpectralDensity]] = []
        if mode in {None, "incoherent"}:
            parts.append(("cxr", "incoherent", self.cxr.incoherent))
        if mode in {None, "coherent"}:
            if self.cxr.coherent is not None:
                parts.append(("cxr", "coherent", self.cxr.coherent))
            elif mode == "coherent":
                raise ValueError("coherent spectrum is not available")
        if self.characteristic is not None:
            parts.append(("characteristic", None, self.characteristic))
        parts.append(("brem", None, self.brem))
        return parts

    def to_dataframe(self) -> pl.DataFrame:
        """Return the separate spectral arrays as a tidy long-form polars DataFrame.

        Columns are ``component`` (``cxr``, ``characteristic``, ``brem``),
        ``mode`` (CXR ``incoherent``/``coherent``, null for other families),
        ``photon_energy_eV`` (eV), ``density`` (photons / incident electron /
        eV / sr), and ``normalization`` (``source`` or ``observation-average``).
        Each array keeps its own grid; nothing is interpolated or padded, and
        no totals are added. Use :meth:`spectral_flux` for a common-grid table
        with physical units and a ``total`` column. Case and provenance stay on
        this result; spatial pixel data stays on :attr:`spatial`.
        """
        import polars as pl

        parts = self._components()
        families = list(dict.fromkeys(name for name, _, _ in parts))
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "component": [name] * len(component.density),
                        "mode": [mode] * len(component.density),
                        "photon_energy_eV": np.asarray(
                            component.photon_energy_eV, dtype=np.float64
                        ),
                        "density": np.asarray(component.density, dtype=np.float64),
                        "normalization": [component.normalization] * len(component.density),
                    },
                    schema={
                        "component": pl.Enum(families),
                        "mode": pl.Enum(["incoherent", "coherent"]),
                        "photon_energy_eV": pl.Float64,
                        "density": pl.Float64,
                        "normalization": pl.Enum(["source", "observation-average"]),
                    },
                )
                for name, mode, component in parts
            ]
        )

    def accepted_solid_angle_sr(self) -> float:
        """Solid angle in sr that turns this result's per-sr density into accepted flux.

        A scalar detector uses the case acceptance ``domega_sr`` (the detector's
        ``solid_angle_sr``, or the default when unset). A planar detector uses
        the summed pixel solid angle, so the observation-average density times
        this value is the flux summed over the detector.
        """
        if self.cxr.incoherent.normalization == "source":
            return float(self.case["domega_sr"])
        if self.spatial is not None:
            return float(np.sum(self.spatial.ray_map.solid_angle_sr))
        from ..instrument.geometry import planar_detector_rays

        detector = getattr(self.scene, "detector", None)
        if not isinstance(detector, PlanarDetector):
            raise ValueError("observation-average result needs its planar detector in scene")
        return float(np.sum(planar_detector_rays(detector).solid_angle_sr))

    def spectral_flux(
        self,
        *,
        units: FluxUnits = "detector",
        mode: str = "incoherent",
    ) -> pl.DataFrame:
        """Return a wide spectrum table on a common grid, in physical units.

        Columns are ``photon_energy_eV``, ``photon_energy_keV``, one column per
        family (``cxr`` in the selected ``mode``, ``characteristic`` when
        present, ``brem``), and ``total``, their sum. The grid is the union of
        every family's native nodes, so each family is exact at its own nodes
        and linearly interpolated between them. Outside its own grid a family
        is zero; the total therefore omits brem below the brem grid start.

        Parameters
        ----------
        units
            Key of :data:`FLUX_UNITS`:

            - ``"detector"``: photons / s / eV into the accepted solid angle
              (:meth:`accepted_solid_angle_sr`) at the mean beam current.
            - ``"nA"``: photons / s / nA / eV into the accepted solid angle.
            - ``"sr"``: photons / s / eV / sr at the mean beam current.
            - ``"electron"``: the stored photons / electron / eV / sr.

            Mean current is ``Q[pC] * f[kHz]`` nA from the beam's bunch charge
            and repetition rate. Scalar-detector densities are evaluated at the
            central observation direction, so ``"detector"`` and ``"nA"`` assume
            the density is uniform over the acceptance.
        mode
            CXR mode for the ``cxr`` column: ``"incoherent"`` or ``"coherent"``.

        Validation: pixel-acquisition-counting
        """
        import polars as pl
        from scipy.constants import elementary_charge

        from .store import beam_current_na

        if units not in FLUX_UNITS:
            raise ValueError(f"units must be one of {', '.join(map(repr, FLUX_UNITS))}")
        electrons_per_s_per_na = 1e-9 / elementary_charge
        if units == "electron":
            scale = 1.0
        elif units == "sr":
            scale = beam_current_na(self.case) * electrons_per_s_per_na
        elif units == "nA":
            scale = self.accepted_solid_angle_sr() * electrons_per_s_per_na
        else:
            scale = (
                beam_current_na(self.case) * electrons_per_s_per_na * self.accepted_solid_angle_sr()
            )

        parts = self._components(mode)
        energy_eV = np.unique(
            np.concatenate([np.asarray(part.photon_energy_eV, float) for _, _, part in parts])
        )
        columns = {}
        for name, _, part in parts:
            order = np.argsort(part.photon_energy_eV)
            columns[name] = scale * np.interp(
                energy_eV,
                np.asarray(part.photon_energy_eV, float)[order],
                np.asarray(part.density, float)[order],
                left=0.0,
                right=0.0,
            )
        return pl.DataFrame(
            {"photon_energy_eV": energy_eV, "photon_energy_keV": energy_eV / 1e3, **columns}
        ).with_columns(total=pl.sum_horizontal(list(columns)))

    def _summary_rows(self) -> list[tuple[str, str, str]]:
        def grid(part: SpectralDensity | None) -> str:
            if part is None:
                return "—"
            energy = part.photon_energy_eV
            return f"{energy.size} pts, {energy.min():g}–{energy.max():g} eV"

        scene = self.provenance.scene
        spatial = "—" if self.spatial is None else f"{self.spatial.shape} pixels"
        return [
            ("cxr.incoherent", grid(self.cxr.incoherent), "incoherent PXR/CBS"),
            ("cxr.coherent", grid(self.cxr.coherent), "coherent PXR/CBS"),
            ("characteristic", grid(self.characteristic), "atomic lines, CXR grid"),
            ("brem", grid(self.brem), "bremsstrahlung continuum"),
            (
                "scene",
                "—"
                if scene is None
                else f"{type(scene.target).__name__}, "
                f"{type(scene.detector).__name__}, {scene.emission}",
                "resolved inputs",
            ),
            ("case", f"Case, {len(self.case)} keys", "low-level transport input"),
            ("provenance", f"{len(self.provenance)} keys", "identity, models, diagnostics, env"),
            ("spatial", spatial, "per-pixel factors"),
        ]

    def __repr__(self) -> str:
        digest = self.provenance.identity_digest
        head = (
            f"Result({self.cxr.incoherent.normalization} normalization"
            f"{'' if digest is None else f', identity {digest[:12]}…'})"
        )
        width = max(len(name) for name, _, _ in self._summary_rows())
        return "\n".join(
            [head] + [f"  {name:<{width}}  {value}" for name, value, _ in self._summary_rows()]
        )

    def _repr_markdown_(self) -> str:
        """Notebook display: the result layout with grid sizes and ranges."""
        units = "photons / (electron eV sr)"
        rows = "\n".join(
            f"| `{name}` | {value} | {description} |"
            for name, value, description in self._summary_rows()
        )
        return (
            f"**Result** — {self.cxr.incoherent.normalization} normalization, "
            f"densities in {units}\n\n| attribute | contents | meaning |\n|---|---|---|\n" + rows
        )

    def _acquisition_context(self):
        scene = self.provenance.get("scene")
        acquisition = getattr(scene, "acquisition", None)
        beam = getattr(scene, "beam", None)
        if acquisition is None or beam is None:
            raise ValueError("result was not produced with an acquisition configuration")
        if self.spatial is None:
            raise ValueError("result does not contain spatial factors")
        digest = self.provenance.get("observation_identity_digest")
        if not isinstance(digest, str):
            raise ValueError("result does not contain an observation identity digest")
        return acquisition, beam, digest, scene

    def acquire(
        self,
        *,
        pixels=None,
        region=None,
        components: tuple[str, ...] | None = None,
    ) -> AcquisitionBatch:
        """Score selected pixels using this result's frozen acquisition identity."""
        acquisition, beam, digest, scene = self._acquisition_context()
        assert self.spatial is not None
        selected = (
            (("coherent", "brem") if scene.emission == "coherent" else ("cxr", "brem"))
            if components is None
            else components
        )
        return self.spatial.acquire(
            acquisition,
            rep_rate_hz=beam.rep_rate_hz,
            bunch_charge_pc=beam.bunch_charge_pc,
            observation_digest=digest,
            pixels=pixels,
            region=region,
            components=selected,
        )

    def acquisition_image(
        self,
        energy_range_eV: tuple[float, float] | None = None,
        *,
        components: tuple[str, ...] | None = None,
        pixel_chunk: int = 1024,
    ) -> np.ndarray:
        """Return the configured total or reporting-window detector count image."""
        acquisition, beam, digest, scene = self._acquisition_context()
        assert self.spatial is not None
        selected = (
            (("coherent", "brem") if scene.emission == "coherent" else ("cxr", "brem"))
            if components is None
            else components
        )
        return self.spatial.acquisition_image(
            acquisition=acquisition,
            rep_rate_hz=beam.rep_rate_hz,
            bunch_charge_pc=beam.bunch_charge_pc,
            observation_digest=digest,
            energy_range_eV=energy_range_eV,
            components=selected,
            pixel_chunk=pixel_chunk,
        )


__all__ = [
    "COMPONENTS",
    "CxrSpectra",
    "FLUX_UNITS",
    "FluxUnits",
    "PixelMetadata",
    "PixelRayMap",
    "Result",
    "SpatialResult",
    "SpectralDensity",
    "SpectralFactors",
]
