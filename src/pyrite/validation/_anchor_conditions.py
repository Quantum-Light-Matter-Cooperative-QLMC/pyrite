"""Zhai literature conditions, case construction, and analytic anchors.

This internal module owns deterministic validation inputs and analytic
cross-checks. Public compatibility imports remain in
:mod:`pyrite.validation.anchor_figures`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.detectors import Detector, EnergyBins
from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    absorption_length_ang,
    reciprocal_g_vector,
)
from pyrite.montecarlo import Case, beta_from_keV, mc_spectrum
from pyrite.validation._zhai import (
    ZHAI_CACHE_FORMAT,
    ZHAI_CACHE_SCHEMA,
    ZHAI_DETECTOR,
    detector_metadata,
)

from .feranchuk_spence import photons_per_electron

GRAPHITE_B_002 = 0.8  # graphite c-axis Debye-Waller B-factor [Ang^2], approx (Zhai SI)


@dataclass(frozen=True)
class SupplementaryCondition:
    """One spectrum condition reported in Zhai et al. Supplementary Fig. 5."""

    energy_keV: float
    polar_tilt_deg: float
    azimuth_deg: float | None


@dataclass(frozen=True)
class SupplementaryCoherentStudy:
    """A Zhai supplementary-information coherent-emission comparison.

    The study holds the reported thicknesses, photon-energy window, and four
    per-spectrum conditions (beam energy, polar tilt, and azimuth — ``None``
    when Zhai does not report it).  Its cached spectra contain response-free source
    coherent emission only; the figure builders apply the Fig. 1c detector
    response when converting them to the displayed observable.
    """

    crystal: str
    label: str
    e_min_eV: float
    e_max_eV: float
    conditions_by_thickness_nm: tuple[tuple[float, tuple[SupplementaryCondition, ...]], ...]
    detector: Detector = ZHAI_DETECTOR

    def __post_init__(self) -> None:
        if self.detector.polar_acceptance_deg is None or self.detector.solid_angle_sr is None:
            raise ValueError("Zhai detector requires polar_acceptance_deg and solid_angle_sr")

    @property
    def theta_obs_rad(self) -> float:
        """Resolved observation angle in the historical case unit."""
        return float(np.deg2rad(self.detector.observation_angle_deg))

    @property
    def dtheta_obs_rad(self) -> float:
        """Resolved full polar acceptance in the historical case unit."""
        assert self.detector.polar_acceptance_deg is not None
        return float(np.deg2rad(self.detector.polar_acceptance_deg))

    @property
    def domega_sr(self) -> float:
        """Resolved collection solid angle."""
        assert self.detector.solid_angle_sr is not None
        return self.detector.solid_angle_sr

    @property
    def E_grid(self) -> np.ndarray:
        """One-eV photon-energy grid, excluding the upper endpoint."""
        return np.arange(self.e_min_eV, self.e_max_eV, 1.0)

    @property
    def thicknesses_nm(self) -> tuple[float, ...]:
        """Reported sample thicknesses in their Table 4 order."""
        return tuple(thickness_nm for thickness_nm, _conditions in self.conditions_by_thickness_nm)

    @property
    def conditions(self) -> tuple[SupplementaryCondition, ...]:
        """Compatibility view of the first sample's condition set.

        New callers must use :meth:`conditions_for`, because Table 4 assigns
        different orientations to different sample thicknesses.
        """
        return self.conditions_for(self.thicknesses_nm[0])

    def conditions_for(self, thickness_nm: float) -> tuple[SupplementaryCondition, ...]:
        """Return the published conditions for one requested sample thickness."""
        for reported_thickness_nm, conditions in self.conditions_by_thickness_nm:
            if thickness_nm == reported_thickness_nm:
                return conditions
        raise ValueError(
            f"{thickness_nm:g} nm is not one of {self.crystal}'s requested "
            f"thicknesses {self.thicknesses_nm}"
        )

    @property
    def has_unreported_azimuth(self) -> bool:
        """Whether any reported condition needs an exploratory azimuth."""
        return any(
            condition.azimuth_deg is None
            for _thickness_nm, conditions in self.conditions_by_thickness_nm
            for condition in conditions
        )


@dataclass(frozen=True)
class ZhaiThermalBeam:
    """Zhai SI S11 thermal-analysis Gaussian electron-beam input.

    Zhai et al. model the XY current density as a 2-D Gaussian (SI Eq. 23) and
    specify a 1 mm diameter containing 99.9% of the electrons for their 300 keV
    thermal analysis. For the simulation's Gaussian FWHM convention, the
    radial containment relation is ``p = 1 - exp(-r**2 / (2 sigma**2))`` with
    ``r = diameter / 2``. This gives
    ``FWHM = diameter * sqrt(log(2) / log(1 / (1 - p)))``.

    Validation: finite-beam-size
    """

    diameter_mm: float = 1.0
    enclosed_fraction: float = 0.999
    energy_keV: float = 300.0

    @property
    def fwhm_mm(self) -> float:
        """Equivalent isotropic-Gaussian beam FWHM in millimetres."""
        return float(
            self.diameter_mm * np.sqrt(np.log(2.0) / np.log(1.0 / (1.0 - self.enclosed_fraction)))
        )


_SEM_ENERGIES_KEV = (17.5, 20.0, 22.5, 25.0)


def _sem_conditions(
    polar_tilt_deg: float, azimuth_deg: float
) -> tuple[SupplementaryCondition, ...]:
    """Four reported SEM beam-energy conditions at one sample orientation."""
    return tuple(
        SupplementaryCondition(energy_keV, polar_tilt_deg, azimuth_deg)
        for energy_keV in _SEM_ENERGIES_KEV
    )


def _same_conditions_by_thickness(
    thicknesses_nm: tuple[float, ...], conditions: tuple[SupplementaryCondition, ...]
) -> tuple[tuple[float, tuple[SupplementaryCondition, ...]], ...]:
    """Associate a shared reported condition set with each listed thickness."""
    return tuple((thickness_nm, conditions) for thickness_nm in thicknesses_nm)


def _thickness_stem(thickness_nm: float) -> str:
    """Stable decimal thickness label for cache/figure identifiers."""
    return str(int(thickness_nm)) if thickness_nm.is_integer() else f"{thickness_nm:g}"


ZHAI_SUPPLEMENTARY_STUDIES = {
    "wse2": SupplementaryCoherentStudy(
        crystal="wse2",
        label=r"$\mathrm{WSe_2}$",
        e_min_eV=800.0,
        e_max_eV=1200.0,
        conditions_by_thickness_nm=_same_conditions_by_thickness(
            (42.0, 55.0, 75.0),
            tuple(
                SupplementaryCondition(200.0, polar_tilt_deg, None)
                for polar_tilt_deg in (10.0, 15.0, 17.5, 20.0)
            ),
        ),
    ),
    "mose2": SupplementaryCoherentStudy(
        crystal="mose2",
        label=r"$\mathrm{MoSe_2}$",
        e_min_eV=800.0,
        e_max_eV=1200.0,
        conditions_by_thickness_nm=_same_conditions_by_thickness(
            (47.0, 112.0, 147.0),
            tuple(
                SupplementaryCondition(200.0, polar_tilt_deg, None)
                for polar_tilt_deg in (10.0, 15.0, 17.5, 20.0)
            ),
        ),
    ),
    "hbn": SupplementaryCoherentStudy(
        crystal="hbn",
        label="h-BN",
        e_min_eV=600.0,
        e_max_eV=1200.0,
        conditions_by_thickness_nm=(
            (42.0, _sem_conditions(13.5, 115.0)),
            (109.0, _sem_conditions(13.5, 130.0)),
            (219.0, _sem_conditions(11.5, 65.0) + _sem_conditions(17.0, 130.0)),
            (659.0, _sem_conditions(14.5, 105.0)),
            (921.0, _sem_conditions(17.0, 130.0)),
            (170_000.0, _sem_conditions(20.0, 65.0)),
        ),
    ),
    "hopg": SupplementaryCoherentStudy(
        crystal="hopg",
        label="HOPG",
        e_min_eV=500.0,
        e_max_eV=1250.0,
        conditions_by_thickness_nm=(
            (29.0, _sem_conditions(9.5, 120.0)),
            (76.0, _sem_conditions(13.0, 120.0)),
            (150.0, _sem_conditions(6.5, 180.0)),
            (17_000.0, _sem_conditions(11.0, 60.0)),
            (500_000.0, _sem_conditions(11.5, 40.0)),
            (1_000_000.0, _sem_conditions(9.5, 40.0)),
        ),
    ),
}


@dataclass(frozen=True)
class ZhaiAnchor:
    """Experimental conditions of Zhai et al. Fig. 1c (SEM/EDS take-off geometry).

    All defaults reproduce the published setup; override fields to explore other
    crystals, energies or detector geometries with the same machinery.
    """

    crystal: str = "hopg"
    hkl: tuple[int, int, int] = (0, 0, 2)
    hkl_list: tuple[tuple[int, int, int], ...] = ((0, 0, 2), (0, 0, -2))
    energies_keV: tuple[float, ...] = (17.5, 20.0, 22.5, 25.0)
    detector: Detector = ZHAI_DETECTOR
    per_nA: float = 6.2415e9  # electrons/s at 1 nA
    thick_bulk_ang: float = 1e7  # 1 mm
    thick_film_ang: float = 290.0  # 29 nm
    B_ang2: float = GRAPHITE_B_002
    e_min_eV: float = 500.0
    e_max_eV: float = 1250.0
    de_eV: float = 1.0

    def __post_init__(self) -> None:
        if self.detector.polar_acceptance_deg is None or self.detector.solid_angle_sr is None:
            raise ValueError("Zhai detector requires polar_acceptance_deg and solid_angle_sr")

    @property
    def E_grid(self) -> np.ndarray:
        return np.arange(self.e_min_eV, self.e_max_eV, self.de_eV)

    @property
    def theta_obs_rad(self) -> float:
        """Resolved observation angle in the historical case unit."""
        return float(np.deg2rad(self.detector.observation_angle_deg))

    @property
    def dtheta_obs_rad(self) -> float:
        """Resolved full polar acceptance in the historical case unit."""
        assert self.detector.polar_acceptance_deg is not None
        return float(np.deg2rad(self.detector.polar_acceptance_deg))

    @property
    def domega_sr(self) -> float:
        """Resolved collection solid angle."""
        assert self.detector.solid_angle_sr is not None
        return self.detector.solid_angle_sr

    @property
    def n_atoms_per_ang3(self) -> float:
        info = CRYSTALS[self.crystal]
        return len(info["basis"]) / info["V_cell"]


def _resolved_case(
    *,
    material: str,
    detector: Detector,
    energy_keV: float,
    thickness_ang: float,
    E_grid: np.ndarray,
    polar_tilt_deg: float,
    azimuth_deg: float,
    ne: int,
    ne_brem: int,
    seed: int,
    hkl_list: tuple[tuple[int, ...], ...] | None = None,
    B_ang2: float | None = None,
) -> Case:
    """Build one current Sweep case for a literature condition."""
    sweep = Sweep(
        material=material,
        thickness_ang=thickness_ang,
        beam=BeamSpec.isotropic(None, energy_keV=energy_keV),
        tilt_deg=polar_tilt_deg,
        tilt_azim_deg=azimuth_deg,
        crystal_width_mm=None,
        crystal_height_mm=None,
        allow_normal_incidence=polar_tilt_deg == 0.0,
        detector=replace(detector, energy_bins=EnergyBins(line=E_grid, brem=E_grid)),
    )
    case = build_cases(sweep, n_electrons=ne, n_electrons_brem=ne_brem)[0]
    # Literature anchors may intentionally pin a narrower reflection set or
    # published Debye-Waller value than the general catalog sweep. The
    # validation compares one reported detector window; build_cases expands
    # uniform production brem grids to E0, so retain this explicit window.
    return replace(
        case,
        hkl_list=case.hkl_list if hkl_list is None else list(hkl_list),
        B_ang2=case.B_ang2 if B_ang2 is None else B_ang2,
        E_grid_brem=E_grid.copy(),
        seed=seed,
    )


def fig1c_case(
    anchor: ZhaiAnchor,
    E0_keV: float,
    thickness_ang: float,
    *,
    ne: int,
    ne_brem: int,
    seed: int,
) -> Case:
    """Build one current case carrying the published Fig. 1c inputs."""
    return _resolved_case(
        material=anchor.crystal,
        detector=anchor.detector,
        energy_keV=E0_keV,
        thickness_ang=thickness_ang,
        E_grid=anchor.E_grid,
        polar_tilt_deg=0.0,
        azimuth_deg=0.0,
        ne=ne,
        ne_brem=ne_brem,
        seed=seed,
        hkl_list=anchor.hkl_list,
        B_ang2=anchor.B_ang2,
    )


# ---- theory anchors (cheap, analytic) ----------------------------------------


def line_energy_eV(anchor: ZhaiAnchor, E0_keV: float) -> float:
    """Zero-scattering resonance energy in the production sign convention.

    Starting from Feranchuk-Spence Eq. (10), the executable anchor uses

        E = hbar c beta g_z / (1 - beta cos theta_obs),

    for beam parallel to the HOPG c-axis and ``g_z = |g|``. Under an
    ``exp(+i g.r)`` reconstruction, this positive numerator would correspond
    to the opposite reciprocal harmonic. Production does not document that
    mapping, so the sign discrepancy remains. In the nonrelativistic limit
    ``beta -> 0``, the line energy tends to zero.

    Validation: line-energy-dispersion
    """
    info = CRYSTALS[anchor.crystal]
    beta = beta_from_keV(E0_keV)
    _, g = reciprocal_g_vector(anchor.hkl, info["lattice"])
    return HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(anchor.theta_obs_rad))


def theory_line_energies(anchor: ZhaiAnchor) -> dict[float, float]:
    """{beam energy [keV]: dispersion-relation line energy [eV]}."""
    return {E0: line_energy_eV(anchor, E0) for E0 in anchor.energies_keV}


def feranchuk_line_flux(anchor: ZhaiAnchor, E0_keV: float, thickness_ang: float) -> float:
    """Feranchuk-Spence Eq. (12) closed-form line flux [photons / electron into
    dOmega], absorption-limited, at the dispersion-relation line energy."""
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
    return photons_per_electron(
        anchor.crystal,
        anchor.hkl,
        E_line,
        anchor.theta_obs_rad,  # geometry="lif": this slot carries theta_obs
        beta,
        L_z_ang=thickness_ang,
        L_abs_ang=L_abs,
        dOmega_sr=anchor.domega_sr,
        polarization="both",
        B_ang2=anchor.B_ang2,
        use_henke=True,
        geometry="lif",
    )


def single_segment_anchor(
    anchor: ZhaiAnchor, E0_keV: float, L_seg_ang: float = 290.0
) -> tuple[float, float, float]:
    """The cleanest analytic<->MC anchor. A single straight segment of length
    L_seg pushed through mc_spectrum, integrated over energy, equals the
    Eq. (12) closed form -- this isolates the finite-segment lineshape
    normalization from electron transport. Returns (mc_integral, closed_form,
    ratio); ratio should be ~1.
    """
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    fake = {
        "r_mid": np.array([[0.0, 0.0, 0.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([L_seg_ang]),
        "E_keV": np.array([E0_keV]),
        "Ne": 1,
        "elec_id": np.array(
            [
                0,
            ]
        ),
        "thickness_ang": anchor.thick_bulk_ang,
    }
    spec = mc_spectrum(
        fake,
        anchor.E_grid,
        crystal=anchor.crystal,
        hkl_list=(anchor.hkl,),
        theta_obs_rad=anchor.theta_obs_rad,
        B_ang2=anchor.B_ang2,
    )
    mc_int = float(np.trapezoid(spec, anchor.E_grid))  # photons / electron / sr
    # Closed form per electron per sr: dOmega=1, L_abs huge so L_eff -> L_seg.
    closed = float(
        photons_per_electron(
            anchor.crystal,
            anchor.hkl,
            E_line,
            anchor.theta_obs_rad,
            beta,
            L_z_ang=L_seg_ang,
            L_abs_ang=1e12,
            dOmega_sr=1.0,
            polarization="both",
            B_ang2=anchor.B_ang2,
            use_henke=True,
            geometry="lif",
        )
    )
    return mc_int, closed, mc_int / closed


def _cache_record(kind: str, detector: Detector, payload: object) -> dict:
    """Wrap generated data with cache schema and resolved detector provenance."""
    return {
        "format": ZHAI_CACHE_FORMAT,
        "schema": ZHAI_CACHE_SCHEMA,
        "kind": kind,
        "detector": detector_metadata(detector),
        "payload": payload,
    }


def _cache_payload(record: object, *, kind: str, detector: Detector) -> object | None:
    """Return a valid v4 payload; reject pre-detector or mismatched records."""
    if not isinstance(record, dict):
        return None
    expected = {
        "format": ZHAI_CACHE_FORMAT,
        "schema": ZHAI_CACHE_SCHEMA,
        "kind": kind,
        "detector": detector_metadata(detector),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    return record.get("payload")


def _encode_supplementary_spectra(
    spectra: dict[SupplementaryCondition, np.ndarray],
) -> dict[tuple[float, float, float | None], np.ndarray]:
    """Lower dataclass keys to schema-native scalar tuples for persistence."""
    return {
        (condition.energy_keV, condition.polar_tilt_deg, condition.azimuth_deg): spectrum
        for condition, spectrum in spectra.items()
    }


def _decode_supplementary_spectra(
    payload: object,
) -> dict[SupplementaryCondition, np.ndarray]:
    """Restore current tuple keys or legacy pickled dataclass keys."""
    if not isinstance(payload, dict):
        raise ValueError("invalid supplementary cache payload")
    spectra: dict[SupplementaryCondition, np.ndarray] = {}
    for key, spectrum in payload.items():
        if isinstance(key, SupplementaryCondition):
            condition = key
        elif isinstance(key, tuple) and len(key) == 3:
            condition = SupplementaryCondition(key[0], key[1], key[2])
        else:
            raise ValueError("invalid supplementary cache condition key")
        spectra[condition] = np.asarray(spectrum)
    return spectra


# Preserve the historical pickle/import identity exposed by anchor_figures.
for _compat_type in (
    SupplementaryCondition,
    SupplementaryCoherentStudy,
    ZhaiThermalBeam,
    ZhaiAnchor,
):
    _compat_type.__module__ = "pyrite.validation.anchor_figures"
