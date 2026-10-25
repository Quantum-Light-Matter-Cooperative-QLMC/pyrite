"""Linear electron bunch-front geometry, independent of source sampling.

Validation: linear-bunch-front-tilt
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .._units import normalize_unit
from ..materials.crystal import beta_from_Ee


@dataclass(frozen=True)
class BunchFrontTilt:
    """Signed electron-front tilt from the beam-normal plane, in degrees.

    Positive ``tilt_deg`` means later arrival along the lab transverse axis
    selected by ``azimuth_deg`` (0 is +x, 90 is +y). The mean direction is
    unchanged. At the beam-normal plane through the target origin,
    ``c*t0 = c*epsilon + tan(theta) * (x*cos(phi)+y*sin(phi))/beta0``.
    ``BunchLength.rms_fs`` describes the independent residual epsilon at
    fixed position, not the duration of the whole tilted envelope.
    Matching this tilt and azimuth to the target's tilt and azimuth makes
    the front parallel to the entrance face: its imposed timing gradient
    cancels the vacuum flight-delay gradient at entry.

    Source: constant-velocity ray kinematics of a tilted equal-time plane;
    electron bunch tilt terminology: Tadenuma et al., PRAB 25, 110102 (2022).
    Assumes monoenergetic parallel rays. Zero tilt adds no delay; zero spot
    has no position-dependent delay. Units: c*t0 and x,y in angstrom.
    Validation: linear-bunch-front-tilt
    """

    tilt_deg: float
    azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tilt_deg",
            normalize_unit(
                self.tilt_deg,
                "degree",
                field="BunchFrontTilt.tilt_deg",
                expected="angle",
                kind="angle",
            ),
        )
        object.__setattr__(
            self,
            "azimuth_deg",
            normalize_unit(
                self.azimuth_deg,
                "degree",
                field="BunchFrontTilt.azimuth_deg",
                expected="angle",
                kind="angle",
            ),
        )
        if not np.isfinite(self.tilt_deg) or not -90.0 < self.tilt_deg < 90.0:
            raise ValueError("bunch-front tilt_deg must be finite and strictly between -90 and 90")
        if not np.isfinite(self.azimuth_deg):
            raise ValueError("bunch-front azimuth_deg must be finite")

    def gradient(self, energy_keV: float) -> np.ndarray:
        """Return d(c*t0)/d(x,y), dimensionless, at the nominal kinetic energy.

        Source/assumptions/limits: the plane equation in this class's
        docstring, with beta0=v0/c. Positive tilt gives positive delay toward
        the chosen axis; zero tilt gives exact zeros.
        Validation: linear-bunch-front-tilt
        """
        if not np.isfinite(energy_keV) or energy_keV <= 0:
            raise ValueError("bunch-front tilt requires finite positive beam energy")
        beta = float(beta_from_Ee(float(energy_keV) * 1e3))
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("bunch-front tilt requires finite positive beam energy")
        phi = np.deg2rad(self.azimuth_deg)
        return np.tan(np.deg2rad(self.tilt_deg)) / beta * np.array([np.cos(phi), np.sin(phi)])


def resolve_front_tilt(value: BunchFrontTilt | Mapping | None) -> BunchFrontTilt | None:
    """Decode a serialized front descriptor; canonicalize zero tilt to None."""
    if value is None:
        return None
    front = BunchFrontTilt(**value) if isinstance(value, Mapping) else value
    if not isinstance(front, BunchFrontTilt):
        raise TypeError("bunch_front_tilt must be a BunchFrontTilt or its serialized mapping")
    return None if front.tilt_deg == 0 else front


def validate_front_source(
    front,
    *,
    transverse=None,
    energy_spread=None,
    source="analytic",
    longitudinal=None,
    offsets=None,
    shape="gaussian",
    groove=None,
):
    """Refuse sources outside the monoenergetic Gaussian parallel-ray model."""
    front = resolve_front_tilt(front)
    if front is not None and (
        transverse is not None
        or energy_spread is not None
        or source != "analytic"
        or longitudinal is not None
        or offsets is not None
        or shape != "gaussian"
        or groove is not None
    ):
        raise ValueError(
            "nonzero bunch-front tilt supports only monoenergetic analytic point or "
            "Gaussian spots with PointBunch or Gaussian BunchLength on a flat face"
        )
    return front
