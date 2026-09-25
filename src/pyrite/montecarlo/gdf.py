"""CPU-only GPT time-output and screen adapter; EasyGDF dictionaries stay at this boundary."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import easygdf
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, electron_mass, elementary_charge

from ..materials._schema import MaterialConfigError
from .geometry import sample_to_lab_R

GDF_FIELDS = (
    "source",
    "gdf_path",
    "gdf_shape_only",
    "gdf_time_s",
    "gdf_time_tolerance_s",
    "gdf_screen_position_m",
    "gdf_screen_tolerance_m",
    "gdf_normalization",
    "gdf_repetition_rate_hz",
    "gdf_z_origin_m",
)
_REQUIRED = ("x", "y", "z", "Bx", "By", "Bz", "m", "q")


def _error(message: str) -> MaterialConfigError:
    return MaterialConfigError((f"GPT GDF: {message}",))


def _read(path: str | Path) -> tuple[list[dict], str]:
    try:
        with Path(path).open("rb") as stream:
            digest = sha256(stream.read()).hexdigest()
            stream.seek(0)
            data = easygdf.load(stream)
            stream.seek(0)
            if sha256(stream.read()).hexdigest() != digest:
                raise _error("file changed while reading; retry with a stable input file")
        if data.get("creator", "").strip().upper() != "GPT":
            raise _error(f"{path}: expected a GPT-produced GDF file")
        blocks = [b for b in data["blocks"] if b["name"] in {"time", "position"}]
        if not blocks:
            raise _error("no time-output or screen blocks")
        for block in blocks:
            coordinate = block["value"]
            if (
                not isinstance(coordinate, (int, float))
                or not np.isfinite(coordinate)
                or (block["name"] == "time" and coordinate < 0)
            ):
                raise _error("output block has an invalid time or screen coordinate")
        return blocks, digest
    except MaterialConfigError:
        raise
    except Exception as exc:
        raise _error(f"cannot read {path}: {exc}; supply a readable GPT time-output GDF") from None


def _inventory(blocks: list[dict]) -> tuple[tuple[float, int], ...]:
    return tuple(
        (
            float(b["value"]),
            next((int(np.size(a["value"])) for a in b["children"] if a["name"] == "x"), 0),
        )
        for b in blocks
    )


def list_gdf_times(path: str | Path) -> tuple[tuple[float, int], ...]:
    """Return GPT time-output times [s] and x-array particle counts."""
    blocks = [b for b in _read(path)[0] if b["name"] == "time"]
    if not blocks:
        raise _error("no time-output blocks; use gdf-inspect to list screens")
    return _inventory(blocks)


@dataclass(frozen=True)
class GDFBeam:
    """Validated complete phase-space records in SI units (energy in keV)."""

    time_s: float | None
    position_m: NDArray[np.float64]
    beta: NDArray[np.float64]
    energy_keV: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    signed_charge_c: float | None
    absolute_charge_c: float | None
    sha256: str
    screen_position_m: float | None = None
    arrival_time_s: NDArray[np.float64] | None = None

    def sample(
        self,
        count: int,
        seed: int,
        z_origin_m: float,
        tilt: float = 0.0,
        azimuth: float = 0.0,
        *,
        energy_keV: float | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Resample whole records and intersect their rays with the entrance.

        Source: ray-plane intersection s=-r_z/d_z; relativistic v=c beta.
        Assumptions: field-free extrapolation, GPT axes are PyRITE lab axes;
        z_origin_m is the explicit lab z coordinate of the target origin.
        Signed extrapolation permits a snapshot on either side of the plane.
        Screens add c*(t_i-min(t)) to the signed flight clock, using the
        full input crossing-time reference and constant velocity during drift.
        Untilted records already at z_origin_m retain their x/y and direction.
        Probability sampling estimates sum(nmacro*f)/sum(nmacro), with no
        second history weight. Stream child 6 is disjoint from analytic draws.
        Optional energy_keV assigns a monoenergetic shape-only beam and drops
        imported crossing times. With k=K/(m_e*c**2), beta=sqrt(k/(1+k)*(1+1/(1+k))).
        This preserves direction, not momentum, and does not model acceleration.
        Validation: gpt-gdf-injection
        """
        if not np.isfinite(z_origin_m):
            raise _error("gdf_z_origin_m must be finite")
        if energy_keV is not None and (not np.isfinite(energy_keV) or energy_keV <= 0):
            raise _error("shape-only energy_keV must be finite and positive")
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(7)[6])
        rotation = sample_to_lab_R(tilt, azimuth)
        all_beta = self.beta @ rotation
        if np.any(all_beta[:, 2] / np.linalg.norm(all_beta, axis=1) <= 1e-6):
            raise _error("selected rays must point into the slab (sample-frame direction z > 1e-6)")
        indices = rng.choice(len(self.energy_keV), size=count, p=self.probabilities)
        position = self.position_m[indices].copy()
        position[:, 2] -= z_origin_m
        position = position @ rotation
        beta = all_beta[indices]
        speed = np.linalg.norm(beta, axis=1)
        direction = beta / speed[:, None]
        distance = -position[:, 2] / direction[:, 2]
        position += distance[:, None] * direction
        position[:, 2] = 0.0
        position_ang = position * 1e10
        energies = self.energy_keV[indices].copy()
        if energy_keV is not None:
            energies.fill(energy_keV)
            k = energy_keV * (1000 * elementary_charge / (electron_mass * c**2))
            speed = np.full(count, np.sqrt((k / (1 + k)) * (1 + 1 / (1 + k))))
        time_ang = distance / speed * 1e10
        if self.arrival_time_s is not None and energy_keV is None:
            # Preserve screen crossing-time correlations relative to a common
            # reference; no absolute simulation clock enters the kernels.
            time_ang += (self.arrival_time_s[indices] - self.arrival_time_s.min()) * c * 1e10
        if not np.all(np.isfinite(position_ang)) or not np.all(np.isfinite(time_ang)):
            raise _error("entrance projection overflow; check positions and gdf_z_origin_m")
        return position_ang, direction, energies, time_ang


def load_gdf_beam(
    path: str | Path,
    time_s: float | None = None,
    tolerance_s: float = 1e-15,
    normalization: str = "pyrite_current",
    *,
    screen_position_m: float | None = None,
    screen_tolerance_m: float = 1e-9,
) -> GDFBeam:
    """Load one native GPT time snapshot or screen without synthesized fields.

    Source: relativistic gamma=(1-beta²)^(-1/2), K=(gamma-1)mc²;
    GPT beta fields and nmacro semantics: EasyGDF/GPT output documentation.
    Assumptions: electrons, finite SI data, 0<beta²<1. Electron mass and charge use
    rtol=1e-6; optional G agrees with gamma to rtol=1e-6, atol=1e-9.
    K approaches mv²/2 at low speed; Q=sum(q*nmacro) is negative for electrons.
    Validation: gpt-gdf-injection
    """
    for name, value in (("gdf_time_s", time_s), ("gdf_time_tolerance_s", tolerance_s)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise _error(f"{name} must be finite and non-negative")
    if normalization not in {"pyrite_current", "gdf_charge"}:
        raise _error("gdf_normalization must be pyrite_current or gdf_charge")
    if time_s is not None and screen_position_m is not None:
        raise _error("gdf_time_s and gdf_screen_position_m are mutually exclusive")
    if screen_position_m is not None and not np.isfinite(screen_position_m):
        raise _error("gdf_screen_position_m must be finite")
    if not np.isfinite(screen_tolerance_m) or screen_tolerance_m < 0:
        raise _error("gdf_screen_tolerance_m must be finite and non-negative")
    all_blocks, digest = _read(path)
    kind = "time" if screen_position_m is None else "position"
    blocks = [b for b in all_blocks if b["name"] == kind]
    if not blocks:
        message = (
            "no time-output blocks; screen-only file: select gdf_screen_position_m"
            if kind == "time"
            else "no screen blocks"
        )
        raise _error(message)
    requested = time_s if kind == "time" else screen_position_m
    tolerance = tolerance_s if kind == "time" else screen_tolerance_m
    unit = "s" if kind == "time" else "m"
    field = "gdf_time_s" if kind == "time" else "gdf_screen_position_m"
    available = ", ".join(f"{t:.12g} {unit} ({n} particles)" for t, n in _inventory(blocks))
    matches = (
        blocks
        if requested is None
        else [b for b in blocks if abs(b["value"] - requested) <= tolerance]
    )
    if len(matches) != 1:
        reason = (
            f"{field} is required" if requested is None else f"unavailable or ambiguous {field}"
        )
        raise _error(f"{reason}; available {kind} values: {available}")
    arrays = {}
    for child in matches[0]["children"]:
        name = child["name"]
        if name in arrays:
            raise _error(f"duplicate array {name}")
        arrays[name] = child["value"]
    required = (*_REQUIRED, "t") if kind == "position" else _REQUIRED
    missing = set(required) - arrays.keys()
    if missing:
        raise _error(f"missing required particle arrays: {', '.join(sorted(missing))}")
    converted = {}
    for name in (*required, "G", "nmacro"):
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        if value.ndim != 1 or value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise _error(f"{name} must be a finite numeric one-dimensional particle array")
        converted[name] = value.astype(np.float64)
    count = len(converted["x"])
    if count == 0:
        raise _error("empty selected output block")
    if any(len(a) != count for a in converted.values()):
        raise _error("inconsistent particle array lengths")
    mass, charge = converted["m"], converted["q"]
    if not np.allclose(charge, -elementary_charge, rtol=1e-6, atol=0) or not np.allclose(
        mass, electron_mass, rtol=1e-6, atol=0
    ):
        raise _error(
            "records must be electrons: q=-elementary_charge and electron mass "
            "(relative tolerance 1e-6)"
        )
    beta = np.column_stack([converted[k] for k in ("Bx", "By", "Bz")])
    beta2 = np.sum(beta * beta, axis=1)
    if np.any((beta2 <= 0) | (beta2 >= 1)):
        raise _error("invalid beta: require nonzero direction and 0 < beta² < 1")
    gamma = 1 / np.sqrt(1 - beta2)
    if "G" in converted and not np.allclose(converted["G"], gamma, rtol=1e-6, atol=1e-9):
        discrepancy = np.max(np.abs(converted["G"] - gamma))
        raise _error(f"G disagrees with beta-derived gamma; maximum discrepancy {discrepancy:.9g}")
    # Stable gamma-1 avoids cancellation for nonrelativistic records.
    energy = (gamma * gamma * beta2 / (gamma + 1)) * mass * c**2 / (elementary_charge * 1e3)
    weights = converted.get("nmacro")
    if weights is None and normalization == "gdf_charge":
        raise _error("gdf_charge requires nmacro")
    if weights is not None and np.any(weights <= 0):
        raise _error("nmacro must be finite and strictly positive")
    signed = None if weights is None else float(np.sum(charge * weights))
    absolute = None if signed is None else -signed
    if absolute is not None and (not np.isfinite(absolute) or absolute <= 0):
        raise _error("total bunch charge must be finite and nonzero")
    relative = np.ones(count) if weights is None else weights / weights.max()
    return GDFBeam(
        float(matches[0]["value"]) if kind == "time" else None,
        np.column_stack([converted[k] for k in ("x", "y", "z")]),
        beta,
        energy,
        relative / relative.sum(),
        signed,
        absolute,
        digest,
        screen_position_m=float(matches[0]["value"]) if kind == "position" else None,
        arrival_time_s=converted.get("t"),
    )


def inspect_gdf(
    path: str | Path,
    time_s: float | None = None,
    tolerance_s: float = 1e-15,
    *,
    screen_position_m: float | None = None,
    screen_tolerance_m: float = 1e-9,
) -> dict:
    """Inventory outputs and describe the selected beam's lab coordinates in SI.

    A weighted centroid is a possible target origin only when the user chooses
    to place the target there; file coordinates do not locate the physical target.
    Screen labels may use GPT's screen coordinate system, so actual z arrays
    are always reported separately. No coordinate transformation is performed.
    """
    blocks, _ = _read(path)
    outputs = [
        dict(
            kind=b["name"],
            coordinate=float(b["value"]),
            unit="s" if b["name"] == "time" else "m",
            particles=_inventory([b])[0][1],
        )
        for b in blocks
    ]
    payload = {"path": str(Path(path).resolve()), "outputs": outputs, "selected": None}
    if (
        time_s is None
        and screen_position_m is None
        and sum(b["name"] == "time" for b in blocks) != 1
    ):
        return payload
    beam = load_gdf_beam(
        path,
        time_s,
        tolerance_s,
        screen_position_m=screen_position_m,
        screen_tolerance_m=screen_tolerance_m,
    )
    coordinates = {}
    for index, axis in enumerate(("x", "y", "z")):
        values = beam.position_m[:, index]
        coordinates[axis] = {
            "min_m": float(values.min()),
            "max_m": float(values.max()),
            "weighted_mean_m": float(np.dot(beam.probabilities, values)),
        }
    payload["selected"] = {
        "time_s": beam.time_s,
        "screen_position_m": beam.screen_position_m,
        "particles": len(beam.energy_keV),
        "coordinates": coordinates,
        "energy_min_keV": float(beam.energy_keV.min()),
        "energy_max_keV": float(beam.energy_keV.max()),
        "centroid_z_origin_m": coordinates["z"]["weighted_mean_m"],
        "origin_guidance": "Use the physical target lab-z coordinate when known. "
        "The centroid is an optional placement choice, not an inferred target location.",
    }
    return payload
