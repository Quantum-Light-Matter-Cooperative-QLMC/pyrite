"""Portable, versioned electron transverse-entry-state specifications (#680).

These are **source descriptions**, not computed wavefunctions. They contain no
Python callables, NumPy arrays, or backend objects. The runtime adapter reuses
:mod:`pyrite.montecarlo.channeling.states` instead of duplicating its physics.

A file-backed sample stores an artifact reference *and* a content digest. The
artifact is checked on every conversion to a runtime state. No initial-state
projection into a production transport Case is implemented by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

SCHEMA = "pyrite.quantum.transverse-state.v1"
_ANALYTIC = {
    "gaussian", "hermite_gauss", "laguerre_gauss", "bessel_gauss", "airy",
    "aperture_probe",
}
_SUPPORTED = _ANALYTIC | {"sampled", "superposition", "mixture", "translated"}
_FIELDS = {
    "gaussian": {"waist_ang", "dimension", "center_ang"},
    "hermite_gauss": {"m", "n", "waist_ang", "center_ang"},
    "laguerre_gauss": {"ell", "p", "waist_ang", "center_ang"},
    "bessel_gauss": {"ell", "k_radial_inv_ang", "waist_ang", "center_ang"},
    "airy": {"scale_ang", "truncation", "dimension", "center_ang"},
    "aperture_probe": {
        "kinetic_energy_eV", "semiangle_rad", "inner_semiangle_rad",
        "aberrations", "ell", "center_ang",
    },
    "sampled": {
        "path", "sha256", "shape", "dtype", "spacing_ang", "origin_ang",
        "kinetic_energy_eV",
    },
    "translated": {"offset_ang"},
    "superposition": set(),
    "mixture": set(),
}
_REQUIRED = {
    "gaussian": {"waist_ang"},
    "hermite_gauss": {"m", "waist_ang"},
    "laguerre_gauss": {"ell", "waist_ang"},
    "bessel_gauss": {"ell", "k_radial_inv_ang", "waist_ang"},
    "airy": {"scale_ang", "truncation"},
    "aperture_probe": {"kinetic_energy_eV", "semiangle_rad"},
    "sampled": {"path", "sha256", "shape", "dtype", "spacing_ang", "origin_ang"},
    "translated": {"offset_ang"},
    "superposition": set(),
    "mixture": set(),
}


def _finite(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _coords(
    name: str, value: object, dimension: int, *, positive: bool = False
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a coordinate sequence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = (value,) * dimension
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be a coordinate sequence") from exc
        if len(values) != dimension:
            raise ValueError(f"{name} must have {dimension} coordinates")
    return tuple(_finite(name, item, positive=positive) for item in values)


def _params(kind: str, raw: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    extra = set(raw) - _FIELDS[kind]
    missing = _REQUIRED[kind] - set(raw)
    if extra or missing:
        raise ValueError(f"{kind}: unknown parameters {sorted(extra)}, missing {sorted(missing)}")
    values = dict(raw)
    if kind in ("gaussian", "airy"):
        dim = _integer("dimension", values.get("dimension", 2 if kind == "gaussian" else 1))
        if dim not in (1, 2):
            raise ValueError("dimension must be 1 or 2")
        values["dimension"] = dim
    elif kind == "hermite_gauss":
        values["m"] = _integer("m", values["m"], minimum=0)
        values["n"] = None if values.get("n") is None else _integer("n", values["n"], minimum=0)
        dim = 1 if values["n"] is None else 2
    elif kind in ("laguerre_gauss", "bessel_gauss", "aperture_probe"):
        dim = 2
        values["ell"] = _integer("ell", values.get("ell", 0))
        if kind == "laguerre_gauss":
            values["p"] = _integer("p", values.get("p", 0), minimum=0)
    elif kind == "sampled":
        shape = values["shape"]
        if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
            raise TypeError("shape must be a sequence of sample counts")
        values["shape"] = tuple(_integer("shape", n, minimum=2) for n in shape)
        dim = len(values["shape"])
        if dim not in (1, 2):
            raise ValueError("sampled states must be one- or two-dimensional")
        if not isinstance(values["path"], str) or not values["path"].strip():
            raise ValueError("sample path must be a nonempty string")
        digest = values["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("sha256 must be a 64-digit lowercase hexadecimal digest")
        # Use NumPy only when constructing sampled specs. Other recipe types
        # remain import-light and can be parsed without a numerical backend.
        import numpy as np

        dtype = np.dtype(values["dtype"])
        if dtype.kind not in ("f", "c") or dtype.hasobject:
            raise ValueError("sample dtype must be a real or complex floating dtype")
        values["dtype"] = dtype.str
        values["spacing_ang"] = _coords("spacing_ang", values["spacing_ang"], dim, positive=True)
        values["origin_ang"] = _coords("origin_ang", values["origin_ang"], dim)
        if values.get("kinetic_energy_eV") is not None:
            values["kinetic_energy_eV"] = _finite(
                "kinetic_energy_eV", values["kinetic_energy_eV"], positive=True
            )
        else:
            values["kinetic_energy_eV"] = None
    elif kind == "translated":
        # Dimension checked against the child by __post_init__.
        dim = 0
    else:
        dim = 0

    for name in ("waist_ang", "scale_ang", "truncation", "k_radial_inv_ang",
                 "kinetic_energy_eV", "semiangle_rad"):
        if name in values and kind != "sampled":
            values[name] = _finite(name, values[name], positive=True)
    if kind == "aperture_probe":
        inner = _finite("inner_semiangle_rad", values.get("inner_semiangle_rad", 0.0))
        if not 0.0 <= inner < values["semiangle_rad"]:
            raise ValueError("need 0 <= inner_semiangle_rad < semiangle_rad")
        values["inner_semiangle_rad"] = inner
        aberrations = values.get("aberrations", {})
        if not isinstance(aberrations, Mapping):
            raise TypeError("aberrations must be a mapping")
        # Keep in lockstep with the existing abTEM convention; arbitrary phase
        # callbacks are intentionally not part of a serializable source.
        allowed = {f"C{n}{m}" for n, ms in ((1,(0,2)),(2,(1,3)),(3,(0,2,4)),
                                            (4,(1,3,5)),(5,(0,2,4,6))) for m in ms}
        allowed |= {"phi"+key[1:] for key in allowed}
        if set(aberrations) - allowed:
            raise ValueError(f"unknown aberration keys: {sorted(set(aberrations) - allowed)}")
        values["aberrations"] = tuple(sorted(
            (str(key), _finite(str(key), v)) for key, v in aberrations.items()
        ))
    if kind not in ("sampled", "translated", "superposition", "mixture"):
        values["center_ang"] = _coords("center_ang", values.get("center_ang", 0.0), dim)
    return tuple(sorted(values.items()))


def _json_parameters(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: dict(value) if key == "aberrations" else list(value) if isinstance(value, tuple)
            else value for key, value in params}


@dataclass(frozen=True)
class TransverseStateSpec:
    """Immutable, validated, JSON-safe electron transverse state recipe.

    Version 1 supports analytic sources, coherent superpositions, incoherent
    mixtures, translations, and content-verified ``.npy`` sample artifacts.
    ``terms`` are ``(weight, child_spec)`` pairs (complex for superpositions,
    nonnegative real for mixtures). Coordinates are Å, momentum is Å^-1.

    A state description is *not* a declaration that production transport
    supports injection of that state: the Beam adapter refuses until the
    projected source is explicitly integrated and validated.
    """

    kind: str
    parameters: tuple[tuple[str, Any], ...] = ()
    terms: tuple[tuple[complex | float, "TransverseStateSpec"], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _SUPPORTED:
            raise ValueError(f"unsupported transverse state kind {self.kind!r}")
        if isinstance(self.parameters, Mapping):
            values = dict(self.parameters)
        else:
            pairs = tuple(self.parameters)
            values = dict(pairs)
            if len(values) != len(pairs):
                raise ValueError("duplicate state parameter")
        object.__setattr__(self, "parameters", _params(self.kind, values))
        pairs = tuple(self.terms)
        if self.kind in _ANALYTIC | {"sampled"} and pairs:
            raise ValueError(f"{self.kind} cannot have child states")
        if self.kind in ("superposition", "mixture", "translated"):
            if not pairs or (self.kind == "translated" and len(pairs) != 1):
                requirement = "one child" if self.kind == "translated" else "at least one child"
                raise ValueError(f"{self.kind} needs {requirement}")
            if any(not isinstance(spec, TransverseStateSpec) for _, spec in pairs):
                raise TypeError("every child must be a TransverseStateSpec")
            if len({spec.dimension for _, spec in pairs}) != 1:
                raise ValueError("all component states must have the same dimension")
            if self.kind == "translated":
                if pairs[0][0] != 1:
                    raise ValueError("translated child must have unit weight")
                offset = _coords("offset_ang", values["offset_ang"], pairs[0][1].dimension)
                object.__setattr__(self, "parameters", (("offset_ang", offset),))
                object.__setattr__(self, "terms", ((1.0, pairs[0][1]),))
            elif self.kind == "mixture":
                weights = tuple(_finite("mixture weight", w) for w, _ in pairs)
                total = sum(weights)
                if any(w < 0 for w in weights) or not math.isfinite(total) or total <= 0:
                    raise ValueError("mixture weights must be nonnegative with positive finite sum")
                object.__setattr__(
                    self, "terms",
                    tuple((w / total, spec) for w, (_, spec) in zip(weights, pairs, strict=True)),
                )
            else:
                weights = tuple(complex(w) for w, _ in pairs)
                if (any(not math.isfinite(w.real) or not math.isfinite(w.imag)
                        for w in weights) or not any(weights)):
                    raise ValueError("superposition needs finite, nonzero coefficients")
                object.__setattr__(
                    self, "terms", tuple((w, s) for w, (_, s) in zip(weights, pairs, strict=True))
                )

    @property
    def dimension(self) -> int:
        if self.kind in ("translated", "mixture", "superposition"):
            return self.terms[0][1].dimension
        p = dict(self.parameters)
        if self.kind == "sampled":
            return len(p["shape"])
        if self.kind == "hermite_gauss":
            return 1 if p["n"] is None else 2
        if self.kind in ("gaussian", "airy"):
            return p["dimension"]
        return 2

    @classmethod
    def recipe(cls, kind: str, **parameters: Any) -> Self:
        """Create a validated analytic-state recipe from named parameters."""
        return cls(kind, tuple(parameters.items()))

    @classmethod
    def superpose(cls, terms: Sequence[tuple[complex, "TransverseStateSpec"]]) -> Self:
        return cls("superposition", terms=tuple(terms))

    @classmethod
    def mixture(cls, components: Sequence[tuple[float, "TransverseStateSpec"]]) -> Self:
        return cls("mixture", terms=tuple(components))

    def translated(self, offset_ang: object) -> Self:
        return type(self)("translated", (("offset_ang", offset_ang),), ((1.0, self),))

    @classmethod
    def from_npy(
        cls, path: str | Path, *, spacing_ang: object, origin_ang: object = 0.0,
        kinetic_energy_eV: float | None = None,
    ) -> Self:
        """Reference a saved .npy array by its exact contents (no embedding).

        Relative artifact paths require ``base_dir`` when later calling
        :meth:`to_runtime`. The SHA covers contiguous array data; dtype and
        shape are tracked separately. Only 1D/2D finite floating arrays with
        nonzero amplitude are accepted. Files remain external to JSON.
        """
        import numpy as np

        file = Path(path)
        if file.suffix != ".npy":
            raise ValueError("sample asset must be a .npy file")
        array = np.load(file, allow_pickle=False)
        _validate_samples(array)
        return cls.recipe(
            "sampled", path=str(file), sha256=_array_digest(array),
            shape=array.shape, dtype=array.dtype.str, spacing_ang=spacing_ang,
            origin_ang=origin_ang, kinetic_energy_eV=kinetic_energy_eV,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-safe, versioned description."""
        payload: dict[str, Any] = {
            "schema": SCHEMA, "kind": self.kind,
            "parameters": _json_parameters(self.parameters),
        }
        if self.kind in ("superposition", "mixture", "translated"):
            payload["terms"] = [
                {"weight": ([complex(weight).real, complex(weight).imag]
                            if self.kind == "superposition" else float(weight)),
                 "state": spec.to_dict()}
                for weight, spec in self.terms
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Parse exact schema-v1 dictionaries; reject unknown/missing keys."""
        if not isinstance(payload, Mapping):
            raise TypeError("transverse-state spec must be a mapping")
        kind = payload.get("kind")
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"unsupported transverse state schema (expected {SCHEMA})")
        expected = {"schema", "kind", "parameters"}
        if kind in ("superposition", "mixture", "translated"):
            expected.add("terms")
        if set(payload) != expected:
            raise ValueError(f"transverse state fields must be exactly {sorted(expected)}")
        parameters = payload["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        terms = ()
        if "terms" in payload:
            raw = payload["terms"]
            if not isinstance(raw, (tuple, list)):
                raise TypeError("terms must be a list")
            out = []
            for item in raw:
                if not isinstance(item, Mapping) or set(item) != {"weight", "state"}:
                    raise ValueError("each term requires only weight and state")
                weight = item["weight"]
                if kind == "superposition":
                    if not isinstance(weight, (tuple, list)) or len(weight) != 2:
                        raise ValueError("complex weights must be [real, imag]")
                    weight = complex(
                        _finite("real weight", weight[0]), _finite("imag weight", weight[1])
                    )
                else:
                    weight = _finite("weight", weight)
                out.append((weight, cls.from_dict(item["state"])))
            terms = tuple(out)
        return cls(str(kind), tuple(parameters.items()), terms)

    def canonical_json(self) -> str:
        """Canonical scientific identity (excludes sample file *locations*).

        Asset paths are local execution references, not physical source
        parameters. The sample digest, shape, dtype and grid metadata remain
        part of the scientific identity, including in nested mixed states.
        """
        def identity(spec: TransverseStateSpec) -> dict[str, Any]:
            payload = spec.to_dict()
            if spec.kind == "sampled":
                del payload["parameters"]["path"]
            if "terms" in payload:
                for item, (_, child) in zip(payload["terms"], spec.terms, strict=True):
                    item["state"] = identity(child)
            return payload

        return json.dumps(identity(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def digest(self) -> str:
        """SHA256 of the canonical source descriptor, independent of runtime arrays."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_runtime(self, *, base_dir: str | Path | None = None):
        """Build an existing channeling EntryState from this portable spec.

        This is a state *construction* adapter, not a production Scene lowerer.
        The data file's content hash, shape and dtype must match its spec.
        """
        from .montecarlo.channeling import states

        params = dict(self.parameters)
        if self.kind in _ANALYTIC:
            if self.kind == "aperture_probe":
                params["aberrations"] = dict(params["aberrations"])
                # Arbitrary phase-plate callables are intentionally excluded.
            return getattr(states, self.kind)(**params)
        if self.kind == "sampled":
            import numpy as np

            path = Path(params["path"])
            if not path.is_absolute():
                if base_dir is None:
                    raise ValueError("relative sample asset requires base_dir")
                path = Path(base_dir) / path
            array = np.load(path, allow_pickle=False)
            _validate_samples(array)
            if (array.dtype.str != params["dtype"] or tuple(array.shape) != params["shape"]
                    or _array_digest(array) != params["sha256"]):
                raise ValueError(f"sample asset content differs from pinned metadata: {path}")
            return states.from_array(
                array, spacing_ang=params["spacing_ang"], origin_ang=params["origin_ang"],
                kinetic_energy_eV=params["kinetic_energy_eV"],
            )
        children = [(w, s.to_runtime(base_dir=base_dir)) for w, s in self.terms]
        if self.kind == "translated":
            return states.translate(children[0][1], params["offset_ang"])
        if self.kind == "mixture":
            return states.mixture(children)
        return states.superpose(children)


def _validate_samples(array: Any) -> None:
    import numpy as np

    if array.ndim not in (1, 2) or any(n < 2 for n in array.shape):
        raise ValueError("sample asset must be a 1D or 2D array with >=2 samples per axis")
    if array.dtype.kind not in ("f", "c") or not np.all(np.isfinite(array)):
        raise ValueError("sample asset must be finite real or complex floating data")
    if not np.any(array):
        raise ValueError("sample asset must have nonzero amplitude")


def _array_digest(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
