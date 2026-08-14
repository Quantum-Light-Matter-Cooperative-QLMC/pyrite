# Result persistence schema

PyRITE result artifacts use HDF5 schema `pyrite.result`, version 1. Historical
path tokens still end in `.pkl`; the suffix is part of the checkpoint/CAS
layout, not a claim about the bytes. Open a current artifact directly with
`h5py.File("line.pkl")` or any HDF5 reader.

## Container header

The HDF5 root has three required attributes:

| Attribute | Type | Version 1 value | Meaning |
| --- | --- | --- | --- |
| `schema` | UTF-8 string | `pyrite.result` | Format discriminator |
| `schema_version` | integer | `1` | Container/tree encoding version |
| `identity_version` | integer | `1` | Dataset-identity normalization version |

The root group `value` contains the encoded payload. Readers must reject an
unknown `schema` or `schema_version`; identity compatibility remains governed by
the checkpoint manifest and dataset-identity rules.

## Typed tree encoding

Every node has a UTF-8 `kind` attribute. The encoding preserves mapping order,
non-string mapping keys, absent mapping entries, explicit nulls, list/tuple
kind, array dtype, and NumPy-scalar dtype.

| `kind` | HDF5 object | Payload |
| --- | --- | --- |
| `mapping` | group | `items/00000000`, …; each item has typed `key` and `value` nodes |
| `list`, `tuple` | group | typed nodes under `items/00000000`, … |
| `null` | group | no payload |
| `array` | dataset | native NumPy numeric/boolean/byte array with exact shape and dtype |
| `unicode-array` | dataset | variable-width UTF-8 values plus `numpy_dtype` attribute |
| `numpy-scalar` | scalar dataset | exact NumPy scalar dtype |
| `python-scalar` | scalar dataset | Boolean, integer, float, or complex value |
| `string` | scalar dataset | variable-width UTF-8 text |
| `bytes` | `uint8[N]` dataset | uninterpreted bytes |

Non-scalar numeric datasets use the standard HDF5 gzip filter and shuffle
filter. Historical compression levels 1–22 remain accepted by the API; levels
above gzip level 9 map to 9. Scalar and one-element datasets are not chunked.
Object-dtype arrays and arbitrary Python objects are rejected.

## Result-store semantics

Component stores have the logical shape `{configuration: {E0_keV: record}}`.
CAS blobs contain the runner output mapping for one case. The typed tree above
is the normative physical encoding; these tables define the scientific fields.

| Record field | Dtype | Shape | Unit / meaning |
| --- | --- | --- | --- |
| `E_grid` | `float64` | `[N]` | photon energy, eV |
| `spec` | `float64` | `[N]` | line spectrum, photons/e/sr/eV |
| `spec_coherent` | `float64` | `[N]` | coherent companion spectrum; key absent when not computed |
| `brem` | `float64` | `[N]` | bremsstrahlung interpolated to `E_grid`, photons/e/sr/eV |
| `E_grid_brem` | `float64` or null | `[M]` | wide bremsstrahlung photon-energy grid, eV |
| `brem_wide` | `float64` or null | `[M]` | wide bremsstrahlung spectrum, photons/e/sr/eV |
| `E_pk` | `float64` | scalar | line peak energy, eV |
| `fwhm` | `float64` | scalar | line full width at half maximum, eV |
| `eta` | `float64` | scalar | backscattered-electron fraction |
| `hit_frac` | `float64` | scalar | finite-footprint hit fraction; `NaN` for old records |
| `scale` | `float64` | scalar | conversion from per-e/sr to per-s/nA |
| `source_current_na` | `float64` | scalar | reporting source current, nA |
| `case` | mapping | — | version-1 resolved simulation case |

CAS runner mappings add `n_segments` (integer count), `crystal` (catalog key),
`E0_keV` (incident electron energy in keV), and optional underscore-prefixed
timing/backend diagnostics. Diagnostics are metadata, not scientific arrays.

## Compatibility and writes

Readers sniff the HDF5 eight-byte signature. If absent, they permanently fall
back to the previous zstd-compressed pickle, gzip-compressed pickle, and plain
pickle readers. Loading does not rewrite. The next normal store save writes the
same logical payload as HDF5 through the existing same-filesystem temporary file
and `os.replace`; no bulk migration is required.

HDF5 requires random-access output. `dump_stream` therefore builds a seekable
spooled file, then copies the completed container to the caller's pipe. This
retains stdout/remote-pull compatibility but transfer begins after encoding.

## Independent inspection

This minimal reader requires h5py but no PyRITE import:

```python
import h5py

with h5py.File("checkpoints/hopg/line.pkl", "r") as result:
    assert result.attrs["schema"] == "pyrite.result"
    print(result.attrs["schema_version"])
    result.visit(print)
```

The numbered tree is deliberately explicit rather than dependent on Python
class names. External tools may inspect arrays directly; reconstructing the
ordered Python-shaped mapping requires following each item's `key` and `value`.
