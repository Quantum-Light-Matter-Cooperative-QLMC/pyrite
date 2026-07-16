# Per-Beam Line Grids and Quantized Angles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each configured beam energy its own exact, uniform coherent-line grid while canonicalizing every generated polar and azimuthal angle to the nearest half degree.

**Architecture:** Extend the immutable catalog scan model with an optional `E_grid_line_by_energy` mapping parsed from an ordered TOML array of `{energy_keV, grid}` entries. Project that mapping into `Sweep`, where `build_cases` selects and encodes the grid after choosing each beam energy; the existing fixed `E_grid_line` remains the higher-precedence compatibility path. Canonicalize angle inputs once at the `build_cases` boundary with symmetric nearest-half rounding and stable deduplication so names, fields, geometry, and seeds share the same values.

**Tech Stack:** Python 3.11+, NumPy, `tomllib`, frozen dataclasses, `MappingProxyType`, pytest, TOML catalog data.

## Global Constraints

- Standard beam energies, in order, are exactly `30, 50, 100, 150, 200, 250, 300 keV`.
- Standard polar inputs are ten endpoint-inclusive values from `linspace(0, 89, 10)`.
- Standard azimuthal inputs are ten endpoint-inclusive values from `linspace(90, 180, 10)`.
- Every angle reaching `build_cases`, including direct-library and override inputs, is rounded to the nearest `0.5` degree with a rule symmetric for positive and negative values.
- Angle duplicates introduced by quantization are removed while preserving first-occurrence order; no other sweep dimension is rounded or deduplicated.
- Standard line-grid endpoints are `10-2500`, `10-3000`, `50-3500`, `50-4000`, `50-4500`, `50-5000`, and `50-5000 eV` for the seven beam energies respectively.
- Every standard line grid is uniform, includes both endpoints exactly, and has spacing approximately `3 eV`.
- A fixed `E_grid_line` overrides `E_grid_line_by_energy`; existing fixed-array and exact energy-grid codec behavior remains compatible.
- Catalog mappings must contain exactly one entry for every configured beam energy and no missing, duplicate, or extra entries.
- All exposed catalog arrays and mappings remain deeply read-only.
- The coherent-spectrum `E_res > 10 eV` physics boundary, bremsstrahlung grids, and catalog schema version `1` remain unchanged.
- Use `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py ...` for repository checks.

---

### Task 1: Immutable catalog and configuration support

**Files:**
- Modify: `src/cxr_mc/materials/catalog.py`
- Modify: `src/cxr_mc/config.py`
- Modify: `src/cxr_mc/sweep.py` (dataclass field only; selection remains Task 2)
- Test: `tests/test_material_catalog.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: existing `_grid(value, path, errors) -> np.ndarray | None`, `_readonly(values) -> np.ndarray`, `ScanSpec`, `material_grid()`, `material_sweep()`, and `trajectory_sweep()`.
- Produces: `LineGridByEnergy = Mapping[float, np.ndarray]`; `ScanSpec.E_grid_line: np.ndarray | None`; `ScanSpec.E_grid_line_by_energy: LineGridByEnergy | None`; and the inert `Sweep.E_grid_line_by_energy: Mapping[float, np.ndarray] | None` field consumed by Task 2.

- [ ] **Step 1: Write failing parser, validation, immutability, and projection tests**

Add a helper variant of `_minimal_catalog` input in `tests/test_material_catalog.py` by replacing its fixed line descriptor with:

```toml
E_grid_line_by_energy = [
  { energy_keV = 25.0, grid = { linspace = { start = 10.0, stop = 58.0, num = 17, endpoint = true } } },
  { energy_keV = 30.0, grid = { values = [20.0, 23.0, 26.0] } },
]
```

Then add tests with these assertions:

```python
def test_per_beam_line_grids_are_exact_read_only_and_projected(tmp_path, monkeypatch):
    from cxr_mc import config
    from cxr_mc.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows='''
[materials.sample]
label = "sample"
profile = "base"
crystal = "mos2"
'''
    ).replace(
        "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }",
        '''E_grid_line_by_energy = [
  { energy_keV = 25.0, grid = { linspace = { start = 10.0, stop = 58.0, num = 17, endpoint = true } } },
  { energy_keV = 30.0, grid = { values = [20.0, 23.0, 26.0] } },
]''',
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    scan = catalog.material("sample").scan

    assert scan.E_grid_line is None
    assert tuple(scan.E_grid_line_by_energy) == (25.0, 30.0)
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[25.0], np.linspace(10.0, 58.0, 17))
    with pytest.raises(TypeError):
        scan.E_grid_line_by_energy[25.0] = np.array([1.0])
    with pytest.raises(ValueError):
        scan.E_grid_line_by_energy[25.0][0] = 1.0

    monkeypatch.setattr(config, "CATALOG", catalog)
    grid = config.material_grid("sample")
    sweep = config.material_sweep("sample")
    assert grid["E_grid_line"] is None
    assert grid["E_grid_line_by_energy"] is scan.E_grid_line_by_energy
    assert sweep.E_grid_line is None
    assert sweep.E_grid_line_by_energy is scan.E_grid_line_by_energy
```

Add a fixed material override test proving it replaces an inherited mapping:

```python
def test_fixed_material_line_grid_overrides_profile_mapping(tmp_path):
    from cxr_mc.materials import load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(
        'profile = "base"',
        'profile = "base"\nE_grid_line = { values = [75.0, 78.0] }',
        1,
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan
    np.testing.assert_array_equal(scan.E_grid_line, [75.0, 78.0])
    assert scan.E_grid_line_by_energy is None
```

Finally parameterize malformed mappings and assert path-qualified errors:

```python
@pytest.mark.parametrize(
    ("replacement", "error_path"),
    [
        ("{ energy_keV = 25.0, grid = 50.0 },\n  { energy_keV = 25.0, grid = 60.0 }", "E_grid_line_by_energy[1].energy_keV"),
        ("{ energy_keV = 25.0, grid = 50.0 }", "E_grid_line_by_energy"),
        ("{ energy_keV = 25.0, grid = 50.0 },\n  { energy_keV = 30.0, grid = 60.0 },\n  { energy_keV = 40.0, grid = 70.0 }", "E_grid_line_by_energy[2].energy_keV"),
    ],
)
def test_per_beam_line_grid_keys_match_beam_energies(tmp_path, replacement, error_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    assert error_path in str(caught.value)
```

- [ ] **Step 2: Run the new tests and confirm the schema is not implemented**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py -k "per_beam or fixed_material_line_grid"
```

Expected: FAIL because `E_grid_line_by_energy` is an unknown key and `ScanSpec`/`Sweep` do not expose the mapping.

- [ ] **Step 3: Implement strict parsing and deep immutability**

In `src/cxr_mc/materials/catalog.py`:

```python
LineGridByEnergy = Mapping[float, np.ndarray]

_SCAN_KEYS = (
    "thickness_ang",
    "thickness_layers",
    "energy_keV",
    "tilt_deg",
    "tilt_azim_deg",
    "E_grid_line",
    "E_grid_line_by_energy",
    "E_grid_brem",
)
```

Change `ScanSpec` so the two line-grid forms are optional:

```python
@dataclass(frozen=True)
class ScanSpec:
    thickness_ang: np.ndarray
    energy_keV: np.ndarray
    tilt_deg: np.ndarray
    tilt_azim_deg: np.ndarray
    E_grid_line: np.ndarray | None
    E_grid_line_by_energy: LineGridByEnergy | None
    E_grid_brem: np.ndarray
    thickness_layers: np.ndarray | None = None
```

Add a parser that validates entry shape, finite positive energies and grids, duplicates, then exact key equality with `energy_keV`; return `MappingProxyType(dict(entries))` so insertion order is retained and each grid comes from `_grid`/`_readonly`:

```python
def _line_grids_by_energy(
    value: object,
    energy_grid: np.ndarray | None,
    path: str,
    errors: _Errors,
) -> LineGridByEnergy | None:
    if not isinstance(value, list) or not value:
        errors.add(path, "must be a nonempty array of line-grid entries")
        return None
    parsed: dict[float, np.ndarray] = {}
    energy_indexes: dict[float, int] = {}
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        row = _table(item, item_path, errors)
        if row is None:
            continue
        errors.keys(row, item_path, {"energy_keV", "grid"})
        energy = _number(row.get("energy_keV"))
        if energy is None or energy <= 0:
            errors.add(f"{item_path}.energy_keV", "must be finite and positive")
            continue
        if energy in parsed:
            errors.add(f"{item_path}.energy_keV", f"duplicates beam energy {energy:g}")
            continue
        energy_indexes[energy] = index
        grid = _grid(row.get("grid"), f"{item_path}.grid", errors)
        if grid is None or np.any(grid <= 0):
            if grid is not None:
                errors.add(f"{item_path}.grid", "values must be positive")
            continue
        parsed[energy] = grid
    if energy_grid is not None:
        configured = set(float(value) for value in energy_grid)
        mapped = set(parsed)
        missing = sorted(configured - mapped)
        extra = sorted(mapped - configured)
        if missing:
            errors.add(path, f"missing beam energies {missing}")
        for energy in extra:
            index = energy_indexes[energy]
            errors.add(f"{path}[{index}].energy_keV", f"is not configured in energy_keV: {energy:g}")
    return MappingProxyType(parsed) if parsed else None
```

Refactor `_scan` and `_parse_profiles` to treat `E_grid_line` and `E_grid_line_by_energy` as mutually exclusive alternatives rather than independently required grids. Parse ordinary grids first, parse the mapping only after `energy_keV` exists, and require exactly one line-grid form. When a material supplies either form, remove the inherited other form before merging its overrides:

```python
if "E_grid_line" in row:
    values.pop("E_grid_line_by_energy", None)
elif "E_grid_line_by_energy" in row:
    values.pop("E_grid_line", None)
```

Export `LineGridByEnergy` in `__all__`.

- [ ] **Step 4: Project both line-grid forms through configuration**

In `src/cxr_mc/sweep.py`, import `Mapping` and add the inert dataclass field without changing `build_cases` yet:

```python
from collections.abc import Mapping, Sequence

E_grid_line: np.ndarray | None = None
E_grid_line_by_energy: Mapping[float, np.ndarray] | None = None
E_grid_brem: np.ndarray | None = None
```

In `src/cxr_mc/config.py`, include the mapping in `material_grid`, `material_sweep`, and `trajectory_sweep`:

```python
"E_grid_line": scan.E_grid_line,
"E_grid_line_by_energy": scan.E_grid_line_by_energy,
```

and:

```python
E_grid_line=scan.E_grid_line,
E_grid_line_by_energy=scan.E_grid_line_by_energy,
```

Keep returning the same immutable mapping rather than copying writable arrays.

- [ ] **Step 5: Run focused catalog and configuration tests**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py tests/test_sweep.py -k "per_beam or fixed_material_line_grid or material_grid or material_sweep or trajectory_sweep"
```

Expected: PASS.

- [ ] **Step 6: Commit the catalog boundary**

```bash
git add src/cxr_mc/materials/catalog.py src/cxr_mc/config.py src/cxr_mc/sweep.py tests/test_material_catalog.py tests/test_sweep.py
git commit -m "feat: add per-beam line grid catalog schema"
```

### Task 2: Per-case grid selection and angle canonicalization

**Files:**
- Modify: `src/cxr_mc/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `Sweep.E_grid_line`, `Sweep.E_grid_line_by_energy`, `encode_energy_grid`, and `_seq`.
- Produces: `_quantized_angles(values: ScalarOrSeq) -> np.ndarray`; `_line_grid_for_energy(sweep: Sweep, default_grid: np.ndarray, energy_keV: float) -> np.ndarray`; and cases whose encoded line grid, angles, names, radians, and seeds are derived from canonical inputs.

- [ ] **Step 1: Write failing per-energy selection and precedence tests**

Add to `tests/test_sweep.py`:

```python
def test_build_cases_selects_and_encodes_line_grid_for_each_beam_energy():
    grids = {
        30.0: np.linspace(10.0, 2500.0, 831),
        50.0: np.linspace(10.0, 3000.0, 998),
    }
    cases = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=[30.0, 50.0],
            tilt_deg=0.0,
            E_grid_line_by_energy=grids,
            E_grid_brem=75.0,
        )
    )

    assert [case["E0_keV"] for case in cases] == [30.0, 50.0]
    for case in cases:
        expected = grids[case["E0_keV"]]
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), expected)
        assert case["E_grid"] == case["E_grid_line"]


def test_fixed_line_grid_takes_precedence_over_per_beam_mapping():
    fixed = np.array([75.0, 78.0])
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=[30.0, 50.0],
            E_grid_line=fixed,
            E_grid_line_by_energy={30.0: np.array([10.0]), 50.0: np.array([20.0])},
        )
    )
    for case in cases:
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), fixed)


def test_missing_per_beam_line_grid_fails_before_cases_are_built():
    with pytest.raises(ValueError, match=r"no E_grid_line configured for beam energy 50"):
        build_cases(
            Sweep(
                material="mose2",
                energy_keV=[30.0, 50.0],
                E_grid_line_by_energy={30.0: np.array([10.0])},
            )
        )
```

Import `decode_energy_grid` from `cxr_mc._energy_grid` in the test module.

- [ ] **Step 2: Write failing symmetric quantization and stable-deduplication tests**

Add:

```python
def test_build_cases_quantizes_angles_symmetrically_and_removes_duplicates():
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=30.0,
            thickness_ang=100.0,
            tilt_deg=[1.24, 1.26, 1.25, 1.24],
            tilt_azim_deg=[-1.24, -1.26, -1.25, -1.24],
            E_grid_line=np.array([75.0]),
            E_grid_brem=np.array([75.0]),
        )
    )

    assert [(case["tilt_deg"], case["tilt_azim_deg"]) for case in cases] == [
        (1.0, -1.0),
        (1.0, -1.5),
        (1.5, -1.0),
        (1.5, -1.5),
    ]
    assert [case["seed"] for case in cases] == [1, 1001, 2001, 3001]
    assert [case["name"] for case in cases] == [
        "MoSe2 10nm pol=1 az=-1",
        "MoSe2 10nm pol=1 az=-1.5",
        "MoSe2 10nm pol=1.5 az=-1",
        "MoSe2 10nm pol=1.5 az=-1.5",
    ]
```

Use exact half ties (`1.25 -> 1.5`, `-1.25 -> -1.5`) to make the specified away-from-zero symmetric rule observable. Add a geometry assertion by monkeypatching `np.deg2rad` or by asserting the stored values convert exactly:

```python
for case in cases:
    assert np.deg2rad(case["tilt_deg"]) == pytest.approx(
        np.deg2rad(float(case["name"].split("pol=")[1].split()[0]))
    )
```

- [ ] **Step 3: Run the new sweep tests and confirm they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_sweep.py -k "selects_and_encodes or precedence_over_per_beam or missing_per_beam or quantizes_angles"
```

Expected: FAIL because `Sweep` lacks the mapping field and angles are not canonicalized.

- [ ] **Step 4: Add mapping selection and symmetric half-degree canonicalization**

In `src/cxr_mc/sweep.py`, use the `Mapping` import and `Sweep.E_grid_line_by_energy` field added by Task 1, then add focused helpers:

```python
def _quantized_angles(values: ScalarOrSeq) -> np.ndarray:
    """Round degrees to nearest half away from zero at ties, then stable-unique."""
    source = _seq(values)
    scaled = source * 2.0
    quantized = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled) / 2.0
    return np.asarray(list(dict.fromkeys(float(value) for value in quantized)), dtype=float)


def _line_grid_for_energy(sweep: Sweep, default_grid: np.ndarray, energy_keV: float) -> np.ndarray:
    fixed = sweep.E_grid_line if sweep.E_grid_line is not None else sweep.e_grid_eV
    if fixed is not None:
        return np.asarray(fixed, dtype=float)
    if sweep.E_grid_line_by_energy is None:
        return np.asarray(default_grid, dtype=float)
    try:
        return np.asarray(sweep.E_grid_line_by_energy[float(energy_keV)], dtype=float)
    except KeyError:
        raise ValueError(f"no E_grid_line configured for beam energy {energy_keV:g} keV") from None
```

In `build_cases`, compute `tilts = _quantized_angles(sweep.tilt_deg)` and `azimuths = _quantized_angles(sweep.tilt_azim_deg)` before the Cartesian product. Remove the one-time `line_case_grid`; inside the energy loop select and encode the line grid:

```python
line_grid = _line_grid_for_energy(sweep, cp["E_grid"], float(E0))
line_case_grid = encode_energy_grid(line_grid)
```

When the bremsstrahlung grid is implicit, keep its existing start behavior by deriving the common start from the fixed/default line grid; for a per-beam mapping use the minimum first value across mapped grids. Do not otherwise alter brem encoding or per-case cutoff behavior.

- [ ] **Step 5: Run all sweep tests**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_sweep.py
```

Expected: PASS.

- [ ] **Step 6: Commit central case construction**

```bash
git add src/cxr_mc/sweep.py tests/test_sweep.py
git commit -m "feat: select line grids per beam energy"
```

### Task 3: Migrate the bundled standard scan

**Files:**
- Modify: `src/cxr_mc/data/materials.toml`
- Modify: `tests/test_material_catalog.py`
- Modify: `tests/data/material_catalog_golden.json`
- Test: `tests/test_check_config.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: the catalog mapping schema from Task 1 and per-case selection from Task 2.
- Produces: one standard profile shared by every runnable material, with seven beam energies, reduced angles, and seven exact endpoint-inclusive line grids.

- [ ] **Step 1: Add failing bundled-profile assertions**

In `tests/test_material_catalog.py`, add:

```python
def test_standard_profile_uses_requested_angles_energies_and_line_grids():
    from cxr_mc.materials import CATALOG

    expected_bounds = {
        30.0: (10.0, 2500.0),
        50.0: (10.0, 3000.0),
        100.0: (50.0, 3500.0),
        150.0: (50.0, 4000.0),
        200.0: (50.0, 4500.0),
        250.0: (50.0, 5000.0),
        300.0: (50.0, 5000.0),
    }
    for material in CATALOG.materials.values():
        scan = material.scan
        np.testing.assert_array_equal(scan.energy_keV, list(expected_bounds))
        np.testing.assert_array_equal(scan.tilt_deg, np.linspace(0.0, 89.0, 10))
        np.testing.assert_array_equal(scan.tilt_azim_deg, np.linspace(90.0, 180.0, 10))
        assert scan.E_grid_line is None
        assert tuple(scan.E_grid_line_by_energy) == tuple(expected_bounds)
        for energy, (start, stop) in expected_bounds.items():
            grid = scan.E_grid_line_by_energy[energy]
            assert grid[0] == start
            assert grid[-1] == stop
            spacing = np.diff(grid)
            assert np.all(spacing == pytest.approx(spacing[0]))
            assert spacing[0] == pytest.approx(3.0, abs=0.002)
```

In `tests/test_sweep.py`, add an integration test that builds one material sweep and verifies each beam energy decodes to the corresponding catalog grid.

- [ ] **Step 2: Run the bundled-profile test and confirm old defaults fail**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py tests/test_sweep.py -k "standard_profile or catalog_line_grid"
```

Expected: FAIL with the old five beam energies, old angle arrays, and fixed material grids.

- [ ] **Step 3: Replace the standard profile and remove current material overrides**

Replace the standard profile in `src/cxr_mc/data/materials.toml` with:

```toml
[profiles.standard]
thickness_ang = 100000.0
energy_keV = { values = [30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0] }
tilt_deg = { linspace = { start = 0.0, stop = 89.0, num = 10, endpoint = true } }
tilt_azim_deg = { linspace = { start = 90.0, stop = 180.0, num = 10, endpoint = true } }
E_grid_line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2500.0, num = 831, endpoint = true } } },
  { energy_keV = 50.0, grid = { linspace = { start = 10.0, stop = 3000.0, num = 998, endpoint = true } } },
  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 3500.0, num = 1151, endpoint = true } } },
  { energy_keV = 150.0, grid = { linspace = { start = 50.0, stop = 4000.0, num = 1318, endpoint = true } } },
  { energy_keV = 200.0, grid = { linspace = { start = 50.0, stop = 4500.0, num = 1484, endpoint = true } } },
  { energy_keV = 250.0, grid = { linspace = { start = 50.0, stop = 5000.0, num = 1651, endpoint = true } } },
  { energy_keV = 300.0, grid = { linspace = { start = 50.0, stop = 5000.0, num = 1651, endpoint = true } } },
]
E_grid_brem = { arange = { start = 0.0, stop = 30000.0, step = 25.0 } }
```

Remove the HOPG `tilt_azim_deg` override, the h-BN `tilt_deg` override, and every material-level `E_grid_line` entry. Keep all thickness, substrate, stack, label, crystal, and profile declarations unchanged.

- [ ] **Step 4: Refresh only catalog-derived expectations and the golden fixture**

Update count-sensitive expectations in `tests/test_check_config.py` and historical h-BN expectations in `tests/test_material_catalog.py` only if the isolated branch's committed catalog currently requires those baseline corrections. Regenerate `tests/data/material_catalog_golden.json` using the repository's existing golden serializer logic, then inspect the diff to ensure changes are limited to the intended scan fingerprints plus already-committed catalog drift; do not copy changes from the other active crystal-material worktree.

- [ ] **Step 5: Run focused catalog, config, and sweep tests**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py tests/test_check_config.py tests/test_sweep.py
```

Expected: PASS.

- [ ] **Step 6: Run lint, type checking, and full verification**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py typecheck
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
```

Expected: all feature-focused checks PASS. If the canonical full verification still reports any of the three recorded baseline catalog failures, compare them against the initial `716 passed, 3 failed` baseline and report them separately; no new failure may be attributed to this change without investigation.

- [ ] **Step 7: Commit the standard-profile migration**

```bash
git add src/cxr_mc/data/materials.toml tests/test_material_catalog.py tests/test_check_config.py tests/test_sweep.py tests/data/material_catalog_golden.json
git commit -m "feat: narrow standard angular and energy grids"
```

### Task 4: Final compatibility and diff review

**Files:**
- Modify only if verification exposes a feature regression in files already listed above.

**Interfaces:**
- Consumes: all preceding task outputs.
- Produces: a verified isolated branch ready for user-directed integration.

- [ ] **Step 1: Verify fixed and exact-grid compatibility explicitly**

Run:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_sweep.py -k "legacy or exact or scalar or precedence"
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_material_catalog.py -k "scalar or read_only or override"
```

Expected: PASS.

- [ ] **Step 2: Inspect the isolated branch diff and worktree status**

Run:

```bash
git diff 1337f75...HEAD --stat
git diff 1337f75...HEAD -- src/cxr_mc/materials/catalog.py src/cxr_mc/config.py src/cxr_mc/sweep.py src/cxr_mc/data/materials.toml
git status --short
```

Expected: only the spec, plan, catalog/config/sweep implementation, bundled catalog migration, and their tests/fixture are changed; worktree is clean after commits.

- [ ] **Step 3: Request code review and address only verified findings**

Use `superpowers:requesting-code-review` for a requirements and quality review against `docs/superpowers/specs/2026-07-16-per-beam-angular-grids-design.md`. If feedback identifies a concrete defect, use `superpowers:receiving-code-review`, reproduce it with a failing test, apply the smallest correction, rerun the affected focused suite, and commit with an issue-specific message.
