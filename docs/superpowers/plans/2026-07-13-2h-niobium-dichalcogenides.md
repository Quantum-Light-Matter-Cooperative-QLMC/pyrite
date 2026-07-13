# 2H Niobium Dichalcogenides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bulk 2H-a NbS2 and NbSe2 as fully runnable, validated cxr-mc crystal materials.

**Architecture:** Extend the existing data-driven crystal catalog with explicit P6_3/mmc 2b/4f bases, then project two ordinary bulk scan rows through the existing material registry. Add Nb once to the shared atomic/transport policies; no new public API or polytype abstraction is introduced.

**Tech Stack:** Python 3.13, NumPy, pytest, xraydb, TOML, Ruff, pyright

## Global Constraints

- Implement only `nbs2` and `nbse2`; In2Se3 remains a follow-up evaluation.
- Both crystals are bulk 2H-a P6_3/mmc structures with aligned Nb columns.
- Use `B_ang2 = 0.6`, `beam_uvw = (0, 0, 2)`, automatic dominant-reflection selection, 10 micrometre bulk thickness, and the existing 25/30/35 keV scan convention.
- Every structure carries a source, assumptions, limiting case, `Validation:` marker, and an `unverified` validation-ledger row.
- Only a human may mark a validation claim `signed-off`.
- Preserve all unrelated working-tree changes; stage and commit only files named by the active task.
- Set `UV_CACHE_DIR=/tmp/uv-cache` when `uv` cannot write its default cache.

---

## File Structure

- Modify `src/cxr_mc/materials/crystal.py`: make Nb use energy-dependent form factors around its soft-X-ray edges.
- Modify `src/cxr_mc/montecarlo/transport.py`: add Nb atomic transport parameters.
- Modify `tests/test_form_factors.py`: cover Nb atomic data and edge policy.
- Modify `tests/test_montecarlo.py`: cover Nb transport and the analytic fallback path.
- Modify `src/cxr_mc/data/crystal_structures.toml`: add explicit 2H-a NbS2 and NbSe2 conventional cells.
- Modify `tests/test_crystallography.py`: pin lattice, basis, stacking, stoichiometry, volume, and finite couplings.
- Create `docs/validation/2ha-niobium-dichalcogenides.md`: record the structural derivation and limiting checks.
- Modify `docs/physics-validation-ledger.md`: add one unverified row per crystal structure.
- Modify `src/cxr_mc/materials/registry.py`: add scan/crystal rows for both materials.
- Modify `tests/test_sweep.py`: cover registry projection, compositions, reflection selection, and case building.
- Modify `README.md`: list the new catalog entries.

### Task 1: Niobium Atomic and Transport Support

**Files:**
- Modify: `tests/test_form_factors.py:8-45`
- Modify: `tests/test_montecarlo.py:47-66`
- Modify: `src/cxr_mc/materials/crystal.py:49-51`
- Modify: `src/cxr_mc/montecarlo/transport.py:27-44`

**Interfaces:**
- Consumes: `materials.atomic.Z_TABLE`, `materials.atomic.load_henke`, `montecarlo.transport.TRANSPORT_ELEMENTS`, and the existing missing-Mott screened-Rutherford fallback.
- Produces: Nb atomic form factors through the existing APIs and `TRANSPORT_ELEMENTS["Nb"] == {"Z": 41, "A": 92.906, "J_keV": 0.417}`.

- [ ] **Step 1: Write failing atomic-data and edge-policy tests**

In `tests/test_form_factors.py`, import the crystal module, add `"Nb"` to `ELEMENTS`, and add:

```python
import cxr_mc.materials.crystal as crystal_module


def test_niobium_registered_and_edge_prone():
    assert Z_TABLE["Nb"] == 41
    assert "Nb" in crystal_module._EDGE_PRONE

    E, f1, f2 = load_henke("Nb")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)
```

- [ ] **Step 2: Write the failing Nb transport test**

Add to `tests/test_montecarlo.py`:

```python
def test_niobium_transport_parameters_and_fallback():
    params = TRANSPORT_ELEMENTS["Nb"]
    assert params == {"Z": 41, "A": pytest.approx(92.906, abs=0.001), "J_keV": 0.417}

    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=[("Nb", 0.05)],
        seed=123,
        max_steps=2,
    )
    assert segs["Ne"] == 4
    assert len(segs["E_keV"]) > 0
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_form_factors.py::test_niobium_registered_and_edge_prone tests/test_montecarlo.py::test_niobium_transport_parameters_and_fallback -v
```

Expected: both tests fail because Nb is absent from `_EDGE_PRONE` and `TRANSPORT_ELEMENTS`.

- [ ] **Step 4: Add the minimal Nb production data**

Change the edge policy in `src/cxr_mc/materials/crystal.py` to:

```python
# elements whose edges fall in the soft-x-ray band -> force Henke correction
# (Nb M and Te L edges land inside the line grids).
_EDGE_PRONE = {"Si", "Ge", "Mo", "Nb", "Se", "Te"}
```

Add the following row beside Mo in `src/cxr_mc/montecarlo/transport.py`:

```python
"Nb": {"Z": 41, "A": 92.906, "J_keV": 0.417},
```

- [ ] **Step 5: Verify GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_form_factors.py tests/test_montecarlo.py -q
```

Expected: PASS. The Nb-only transport call uses the existing analytic screened-Rutherford fallback when no bundled Nb Mott table exists.

- [ ] **Step 6: Commit Task 1**

```bash
git add tests/test_form_factors.py tests/test_montecarlo.py src/cxr_mc/materials/crystal.py src/cxr_mc/montecarlo/transport.py
git commit -m "feat: add niobium atomic transport support"
```

### Task 2: Explicit 2H-a Crystal Structures and Validation

**Files:**
- Modify: `tests/test_crystallography.py:7-125`
- Modify: `src/cxr_mc/data/crystal_structures.toml:239-267`
- Create: `docs/validation/2ha-niobium-dichalcogenides.md`
- Modify: `docs/physics-validation-ledger.md:33-36`

**Interfaces:**
- Consumes: Nb support from Task 1 and the existing `CRYSTALS`, `structure_factor`, `chi_g`, and `U_g` APIs.
- Produces: `CRYSTALS["nbs2"]` and `CRYSTALS["nbse2"]`, each with a six-atom 2H-a conventional cell.

- [ ] **Step 1: Write failing catalog and structural tests**

Import `U_g` in `tests/test_crystallography.py`, add `"nbs2"` and `"nbse2"` to `EXPECTED`, and add:

```python
@pytest.mark.parametrize(
    ("name", "chalcogen", "a", "c", "z"),
    [
        ("nbs2", "S", 3.320, 11.970, 0.113),
        ("nbse2", "Se", 3.4459, 12.5607, 0.116),
    ],
)
def test_2ha_niobium_dichalcogenide_structure(name, chalcogen, a, c, z):
    info = CRYSTALS[name]
    basis = info["basis"]

    assert info["lattice"]["a"] == pytest.approx(a)
    assert info["lattice"]["c"] == pytest.approx(c)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * a**2 * c / 2.0)
    assert len(basis) == 6
    assert sum(element == "Nb" for element, _ in basis) == 2
    assert sum(element == chalcogen for element, _ in basis) == 4

    nb_positions = sorted(tuple(position) for element, position in basis if element == "Nb")
    assert nb_positions == [(0.0, 0.0, 0.25), (0.0, 0.0, 0.75)]

    expected_chalcogen = {
        tuple(round(value, 6) for value in position)
        for position in (
            (1 / 3, 2 / 3, z),
            (1 / 3, 2 / 3, 0.5 - z),
            (2 / 3, 1 / 3, 0.5 + z),
            (2 / 3, 1 / 3, 1.0 - z),
        )
    }
    actual_chalcogen = {
        tuple(round(float(value), 6) for value in position)
        for element, position in basis
        if element == chalcogen
    }
    assert actual_chalcogen == expected_chalcogen


@pytest.mark.parametrize("name", ["nbs2", "nbse2"])
def test_2ha_niobium_dichalcogenide_couplings_are_finite(name):
    hkl = (1, 0, 0)
    structure, g = structure_factor(name, hkl, 1500.0, B_ang2=0.6)
    susceptibility = chi_g(name, hkl, 1500.0, B_ang2=0.6)
    potential = U_g(name, hkl, 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0
```

- [ ] **Step 2: Run the structural tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_crystallography.py::test_catalog_loads tests/test_crystallography.py::test_2ha_niobium_dichalcogenide_structure tests/test_crystallography.py::test_2ha_niobium_dichalcogenide_couplings_are_finite -v
```

Expected: FAIL with missing `nbs2` and `nbse2` catalog keys.

- [ ] **Step 3: Add the explicit TOML structures**

Append these entries after the existing 2H disulfides in `src/cxr_mc/data/crystal_structures.toml`:

```toml
# Metallic 2H-a NbS2, P6_3/mmc: Nb occupies aligned 2b columns and S occupies
# 4f with z=0.113. Source: Heil et al., Phys. Rev. B 103, 155105 (2021),
# supplemental crystallographic parameters; 2b/4f expansion cross-checked
# against AFLOW AB2_hP6_194_b_f-002. Assumes a stoichiometric, unmodulated bulk
# cell. Limiting checks: two Nb + four S, V=(sqrt(3)/2)a^2c, and Nb(x,y) remains
# aligned across z=1/4 and 3/4. Validation: nbs2-2ha-structure
[nbs2]
system = "hexagonal"
a = 3.320
c = 11.970
basis = [
    { element = "Nb", pos = [0.0, 0.0, 0.25] },
    { element = "S",  pos = [0.3333333333333333, 0.6666666666666667, 0.113] },
    { element = "S",  pos = [0.3333333333333333, 0.6666666666666667, 0.387] },
    { element = "Nb", pos = [0.0, 0.0, 0.75] },
    { element = "S",  pos = [0.6666666666666667, 0.3333333333333333, 0.613] },
    { element = "S",  pos = [0.6666666666666667, 0.3333333333333333, 0.887] },
]

# Metallic 2H-a NbSe2, P6_3/mmc: Nb occupies aligned 2b columns and Se occupies
# 4f with z=0.116 (origin-equivalent to the reported ~0.616 representative).
# Lattice: Yan et al., J. Appl. Phys. 134 (2023), doi:10.1063/5.0172460;
# basis prototype: AFLOW AB2_hP6_194_b_f-002. Assumes the room-temperature,
# unmodulated bulk cell (not the low-temperature CDW supercell). Limiting checks:
# two Nb + four Se, V=(sqrt(3)/2)a^2c, and aligned Nb columns.
# Validation: nbse2-2ha-structure
[nbse2]
system = "hexagonal"
a = 3.4459
c = 12.5607
basis = [
    { element = "Nb", pos = [0.0, 0.0, 0.25] },
    { element = "Se", pos = [0.3333333333333333, 0.6666666666666667, 0.116] },
    { element = "Se", pos = [0.3333333333333333, 0.6666666666666667, 0.384] },
    { element = "Nb", pos = [0.0, 0.0, 0.75] },
    { element = "Se", pos = [0.6666666666666667, 0.3333333333333333, 0.616] },
    { element = "Se", pos = [0.6666666666666667, 0.3333333333333333, 0.884] },
]
```

- [ ] **Step 4: Verify the crystal tests are GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_crystallography.py -q
```

Expected: PASS.

- [ ] **Step 5: Write the validation derivation**

Create `docs/validation/2ha-niobium-dichalcogenides.md` with:

```markdown
# 2H-a niobium dichalcogenide structures

**Validation IDs:** `nbs2-2ha-structure`, `nbse2-2ha-structure`

## Claim and sources

The production `nbs2` and `nbse2` entries use the metallic 2H-a
P6_3/mmc conventional cell. Nb occupies Wyckoff 2b and the chalcogen occupies
4f. NbS2 uses `(a, c, z) = (3.320 A, 11.970 A, 0.113)` from Heil et al.,
Phys. Rev. B 103, 155105 (2021), supplemental material. NbSe2 uses
`(a, c, z) = (3.4459 A, 12.5607 A, 0.116)` from the room-temperature lattice
refinement of Yan et al., J. Appl. Phys. 134 (2023), with the 2b/4f basis
cross-checked against AFLOW prototype `AB2_hP6_194_b_f-002`.

## Explicit basis derivation

The loader does not apply space-group operations, so the conventional basis is
expanded explicitly. The 2b representative gives Nb at `(0,0,1/4)` and
`(0,0,3/4)`. Expanding chalcogen 4f `(1/3,2/3,z)` gives
`(1/3,2/3,z)`, `(1/3,2/3,1/2-z)`, `(2/3,1/3,1/2+z)`, and
`(2/3,1/3,1-z)`. The assumptions are stoichiometric occupancy, the
room-temperature unmodulated bulk cell, and no CDW supercell.

## Checks and limiting case

For a conventional hexagonal cell, `V = (sqrt(3)/2) a^2 c`; production volumes
match this identity. Each cell contains two Nb and four chalcogen atoms, and
representative `F_g`, `chi_g`, and `U_g` values are finite and nonzero. The
stacking discriminator is the aligned Nb limit: both Nb sites have `(x,y)=(0,0)`
at `z=1/4` and `3/4`. This distinguishes metallic 2H-a from the offset 2H-c
metal columns used by the Mo/W catalog entries.

Status remains **unverified** pending an independent crystallography oracle or
human review; this document does not sign off either claim.
```

- [ ] **Step 6: Add validation-ledger rows**

Add after the existing structure rows in `docs/physics-validation-ledger.md`:

```markdown
| `nbs2-2ha-structure` | 2H-a NbS2 P6_3/mmc lattice + explicit 2b/4f conventional-cell basis | `data/crystal_structures.toml::nbs2` | Heil et al., PRB 103, 155105 (2021) supplement; AFLOW `AB2_hP6_194_b_f-002` | unverified | cell volume; 2Nb+4S; aligned Nb columns; finite `F_g`, `chi_g`, `U_g` | `tests/test_crystallography.py::test_2ha_niobium_dichalcogenide_structure`, `::test_2ha_niobium_dichalcogenide_couplings_are_finite` | [validation write-up](validation/2ha-niobium-dichalcogenides.md) |
| `nbse2-2ha-structure` | 2H-a NbSe2 P6_3/mmc lattice + explicit 2b/4f conventional-cell basis | `data/crystal_structures.toml::nbse2` | Yan et al., J. Appl. Phys. 134 (2023); AFLOW `AB2_hP6_194_b_f-002` | unverified | cell volume; 2Nb+4Se; aligned Nb columns; finite `F_g`, `chi_g`, `U_g` | `tests/test_crystallography.py::test_2ha_niobium_dichalcogenide_structure`, `::test_2ha_niobium_dichalcogenide_couplings_are_finite` | room-temperature unmodulated cell; [validation write-up](validation/2ha-niobium-dichalcogenides.md) |
```

- [ ] **Step 7: Check validation markers and formatting**

Run:

```bash
rg -n "Validation: (nbs2|nbse2)-2ha-structure" src/cxr_mc/data/crystal_structures.toml
git diff --check -- src/cxr_mc/data/crystal_structures.toml tests/test_crystallography.py docs/validation/2ha-niobium-dichalcogenides.md docs/physics-validation-ledger.md
```

Expected: two TOML validation markers and no whitespace errors.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/cxr_mc/data/crystal_structures.toml tests/test_crystallography.py docs/validation/2ha-niobium-dichalcogenides.md docs/physics-validation-ledger.md
git commit -m "feat: add 2Ha niobium dichalcogenide structures"
```

### Task 3: Runnable Material Registry Entries

**Files:**
- Modify: `tests/test_sweep.py:22-120`
- Modify: `src/cxr_mc/materials/registry.py:177-218`
- Modify: `README.md:203-211`

**Interfaces:**
- Consumes: `CRYSTALS["nbs2"]`, `CRYSTALS["nbse2"]`, and the existing registry projections.
- Produces: `MATERIAL_CONFIGS`, `MATERIAL_GRIDS`, `CRYSTAL_PARAMS`, `MATERIALS`, and `MATERIAL_LABELS` entries for both keys; `material_sweep()` and `build_cases()` work without special handling.

- [ ] **Step 1: Write failing registry and case tests**

Add `"nbs2"` and `"nbse2"` to `ALL` in `tests/test_sweep.py`, then add:

```python
@pytest.mark.parametrize(
    ("material", "label", "chalcogen"),
    [("nbs2", "NbS2", "S"), ("nbse2", "NbSe2", "Se")],
)
def test_niobium_dichalcogenide_registered_and_runnable(material, label, chalcogen):
    assert MATERIAL_LABELS[material] == label
    assert material in MATERIALS

    grid = material_grid(material)
    assert grid["thickness_ang"] == 10e4
    assert "substrate" not in grid

    params = crystal_params(material)
    composition = dict(params["composition"])
    assert params["beam_uvw"] == (0, 0, 2)
    assert params["hkl_list"]
    assert composition[chalcogen] == pytest.approx(2.0 * composition["Nb"])

    sweep = Sweep(
        material=material,
        thickness_ang=100.0,
        energy_keV=30.0,
        tilt_deg=30.0,
        tilt_azim_deg=0.0,
        E_grid_line=np.arange(500.0, 520.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=1)[0]
    assert case["crystal"] == material
    assert case["composition"] == params["composition"]
```

- [ ] **Step 2: Run the registry tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sweep.py::test_crystal_params_complete tests/test_sweep.py::test_niobium_dichalcogenide_registered_and_runnable -v
```

Expected: FAIL because neither key exists in `MATERIAL_CONFIGS`.

- [ ] **Step 3: Add the two bulk registry rows**

Insert after the existing WSe2 row in `src/cxr_mc/materials/registry.py`:

```python
# Metallic 2H-a niobium dichalcogenides: Nb occupies aligned 2b columns,
# unlike the offset 2H-c Mo/W entries. Nb has no bundled NIST Mott table, so
# transport uses the analytic screened-Rutherford fallback.
"nbs2": {
    "label": "NbS2",
    "B_ang2": 0.6,
    "beam_uvw": (0, 0, 2),
    "E_grid": np.arange(350.0, 2500.0, 3.0),
    "thickness_ang": 10e4,
    "energy_keV": [25, 30, 35],
    "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
    "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
    "E_grid_line": np.arange(50.0, 4500.0, 3.0),
    "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
},
"nbse2": {
    "label": "NbSe2",
    "B_ang2": 0.6,
    "beam_uvw": (0, 0, 2),
    "E_grid": np.arange(350.0, 2500.0, 3.0),
    "thickness_ang": 10e4,
    "energy_keV": [25, 30, 35],
    "tilt_deg": np.linspace(0, 85, 20, endpoint=False),
    "tilt_azim_deg": np.linspace(0, 180, 9, endpoint=True),
    "E_grid_line": np.arange(50.0, 4500.0, 3.0),
    "E_grid_brem": np.arange(0.0, 30000.0, 25.0),
},
```

- [ ] **Step 4: Update the README catalog**

Add this row after the Mo/W 2H rows in `README.md`:

```markdown
| `nbs2`, `nbse2`          | 2H-a-NbS2, 2H-a-NbSe2             | hexagonal (metallic 2H-a TMD) |
```

- [ ] **Step 5: Verify the registry tests are GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sweep.py tests/test_materials_package.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-test one built case per material**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c 'from cxr_mc.sweep import Sweep, build_cases; print([(m, build_cases(Sweep(material=m), 1, 1)[0]["crystal"]) for m in ("nbs2", "nbse2")])'
```

Expected:

```text
[('nbs2', 'nbs2'), ('nbse2', 'nbse2')]
```

- [ ] **Step 7: Commit Task 3**

```bash
git add tests/test_sweep.py src/cxr_mc/materials/registry.py README.md
git commit -m "feat: register NbS2 and NbSe2 scans"
```

### Task 4: Repository Verification

**Files:**
- Modify only files changed by Ruff if they are within Tasks 1-3; do not accept formatting changes to unrelated files.

**Interfaces:**
- Consumes: all implementation from Tasks 1-3.
- Produces: evidence that the focused tests, full suite, lint, format, and type checks pass.

- [ ] **Step 1: Run focused verification**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_form_factors.py tests/test_crystallography.py tests/test_sweep.py tests/test_montecarlo.py tests/test_materials_package.py -q
```

Expected: PASS.

- [ ] **Step 2: Run lint and formatting checks**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check .
```

Expected: PASS. If Ruff reports unrelated dirty files, report them separately and do not rewrite them as part of this feature.

- [ ] **Step 3: Run the type checker**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pyright
```

Expected: zero errors attributable to the feature. Preserve and report any pre-existing errors separately.

- [ ] **Step 4: Run the full test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Expected: PASS.

- [ ] **Step 5: Run the required pre-commit suite and audit scope**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --all-files
git status --short
git diff --check
```

Expected: hooks pass; only intended feature files plus the user's pre-existing unrelated changes appear. Do not stage unrelated files changed by an auto-fixing hook.

- [ ] **Step 6: Commit verification-only formatting if needed**

If and only if Ruff changed a Task 1-3 file after the preceding commits:

```bash
git add src/cxr_mc/materials/crystal.py src/cxr_mc/montecarlo/transport.py src/cxr_mc/data/crystal_structures.toml src/cxr_mc/materials/registry.py tests/test_form_factors.py tests/test_montecarlo.py tests/test_crystallography.py tests/test_sweep.py docs/validation/2ha-niobium-dichalcogenides.md docs/physics-validation-ledger.md README.md
git commit -m "style: format niobium crystal support"
```

Otherwise, make no final commit.
