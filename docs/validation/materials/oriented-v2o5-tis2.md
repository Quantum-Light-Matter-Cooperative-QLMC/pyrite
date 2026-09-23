# Oriented V2O5 and 1T-TiS2 structures

**Validation IDs:** `v2o5-010-structure`, `tis2-003-structure`

## Claim and sources

`v2o5` represents stoichiometric ambient alpha-V2O5 in standard Pmmn with `(a,b,c)=(11.512,3.564,4.368)` Angstrom. The lattice source is McColl, Johnson, and Cora, *Phys. Chem. Chem. Phys.* **20**, 15002 (2018), doi:10.1039/C8CP02187B; fractional coordinates are from Sipr et al., *Phys. Rev. B* **60**, 14115 (1999), doi:10.1103/PhysRevB.60.14115.

`tis2` represents stoichiometric ambient 1T-TiS2 in P-3m1 with `a=3.407` Angstrom, `c=5.695` Angstrom, and sulfur `z=0.2493`, from Brown et al., *Phys. Rev. B* **57**, 5101 (1998), doi:10.1103/PhysRevB.57.5101.

## Explicit basis and orientation derivation

The V2O5 Pmmn 4f sites are expanded explicitly because the TOML loader does not apply space-group symmetry; the resulting conventional cell contains 4 V and 10 O atoms. The literature's historical Pmnm `(010)` layered-cut index maps to standard-Pmmn `(001)`, so the V2O5 row pins that `[001]` normal and `±(001)` reflection. The standard-Pmmn `(010)` extinction applies to a distinct physical direction. The 1T-TiS2 cell contains Ti on 1a and S on the symmetry-paired 2d sites, so it contains 1 Ti and 2 S atoms; its row pins `[001]`/`±(003)`. Each pair places the selected reciprocal vector parallel to the slab normal, avoiding an unmodeled asymmetric-reflection miscut.

## Checks and limiting case

V2O5 obeys `V_cell=a*b*c`; TiS2 obeys `V_cell=(sqrt(3)/2)a^2*c`. The catalog tests assert those volumes and stoichiometries, while finite, nonzero `F_g`, `chi_g`, and `U_g` at `(001)`/`(003)` prove the requested reflection can enter the existing coherent-spectrum pipeline. Registry tests assert that the pinned lists survive a `n_families=999` request, the limiting case for bypassing automatic reflection selection.

Status remains **unverified**: these are crystallographic-data and registry checks, not an experimental PXR/CBS validation or a human sign-off.
