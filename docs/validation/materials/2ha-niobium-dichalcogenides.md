# 2H-a niobium dichalcogenide structures

**Validation IDs:** `nbs2-2ha-structure`, `nbse2-2ha-structure`

## Claim and sources

The production `nbs2` and `nbse2` entries use the metallic 2H-a
P6_3/mmc conventional cell. Nb occupies Wyckoff 2b and the chalcogen occupies
4f.

For NbS2, `(a, c) = (3.320 A, 11.970 A)` are the experimental lattice
parameters supplied with the crystals and reported in the supplement to El
Youbi et al., Phys. Rev. B 103, 155105 (2021),
doi:10.1103/PhysRevB.103.155105. That supplement separately states that
`z = 0.113` was adopted from Heil, Schlipf, and Giustino, Phys. Rev. B 98,
075120 (2018), doi:10.1103/PhysRevB.98.075120; it is not a crystallographic
refinement by El Youbi et al.

For NbSe2, `(a, c) = (3.4459 A, 12.5607 A)` are the refined room-temperature
lattice constants reported by Wang et al., Appl. Phys. Lett. 123, 153505
(2023), doi:10.1063/5.0172460. The adopted `z = 0.116` is the lower endpoint
of the experimental `0.116-0.118` range summarized by Johannes, Mazin, and
Howells, Phys. Rev. B 73, 205102 (2006),
doi:10.1103/PhysRevB.73.205102, from primary crystallographic reports including
Brown and Beerntsen, Acta Cryst. 18, 31-36 (1965),
doi:10.1107/S0365110X65000063, and Marezio et al., J. Solid State Chem. 4,
425-429 (1972), doi:10.1016/0022-4596(72)90158-2. The 2b/4f basis for both
materials is cross-checked against AFLOW prototype `AB2_hP6_194_b_f-002`.

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
