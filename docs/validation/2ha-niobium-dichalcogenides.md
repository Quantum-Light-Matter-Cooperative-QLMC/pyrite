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
