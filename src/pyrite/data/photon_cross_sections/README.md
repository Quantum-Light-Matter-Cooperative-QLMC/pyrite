# Photon interaction cross sections

`epdl2025_mf23.npz` holds per-atom photon cross sections for Z = 1--100 from
1 eV to 100 GeV. It is **derived** from the 2025 Livermore Evaluated Photon
Data Library (EPDL) by D. E. Cullen, distributed by the IAEA Nuclear Data
Section as part of EPICS2025 (`NDS-IAEA-225`, evaluated August 2023,
distributed January 2025). EPICS distributes the data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the
repository's [third-party notices](../../../../THIRD-PARTY-NOTICES.md).

## Provenance

- Upstream: <https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL>, 85,928,920
  bytes, SHA-256
  `59bbd8c559685dda0bf0de2762bc43126f599cd154d635940f17b6a59c1c43fd`.
- Generator: `scripts/release_epdl_table.py --source EPDL2025.ALL --write`
  (default tolerance `5e-4`), which verifies the upstream SHA-256 first.
- Output SHA-256, pinned in
  `pyrite.materials.photon_cross_sections.EPDL_TABLE_SHA256`:
  `fcc2f00c5bb969e99bc84cac16762f13e939f071d585c433a5c0f420913fcfc9`. The
  archive is written deterministically (sorted members, fixed timestamps), so
  a rerun reproduces it byte for byte.

## Contents

ENDF-6 File 23 sections, cross sections in barns:

| MT | Channel | Key in `PHOTON_CHANNELS` |
|---|---|---|
| 522 | photoionization, all subshells | `photoelectric` |
| 502 | coherent (Rayleigh) scattering | `coherent` |
| 504 | incoherent (Compton) scattering | `incoherent` |
| 517 | pair production, nuclear field | `pair_nuclear` |
| 515 | pair (triplet) production, electron field | `pair_electron` |

MT 501 (total) and MT 516 (pair total) are sums of these and are not stored.
Arrays: `z`, `mt` (one row per table), `offsets` (into the flat arrays),
`energy_eV` (float64), `sigma_barn` (float32), `tolerance`.

## Modifications

Every upstream section is pre-linearized (ENDF interpolation law 2, lin-lin).
Within each run of strictly increasing energies, knots were removed greedily
while lin-lin interpolation through the kept knots reproduces every removed
upstream value to `tolerance` (5e-4) relative. Photoionization edges (repeated
energies) and zero values (pair thresholds, ionization onset) are kept
verbatim. Cross sections are stored as float32 (relative rounding ~6e-8).
Measured against the full upstream MT 501 total, the packaged sum agrees to
5.0e-4 at every upstream node and interval midpoint.

The upstream file is 86 MB; this table is 1.5 MB. Under
[ADR-0014](../../../../docs/adr/0014-packaged-data-layout.md) it is a class (b)
candidate (upstream-derived, over ~1 MiB compressed), packaged for now like
EEDL and EADL until hash-pinned fetching lands.
