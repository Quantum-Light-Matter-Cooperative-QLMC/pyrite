# SBETHE atomic shell inputs

Validation: `sbethe-atomic-shell-inputs`. Status: unverified. The parsed
inputs are host-side preparation and do not select a transport mode.

## Source and assumptions

The publicly posted [SBETHE v2 dataset](https://data.mendeley.com/datasets/7zw25f428t/2)
supplies `sdbase/pdatconf.p14`, which lists each free-atom shell's
designator, label, orbital, occupation, ionization energy, Compton
profile, and two width columns. Its header attributes ionization energies to
Carlson (1975), profiles to Salvat's MCDF calculation, and widths to EADL
and Campbell–Papp. The default loader requires SHA-256
`cd239554bb6e823692ea4611d443df8684b4cace06006fc271a4168cb78c62d2`.
The source is installed by `pyrite tables fetch sbethe`, not redistributed in
the package. The dataset page lists CC BY-NC 3.0. The PENELOPE-2024 manual
describes a file with this name and role, but we have not compared the SBETHE
file byte-for-byte with the separately licensed NEA PENELOPE distribution.

For each atom, the input occupation check is

$$
\sum_s f_{Z,s}=Z.
$$

The parser requires unique labels whose $n,l$ agrees with the listed
orbital, positive ionization energies, nonnegative profile and
width values, unique designators within each element, and finite numeric
columns. The free-atom occupations are not a material conduction-band
assignment. The material mean excitation energy and all-electron plasma
energy are separate inputs; neither determines a conduction partition.

## EEDL label join

EEDL MF=23/MT=534–572 vacancy channels use the ENDF-6 subshell order,
designator $d=\mathrm{MT}-533$: K, L1–L3, M1–M5, N1–N7, O1–O9, P1–P11,
Q1–Q3. `pdatconf.p14` numbers only the PENELOPE shells K … O7, P1–P4, Q1,
so the two integer codes agree through O7 and diverge after it (ENDF P1 is
26, PENELOPE P1 is 24). The host join therefore matches x-ray labels, never
raw integers. A raw-integer join would silently mislabel shells from Cs
onward.

With labels, every EEDL channel in the fetched source has an SBETHE label
match except open spin-orbit partners. For example, SBETHE places both Si
3p electrons in M2 (3p1/2), while EEDL has M2 and M3 at the same 8.15 eV.
Such an EEDL subshell joins the unique SBETHE shell with the same $n,l$.
Summing the partner rates gives the total $n,l$ ionization rate,

$$
\sigma_{n\ell}=\sum_j \sigma_{n\ell j},
$$

and the join keeps every EEDL label and binding energy. The merged rate
carries EEDL's $n,l$ occupancy, not SBETHE's $j$ assignment. An EEDL shell whose
$n,l$ is absent from SBETHE raises. All 99 elements join; 41 shells merge
partners, all in open valence or near-valence subshells. Their EEDL − SBETHE
binding differences are within 3.4 eV (largest: Ga N3). Exact-label
differences are larger in places, up to 17.9 eV (Mn L1) and 5–11 eV for some
heavy-element P shells. Where the two ground-state
configurations disagree (e.g. Pd 5s, lanthanide 5d versus 4f), the SBETHE
shell receives no EEDL rate.

The partner merge assumes that the valence $j$ label does not select a
relaxation path. That holds for the shells merged here; an inner-shell
vacancy consumer (#94) must still use the EEDL label. Equal labels do not
validate binding energies or a transfer spectrum.

Fixture checks cover occupation totals, duplicate designators and labels,
label–orbital disagreement, invalid values, the
signed binding difference, a spin-orbit partner merge, and ENDF P1 at
designator 26. An optional local-source check joins all 99 elements, including
Si M2+M3 and Cs P1, when the pinned reference data are fetched.
Independent source-to-code verification remains pending.

Fresh-context re-derivation, including the ENDF-102 Appendix B designator
table: [verification record](sbethe-atomic-shell-inputs-verification.md).
