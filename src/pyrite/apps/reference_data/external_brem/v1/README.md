# Zhai Figure 3b external-background fixture, version 1

These two files preserve one published validation condition: 25 keV electrons
on 1 mm HOPG, detected over 0.066 sr in Zhai et al. Figure 3b. They are
column-only transformations of deposited source files; numeric strings and
units are unchanged.

Source dataset:

- Q. Zhai et al., “Related Data for: Enhanced tunable X-rays from bulk
  crystals driven by table-top free electron energies,” DR-NTU (Data), V1,
  DOI [`10.21979/N9/WZAMZ0`](https://doi.org/10.21979/N9/WZAMZ0).
- Dataset V1 UNF: `UNF:6:nJhYFFasbKoHid6aS6g/PA==`; license: CC BY-NC 4.0.
- `zhai_fig3b_25kev_1mm_brem.csv`: columns 7–8 from Dataverse file 287301,
  `Figure 3b Bremsstrahlung.tab`, source MD5
  `930509a7e99c67a61936f94b7ad0510b`.
- `zhai_fig3b_25kev_1mm_experiment.csv`: columns 13, 15, and 16 from
  Dataverse file 287302, `Figure 3b Experiment.tab`, source MD5
  `68c414f7c14a8d323abdd994d6340152`.

“NIST DTSA-II” identifies software used by Zhai et al.; it does not identify a
single canonical NIST spectrum. Fixture identity includes authors' deposited
dataset version, figure, condition, and source-file checksums. Zhai SI S3 says
experimental bremsstrahlung was subtracted with DTSA-II through a numerical
method citing Clayton et al. (1987), but does not publish enough fit detail for
exact reproduction. PyRITE therefore documents and tests its own scale-only
weighted sideband fit; it does not label that fit as Zhai's exact algorithm.

Normalization is already detected intensity in `Phs/eV/s/nA`. No detector
efficiency, convolution, channel-width, live-time, beam-current, or solid-angle
factor may be reapplied. `load_external_brem` reads the two-column background
fixture directly and interpolates out-of-range energies to zero.
