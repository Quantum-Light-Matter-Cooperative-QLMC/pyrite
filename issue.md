# PyRITE 0.24.1

Changes since `v0.24.0` (30 commits).

## Features

- **beams**: #840 deprecate BunchLength for LongitudinalDistribution gaussian parity (184f4bb88)
- rewrite tutorial 17 (af91cc493)
- **units**: opt-in quantity interfaces for materials, detectors and channeling states (#816) (#848) (7c9a2f3e7)
- **checks**: matched-bounds Geant4 SS vs PyRITE CUDA transport + brem benchmark (#824) (ecee1ae71)
- add new api to api-surface.toml, regenerate notebooks (703329d34)
- **tutorials**: add notebook detailing persistent sweep saving thru API (a3fd73ef7)
- add units and quantities tutorial (05e77acd4)
- expose temporal plots thru SPI, add coherent phase + temporal tracking tutorial (b262900ab)
- **spectrum**: #819 block continuum scoring over the photon grid (#829) (9ebe368af)
- **optimization**: phased CUDA tuning, phase API and simulation progress (#776) (8b508b7b2)
- **tutorials**: add provisional MeV notebook (4937b37e8)
- **units**: app export unit metadata; close #755 inventory (a9a524cac)
- move tutorials onto ureg, then regenerate notebooks (5b2610041)

## Fixes

- **transport**: #807 host-memory admission for retained transport output (#846) (ab35a81bb)
- **ci**: register new public surface, kernel checks and tutorial lint (#839) (12fc415b0)
- add '_' to variables rederined across cells (ade1e622e)
- **campaign**: keep explicit uniform brem grids exact (00f659eab)
- **docs**: restore tutorial lint, marimo check and fresh captures (9888c0119)
- **optimization**: restore kernel_tuning static gates (e376b6783)

## Performance

- **checks**: #803 energy-LUT attribution harness and go/no-go (#822) (41af80bfc)
- **transport**: #841 exact CPU lazy cutoff and inlined moment lookup (#843) (6b5bbf6ca)
- **transport**: #842 default to bounded device staging for continuation (9718679b4)
- **checks**: #831 FP32 speed-of-light for the exact CUDA transport kernel (#836) (d5bcd9572)
- **transport**: #796 register cap and read-only loads for the exact CUDA kernel (#835) (d8597cf6e)
- **spectrum**: stage BremsLib nominal nodes once per table (#832) (#834) (fb70dd9a2)
- **transport**: compile the exact CUDA kernel per physics configuration (#827) (#830) (6613e7392)
- **transport**: exact guide-table ELSEPA panel search (#826) (#828) (f9d84ed62)

## Documentation

- **validation**: anchor thick-target photon full-track claim (#823) (522ae5f3d)

## Chores

- **deps**: bump tornado from 6.5.10 to 6.6 (d9232f92f)