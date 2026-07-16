# TODO / Backlog

## Default thickness sweeps + penetration watchdog

Give every `mats_to_sim.toml` material the same default thickness sweep
(100nm/500nm/1um/4um/10um/20um/50um/100um) via `[profiles.standard]`,
including hopg/hbn (previously pinned to a single ~1mm slab) and mos2
(split into a free-standing bulk entry plus a new `mos2-on-sapphire`
device entry that keeps the old 3-layer-on-sapphire config). Before each
`cxr scan` case runs, a penetration watchdog runs a cheap trajectory-only
Monte Carlo (`montecarlo.simulate_trajectories`, the prebuilt
penetration-depth transport) at normal incidence for each (material, beam
energy, thickness); once the transmitted electron fraction drops below
5%, that thickness is kept (so the dying case is represented) but every
thicker one is skipped for that beam energy, so effectively dead cases
are not wastefully re-simulated.
