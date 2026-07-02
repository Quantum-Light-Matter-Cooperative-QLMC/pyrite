# TODO: diffpy.structure structural importer

Implement TODO P2 #1 using `diffpy.structure` rather than `crystals`: add a thin importer/adapter that converts a `diffpy.structure.Structure` into the existing `cxr_mc.crystallography.CRYSTALS` shape, leaving the X-ray physics (`structure_factor`, `chi_g`, `U_g`) in `cxr_mc`.

Keep `src/cxr_mc/data/crystal_structures.toml` as the canonical fallback and use the adapter first for explicit CIF/Structure import workflows. First slice: test and implement conversion of lattice parameters, fractional coordinates, and composition preservation without changing downstream physics APIs.
