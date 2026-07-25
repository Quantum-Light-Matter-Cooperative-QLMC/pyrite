# General issues/notes

1) # ! IMPORTANT FOR VALIDATION ! # Confirm no more placeholder debye-waller factors remain in any crystal configs, all are supported by literature. Also, confirm no crystals have significant anisotropy in Debye-Waller (currently only single scalar value supported for each mat)
2) Eventually: add tests comparing local database crystal parameters to external database (use `crystal` library methods with selectable database from their log -- select most reputable/complete for default). Scoped design + implementation path: `docs/crystal-db-comparison.md` (COD default, geometry-only, does NOT cover #1 Debye-Waller).
3) add flags to `rebrem` and `reline` that let you set new default values for each (step, start, stop, for all or per material)
4) Confirm that if user types `cxr pull --brem-only <mat1> <mat2>`, it will pull brem-only for BOTH materials (or full set of materials)
5) split up material pickles into individual line and brem pickles