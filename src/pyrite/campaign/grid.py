from .config import default_settings, material_sweep
from .sweep import build_cases


def grid_names(material, fidelity="full", catalog_profile="standard"):
    """Config names in the CURRENT grid for ``material`` -- exactly the set
    ``config.material_sweep(material, catalog_profile=catalog_profile)`` ->
    ``sweep.build_cases`` produces now. A stale config is any name NOT in this
    set. ``catalog_profile`` selects the named materials.toml profile whose
    grid a profile-variant checkpoint was swept on.

    The ``config`` / ``sweep`` imports are function-local on purpose: ``config``
    imports ``results`` at module load (``from .results import Settings``), so a
    module-level ``config`` import here would close an import cycle. Deferring it
    to call time breaks the cycle.
    """
    settings = default_settings(fidelity)
    sweep = material_sweep(material, fidelity=fidelity, catalog_profile=catalog_profile)
    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    return {c["name"] for c in cases}
