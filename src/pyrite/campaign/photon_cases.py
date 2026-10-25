"""Lower opt-in physical photon observations into replayable Case records."""

from dataclasses import replace


def resolve_photon_transport(scene, case, numerics=None):
    """Leave mode-off Cases untouched; pin enabled geometry, models and data."""
    selection = scene.physics.photon_transport
    if selection is None:
        return case
    from ..montecarlo.runner.case_tables import case_table_markers
    from ..montecarlo.runner.photon_aperture import photon_case_record, prepare_source
    from ..photon_transport import _preflight_scene, _scene_geometry
    from .profiles import case_content_key

    if numerics is not None:
        scene = replace(scene, numerics=numerics)
    detector, numerics = _preflight_scene(scene)
    _, aperture, boxes, (width, height) = _scene_geometry(case, detector, scene.filter_plates)
    context = prepare_source(
        case, selection, aperture, boxes, width, height, numerics.transport_core
    )
    source_digest = case_content_key(case, xsgen_tables=case_table_markers(case))
    return replace(case, photon_transport=photon_case_record(case, source_digest, context))
