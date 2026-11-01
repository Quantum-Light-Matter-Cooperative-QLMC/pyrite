"""Explicit CUDA first-flight reference for the optional photon tracker (#341).

This standalone boundary accepts host arrays and returns CuPy arrays. It does
not select a production backend, generate random draws, or track descendants.
CUDA is selected explicitly at the call boundary; the enclosing transport
package retains its existing import behavior.
"""

from collections.abc import Sequence
from numbers import Integral

from ..geometry import validate_transverse_dimensions
from . import photon_navigation as nav


def scene_first_interactions_cuda(
    origins,
    directions,
    energies_eV,
    uniforms,
    layers: Sequence,
    boxes: Sequence[nav.PhotonBox] = (),
    *,
    width_ang=None,
    height_ang=None,
    implementation="fused",
    threads=128,
):
    """Sample stack/prism/box first flights on CUDA, with explicit host draws.

    Inputs, channel/element order, units and escape sentinels match
    :func:`photon_navigation.scene_first_interactions`. All four returned
    arrays reside on the current CUDA device: binary64 distances [Å] and
    int64 region/channel/element indices. CUDA availability is required even
    for an empty batch; no CPU fallback is allowed. Validate disjoint regions
    with :func:`photon_navigation.validate_photon_regions` before repeated calls.

    EPDL evaluation, input validation and ``-log1p(-u0)`` conversion use the
    host reference. ``implementation="fused"`` uses two binary64 kernels for
    ray clipping and stable interval/channel selection, evaluating each unique
    composition's rates once per call. ``implementation="array"`` retains the
    shared NumPy/CuPy reference. ``threads`` is the fused launch block size
    (32, 64, 128, 256 or 512); it does not change draws or physical parameters.
    Ray clipping, optical-depth inversion and channel selection run on device. Working
    storage scales as rays times regions and material channels; callers bound
    memory by splitting the batch, retaining each ray's original two draws.
    Reordering/chunking consume no RNG and preserve the represented histories.
    Floating-point reductions may differ from NumPy; bitwise CPU parity is
    not promised at numerically ambiguous CDF or geometric boundaries.

    Source, assumptions and limits: the unchanged piecewise homogeneous,
    fixed-energy first-flight law in ``photon_navigation``. Zero rate gives
    escape and vacuum gaps contribute no optical depth.
    Validation: photon-channel-first-interaction
    Validation: photon-scene-navigation
    """
    origins, directions, energies, uniforms = nav._validate_flights(
        origins, directions, energies_eV, uniforms
    )
    nav._validate_layers(layers)
    width_ang, height_ang = validate_transverse_dimensions(width_ang, height_ang, unit="Ang")
    if implementation not in ("array", "fused"):
        raise ValueError("implementation must be 'array' or 'fused'")
    if (
        isinstance(threads, bool)
        or not isinstance(threads, Integral)
        or threads not in (32, 64, 128, 256, 512)
    ):
        raise ValueError("threads must be one of 32, 64, 128, 256 or 512")

    from ..._backend import select_cupy_backend

    cp = select_cupy_backend().xp
    if implementation == "fused":
        from ._photon_cuda_kernel import fused_first_interactions

        return fused_first_interactions(
            cp,
            origins,
            directions,
            energies,
            uniforms,
            layers,
            boxes,
            width_ang,
            height_ang,
            threads=int(threads),
        )
    device_origins, device_directions = cp.asarray(origins), cp.asarray(directions)
    start, end = nav._layer_intervals(
        device_origins, device_directions, layers, width_ang, height_ang, xp=cp
    )
    if boxes:
        box_start, box_end = nav._box_intervals(device_origins, device_directions, boxes, xp=cp)
        start, end = cp.hstack((start, box_start)), cp.hstack((end, box_end))
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    return nav._interval_first_interactions(start, end, compositions, energies, uniforms, xp=cp)
