"""Call-signature adapters for the axis-specialized CPU transport body.

These wrappers only arrange kernel arguments. Numerical steps and all RNG
operations remain in ``cores.make_cpu_transport_core``'s compiled body.
"""

from numba import njit


def validate_core_axes(grooved, per_electron, lut, inelastic, radiative, cone_split, mixed):
    """Refuse axis combinations that have no CPU core specialization."""
    if grooved and (per_electron or lut):
        raise ValueError("grooved transport has only an exact lockstep specialization")
    if grooved and inelastic:
        raise ValueError("shell soft/hard inelastic transport has no grooved specialization")
    if radiative and (grooved or lut):
        raise ValueError("coupled radiative transport has only exact ungrooved specializations")
    if cone_split and (grooved or lut or not per_electron):
        raise ValueError("detector-cone splitting has only exact per-electron specializations")
    if mixed and (grooved or lut or cone_split or not per_electron):
        raise ValueError("mixed elastic transport has only exact per-electron specializations")


def adapt_cpu_transport_core(body, *, per_electron, lut, grooved, cone_split, mixed=False):
    """Expose the existing lockstep/per-electron/LUT/split kernel signatures."""
    if per_electron and lut:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            lut_args,
            materials,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args=(),
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                (),
                lut_args,
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
            )
    elif per_electron and mixed:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args,
            radiative_args,
            mixed_args,
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
                radiative_args,
                None,
                mixed_args,
            )
    elif per_electron and cone_split:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args,
            radiative_args,
            split_args,
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
                radiative_args,
                split_args,
            )
    elif per_electron:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args=(),
            radiative_args=None,
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
                radiative_args,
            )
    elif grooved:

        @njit(cache=True)
        def core(Ne, rng, control, geometry, groove, materials, mott, state, segments, straggling):
            return body(
                Ne,
                rng,
                control,
                geometry,
                groove,
                materials,
                mott,
                (),
                state,
                segments,
                (),
                straggling,
                (),
            )
    elif lut:

        @njit(cache=True)
        def core(
            Ne,
            rng,
            control,
            geometry,
            lut_args,
            materials,
            state,
            segments,
            straggling,
            inelastic_args=(),
        ):
            return body(
                Ne,
                rng,
                control,
                geometry,
                (),
                materials,
                (),
                lut_args,
                state,
                segments,
                (),
                straggling,
                inelastic_args,
            )
    else:

        @njit(cache=True)
        def core(
            Ne,
            rng,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            straggling,
            inelastic_args=(),
            radiative_args=None,
        ):
            return body(
                Ne,
                rng,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                (),
                straggling,
                inelastic_args,
                radiative_args,
            )

    return core
