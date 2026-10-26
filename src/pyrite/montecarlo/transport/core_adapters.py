"""Call-signature adapters for the axis-specialized CPU transport body.

These wrappers only arrange kernel arguments. Numerical steps and all RNG
operations remain in ``cores.make_cpu_transport_core``'s compiled body.
"""

from numba import njit


def adapt_cpu_transport_core(body, *, per_electron, lut, grooved, cone_split):
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
