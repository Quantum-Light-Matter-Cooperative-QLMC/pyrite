# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.33.0",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    _ = os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt

    import pyrite

    with mo.status.spinner(title="Preparing PyRITE reference tables, warming JIT caches"):
        _ = pyrite.fetch_tables()  # Fetch data tables
        _ = pyrite.execution.warm_cache()
    tables_ready = True
    return mo, plt, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make Some Radiation

    Set up PyRITE, simulate a small crystalline target, and plot its X-ray spectrum.

    ## Environment and reference tables

    Locally, install the project environment and open this file with
    `uv run marimo edit docs/notebooks/01-first-spectrum.py`.
    In Molab, the inline dependencies provide the environment. Use its
    **built-in package manager** to add packages; if table loading reports a
    missing MF=26/MT=527 photon yield, enter `endf-parserpy==0.17.0` and restart.

    The first cell calls `pyrite.fetch_tables()` to install about 100 MB of
    pinned interaction data once; subsequent runs reuse the installed tables.
    Start on CPU. For NVIDIA execution, add `pyrite-mc[nvidia]`, select an
    available GPU in Molab (or use your local CUDA GPU), and set
    `PYRITE_MC_BACKEND` to `cuda` **before importing PyRITE**. Restart the
    kernel after changing the backend or installing CuPy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Crystals and Reciprocal Lattice Vectors

    In general, crystals are defined by planes of symmetry known as lattice planes.
    Lattice planes are generally defined by their [Miller indices](https://en.wikipedia.org/wiki/Miller_index) $(hkl)$,
    which describe their orientation in the unit cell. Taking the spatial Fourier transform of the lattice leads
    to the concept of a 'reciprocal lattice vector' $\mathbf{g}_{hkl}$, which is orthogonal in real space to its
    associated $(hkl)$ lattice plane. Its magnitude is related to the plane spacing by

    $$
    d = \frac{2 \pi}{|\mathbf{g}_{hkl}|}
    $$

    where $d$ is the spacing between two parallel $(hkl)$ lattice planes.

    Parametric X-ray Radiation (PXR) and Coherent Bremsstrahlung (CBS) are both generated through
    interactions which are fundamentally mediated by these reciprocal lattice vectors, as the lattice
    vectors define the momentum exchange which can occur between electrons and atoms in the lattice.
    For a given reciprocal lattice vector $\mathbf{g}_{hkl}$ and an incident electron with velocity $\mathbf{v}$,
    at an emission (or "observation") angle $\hat{n}$, both PXR and CBS produce photons with energy

    $$
    E_p = \hbar \frac{\mathbf{v} \cdot \mathbf{g}_{hkl}}{1 - (\mathbf{v} \cdot \hat{n})/c}
    $$

    A lattice which can exchange momentum $\mathbf{g}_{hkl}$ is also capable of exchanging $n\mathbf{g}_{hkl}$,
    meaning that photons can also be emitted at $nE_p$ for integer $n$.

    While an ideal crystal has an infinite number of reciprocal lattice vectors, not all contribute
    equally to X-ray emission. The strength of each contribution depends in part on the crystal's
    periodic structure, described by the Fourier coefficients of the X-ray dielectric susceptibility
    $\chi_{\mathbf{g}}$ for PXR, and the screened electrostatic crystal potential $U_{\mathbf{g}}$ for CBS.
    These coefficients depend on the arrangement of atoms within the unit cell: some reciprocal lattice vectors
    produce strong radiation amplitudes, while others are suppressed or even forbidden by destructive interference.

    Layered materials such as graphite, hexagonal boron nitride (hBN), and transition metal dichalcogenides (TMDs)
    are particularly interesting because of their pronounced structural anisotropy. In highly oriented pyrolytic graphite (HOPG),
    for example, the crystallites are preferentially aligned along the crystallographic $c$-axis (perpendicular to
    the basal planes), even though their in-plane orientations may vary. This makes reflections from the basal
    $\{00l\}$ planes particularly useful for generating well-defined X-ray spectral lines. Moreover, graphite's
    ABAB stacking produces systematic extinctions: the contributions from successive layers cancel for odd $l$,
    leaving $(002)$, $(004)$, etc., as the only allowed basal reflections.

    The resulting spectral intensity depends not only on the magnitudes of $\chi_{\mathbf{g}}$ and $U_{\mathbf{g}}$,
    but also on the emission geometry and interference between PXR and CBS. The linewidth additionally depends on
    factors such as crystal thickness, mosaic spread, electron-beam properties, and detector acceptance.
    These properties make suitably oriented layered crystals attractive candidates for tunable, narrow-linewidth soft-X-ray sources.

    For this introductory example, we use HOPG and retain only two automatically selected reflection families to keep the simulation inexpensive.

    This example uses 30 keV electrons incident upon a 400 nm HOPG slab.
    It's important to note that the sampled electron histories do *not* represent
    a physical number of electrons in a bunch -- rather, they are simply the number
    of Monte Carlo trials that are simulated to produce these plots. The more histories,
    the less noisy the spectrum, with the statistical noise generally going down as $1/\sqrt{N}$.

    We use Pint quantities from `pyrite.units.ureg` for dimensional inputs,
    as we just learned in [Units and Quantities](https://pyrite.readthedocs.io/en/latest/notebooks/fundamentals/00a-units-and-quantities.html).
    """)
    return


@app.cell
def _(mo, tables_ready):
    assert tables_ready
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector
    from pyrite.geometry import Slab
    from pyrite.units import ureg

    # First, we define our electron beam:
    beam = Beam(
        kinetic_energy=30 * ureg.keV,  # 30 keV beam energy
        transverse=PointSpot(),  # point-like bunch (nonrealistic, of course),
        longitudinal=PointBunch(),
        repetition_rate=5.0 * ureg.kHz,
        bunch_charge=1.0 * ureg.pC,
    )

    # Next, our crystalline target: a 400 nm thick slab of Highly Oriented Pyrolytic Graphite
    target = Slab(
        "hopg",
        thickness=400 * ureg.nm,
        tilt=30 * ureg.degree,
        width=5 * ureg.mm,
        height=5 * ureg.mm,
    )

    # Our detector: in this case, an ideal point-like detector sampling per steradian
    detector = Detector(observation_angle=135 * ureg.degree)

    # Numerical settings for the simulation: how many trials to score for PXR/CBS spectra,
    # and how many for the brem spectrum. Note that these spectra reuse the same electron trajectories,
    # so the total number of transported electrons is simply MAX(cxr_histories, brem_histories)
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=32_768, brem_histories=128),
        cxr_reflections=CxrReflections(
            n_families=2
        ),  # Here we only consider the first two basal-plane reflections
    )

    with mo.status.spinner(title="Simulating electron transport and X-ray spectrum"):
        result = simulate(beam, target, detector, numerics=numerics)

    cxr = result.cxr.incoherent
    brem = result.brem
    characteristic = result.characteristic
    return beam, brem, characteristic, cxr, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(brem, characteristic, cxr, plt):
    _figure, _axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    _axis.plot(cxr.photon_energy_eV / 1_000, cxr.density, label="PXR + CBS", color="blue")
    _axis.plot(brem.photon_energy_eV / 1_000, brem.density, label="Bremsstrahlung", color="orange")
    _axis.plot(
        characteristic.photon_energy_eV / 1_000,
        characteristic.density,
        label="Characteristic Radiation",
        color="green",
    )
    _axis.set(
        xlabel="Photon energy (keV)",
        ylabel=r"Photons / (electron $\cdot$ eV $\cdot$ sr)",
        title=r"30 keV electrons $\rightarrow$ 400 nm HOPG",
        xlim=(0.1, 3.7),
        yscale="log",
        ylim=(1e-14, 1e-7),
    )
    _axis.grid(alpha=0.25)
    _axis.legend()
    _figure  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting the spectrum

    In blue, we see our desired result! The pair of spectral lines corresponding to $(002)$ and $(004)$, with the second line looking to be ~2x the photon energy of the first.

    In green, we see the *extremely* dominant characteristic radiation line from the carbon atoms in the HOPG, at ~277 eV. Luckily, these
    characteristic X-rays are absorbed by a slight breeze (literally), so they shouldn't be too hard to filter out, leaving our
    soft-X-ray lines of interest.

    Lastly, in orange, we see our incoherent bremsstrahlung background, and, interestingly, we notice that it has a sharp drop right at the same energy
    as the characteristic line. This is because the absorption $K$-edge of carbon (i.e. the photon energy required to strip an inner-shell electron
    from the atom) is directly tied to the characteristic X-ray emission line, as this line is caused by excited conduction-band electrons decaying down
    into empty inner-shell states and emitting photons of very similar energy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now, together, and linear

    Plot components on their own native grids; they need no interpolation to
    share axes. Convert energy and density together with `to_energy("keV")`.
    The explicit mean current is bunch charge times repetition rate. Here we
    also acknowledge uniform directional density over the scalar detector's
    matching acceptance. The resulting curves are expected accepted photon
    rates before detector response or exposure.
    """)
    return


@app.cell
def _(beam, plt, result):
    from pyrite.units import Q_

    mean_current = Q_(beam.bunch_charge_pc, "pC") * Q_(beam.rep_rate_hz, "Hz")
    accepted_components = result.accepted_spectra(assume_uniform=True)
    _figure, _axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for _name, _component in accepted_components.items():
        _rate = _component.rate(current=mean_current).to_energy("keV")
        _energy = _rate.photon_energy.magnitude
        _visible = (_energy >= 0.3) & (_energy <= 3.7)
        _axis.plot(_energy[_visible], _rate.density.magnitude[_visible], label=_name)
    _axis.set(
        xlabel="Photon energy (keV)",
        ylabel=r"Photons / (s $\cdot$ keV)",
        title=r"30 keV electrons $\rightarrow$ 400 nm HOPG",
        xlim=(0.3, 3.7),
        yscale="linear",
    )
    _axis.set_ylim(bottom=0)
    _axis.grid(alpha=0.25)
    _axis.legend()
    _figure  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    PXR and CBS are combined here because their amplitudes interfere. The electron bunch duration
    is, however, generally far longer than the wavelength of interest, meaning the interference
    is generally between many incoherent emitters which averages to the spectrum above.
    The last plot shows separate accepted components, each on its native grid.
    For an integrated total, `result.integrated_yield()` integrates each original
    component according to its node/bin-mean sampling before adding yields.
    Apply current and matching acceptance to those represented integrals; report
    uncovered windows rather than assuming missing emission is zero. Never sum
    alternative coherent and incoherent CXR modes.

    Try halving or doubling the history counts; see what happens to the spectrum, especially the CXR spectrum
    in the log plot. Then, change only the target thickness, or just the observation angle -- watch how the lines change.

    Keep a record of the resolved seed when comparing
    configurations; different backends need not produce identical random trajectories. Continue to notebook 02
    to dive deeper into dissecting simulation results. Notebook 03 discusses how to predict CXR line
    energies; the [results how-to](https://pyrite.readthedocs.io/en/latest/guides/working-with-results.html)
    covers saved checkpoints, to keep from rerunning expensive simulations over and over. See
    [model limitations](https://pyrite.readthedocs.io/en/latest/physics/capabilities.html)
    and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html)
    before interpreting these small-sample results quantitatively.
    """)
    return


if __name__ == "__main__":
    app.run()
