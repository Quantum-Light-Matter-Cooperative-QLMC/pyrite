# Diamond valence oscillator

Validation: `diamond-valence-oscillator`.

## Independent source reconstruction

This section was written in a fresh verifier context before inspecting the
implementation or the new data row. The intended interface is
`catalog_shell_oscillators('diamond') -> MaterialShellOscillators`; the claim is
a material input to the existing PENELOPE shell model.

{cite:t}`bhattacharya2025`
reports an approximately 34 eV volume plasmon in both undoped and boron-doped
diamond in the Results subsection on experimental dielectric response, referring
to Supplementary Fig. 2. The undoped control uses HPHT diamond particles.
The Methods describe the synthesis of both controls and doped samples.
The paper attributes this peak to collective valence-to-conduction excitations.
Its separate sub-0.5 eV intervalence response occurs only with boron doping.

Neutral carbon has six electrons, partitioned into a two-electron K core and
four valence electrons. Formula C contains one carbon atom per formula unit, so
the independently expected effective partition is

$$
f_{\mathrm{val}}=4,\qquad f_{\mathrm{K}}=2,\qquad
f_{\mathrm{val}}+f_{\mathrm{K}}=6.
$$

The valence effective resonance energy is

$$
W_{\mathrm{val}}=\hbar\omega_{\mathrm{val}}\simeq34\;\mathrm{eV}.
$$

The four-electron count comes from carbon electron bookkeeping, not a carrier
density measured by the cited paper. The energy is an approximate observed loss
peak, not a fitted dielectric spectrum, damping width, band gap, atomic binding
energy, or a frequency in inverse seconds. With positive oscillator strengths,
the partition preserves the six-electron sum; removing the collective allocation
must restore the four outer electrons to the ordinary atomic-shell bookkeeping.

This is a single effective valence oscillator in the existing model. It does
not assert a nonzero equilibrium conduction-electron population in intrinsic
diamond. The paper describes intrinsic diamond as effectively carrier-free at
room temperature because of its roughly 5.5 eV band gap. No boron hole density
or low-energy intervalence peak belongs in this undoped material input.

Cheap filters: electron counts are dimensionless per formula unit; the positive
resonance energy is in eV; the six-electron sum and the zero-collective-allocation
limit are consistent. No sign or normalization ambiguity is present at the
input level. Applicability is an effective bulk valence-loss representation,
not a derivation of all dielectric or threshold behavior of diamond.

## Comparison with implementation

After the reconstruction above, the verifier inspected
`src/pyrite/data/conduction_band.toml::diamond`,
`shell_rates.py::catalog_shell_oscillators`, and
`shell_oscillators.py::build_shell_oscillators` in
`src/pyrite/montecarlo/transport/`. The data row has formula C, four electrons
per formula unit and resonance 34.0 eV, with the correct DOI and explicit
undoped-control applicability. No divergent count, energy unit, or excitation
assignment was found.

The constructor consumes the carbon L2 and L1 atomic shells and leaves the
two-electron K core. The public interface returned one oscillator labelled
`cb` with strength 4, zero binding energy and resonance 34 eV, plus a K
oscillator with strength 2, binding energy 288 eV and resonance
410.5121107266437 eV. The catalog mean excitation energy is 78 eV and the
all-electron plasma energy is 38.190890028659695 eV. The measured 34 eV input
is used directly; it is not replaced by a free-electron-density estimate.

An independent algebraic reduction of the existing logarithmic moment
constraint gives, with all energies expressed in eV,

$$
6\ln I=4\ln W_{\mathrm{val}}+2\ln W_{\mathrm K},\qquad
W_{\mathrm K}=\frac{I^3}{W_{\mathrm{val}}^2}
=\frac{78^3}{34^2}\;\mathrm{eV}
=410.5121107266436\;\mathrm{eV}.
$$

This matches the returned K resonance to floating-point precision and exceeds
the K binding energy; the valence resonance is positive and below the mean
excitation energy. The total strength is exactly six. This closure check
validates integration of the new input into the existing model, not the
physical accuracy of its full generalized oscillator strength.

## Verdict and remaining limits

**Verdict: `rederived` for the material-input claim.** Units, electron sum,
signs, source applicability and the model moment constraint pass. Suggested
ledger change: add `diamond-valence-oscillator` at `rederived`, anchored on the
diamond TOML record, link this write-up, and name any separately passing
regression test before promoting to `anchored`. Only a human may sign off.

The `cb` oscillator's zero binding energy is an inherited effective-model
convention, not a statement that undoped diamond is metallic. In particular,
the model does not reproduce the insulating band-gap onset, measured loss
width, momentum-dependent loss spectrum, boron-doped intervalence mode, or
temperature dependence. The source's approximately 34 eV report does not
justify sub-eV precision or an uncertainty bound. Electronic channeling rates
computed with this input still require independent experimental validation.

The numerical check ran with the project runner and `PYRITE_MC_BACKEND=cpu`.
The integration additionally pins the source input and carbon electron
partition in `test_diamond_valence_oscillator_keeps_two_core_electrons` and
checks the production kernel in `test_diamond_uses_its_own_valence_oscillator`.
The ledger's `anchored` input status includes these passing regressions;
the independent derivation verdict remains `rederived`.
