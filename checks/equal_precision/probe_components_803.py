"""Component amplification probe for #803: marginal cost of each transport lookup.

Generalizes the #792 flight-rate probe. For one named component, an isolated
copy of the CPU (``cores.py``) or CUDA (``_jit_kernel.py``) transport source
re-evaluates that component ``K`` extra times at its own call site, with
energies (and, for angles, the uniform) perturbed downward (inside table coverage) by ``(k + 1) * 1e-13``
relative so the compiler cannot reuse the first evaluation. The extra results
only feed a sink that compares their sum with an impossible value
(``SINK``), so trajectories stay bitwise identical to ``K = 0``: the probe
refuses a changed trajectory digest. The run itself is ``baseline_803.py``, so
the record keeps the warm wall, NVTX stage split and, on CUDA, CUDA-event
kernel time.

The time slope over ``K`` is the cost of one evaluation of the component at
every execution of its site: the most a zero-cost replacement could save
there. It is a marginal estimate. Extra work can change register allocation,
cache residency and (on CUDA) issue contention, so the sum of components is
not the kernel time; the remainder is reported as unattributed. No production
module imports this probe.

Components (one site each; the site's own control flow decides how often it
runs):

``rate``      ELSEPA rate sum over the layer's elements at every step start.
``moments``   Soft stopping and hard rate radiative moments at step start.
``stopping``  SBETHE collision stopping at step start.
``cutoff``    Stopping plus soft radiative moment at the midpoint cutoff energy
              (evaluated every midpoint step, whether or not it truncates).
``midpoint``  Stopping plus soft radiative moment at the RK2 midpoint.
``clock``     Velocity for the transport clock (``beta``).
``select``    ELSEPA per-element rate recomputation for element selection at an
              elastic collision (CUDA: multi-element layers only, as the kernel).
``angle``     ELSEPA polar-angle inversion at the collision energy, including
              its energy-grid lookup.
"""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

PACKAGE = "pyrite.montecarlo.transport"
HERE = Path(__file__).parent
SINK = "-1.25e308"

CUDA_RAD_ARGS = """\
layer_index,
rad_nel,
rad_nt,
rad_z,
rad_ncm3,
rad_incident,
rad_nominal,
rad_top,
rad_chi,
rad_max_el,
rad_max_t,
rad_n_red,"""

CPU_RAD_ARGS = """\
rad_nel,
rad_nT,
rad_Z,
rad_ncm3,
rad_incident,
rad_nominal,
rad_top,
rad_chi,
L,"""


def _rad_cuda(which, energy_eV):
    args = CUDA_RAD_ARGS.replace("\n", " ")
    return f"_rad_transport_moment({which}, {args} {energy_eV}, rad_cutoff_eV, rad_plasma)"


def _rad_cpu(energy_eV):
    args = CPU_RAD_ARGS.replace("\n", " ")
    return (
        f"transport_layer_moments({args} {energy_eV}, rad_cutoff_eV, rad_scratch_rates, rad_plasma)"
    )


CUDA_PERTURB = "(F64_ONE - np.float64(1e-13) * amp_k)"
CUDA_SBETHE = (
    "_dEds_sbethe(L_sbethe_logE, L_sbethe_logS, layer_index * sbethe_width, "
    "L_sbethe_n[layer_index], {energy})"
)
CUDA_PACKED = (
    "_dEds_packed(L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, {energy})"
)
CUDA_RATE_TERM = """\
probe_row = el_start[element_row_offset + probe_el]
probe_count = el_len[element_row_offset + probe_el]
probe_lo = _log_grid_lower(el_logE, probe_row, probe_count, probe_log)
probe_f = _log_grid_fraction(el_logE, probe_row, probe_count, probe_log, probe_lo)
probe_low = el_log_rate[probe_row + probe_lo]
probe_sum += xp.exp(probe_low + probe_f * (el_log_rate[probe_row + probe_lo + I32_ONE] - probe_low))
"""


def _indent(text, spaces):
    pad = " " * spaces
    return "".join(pad + line if line.strip() else line for line in text.splitlines(True))


def _cuda_loop(body, sink_target, guard=None):
    """``K``-times loop accumulating ``probe_sum``, then the impossible sink."""
    loop = (
        "amp_k = I32_ZERO\n"
        "probe_sum = F64_ZERO\n"
        "while amp_k < np.int32({k}):\n"
        "    amp_k += I32_ONE\n" + _indent(body, 4) + f"if probe_sum == np.float64({SINK}):\n"
        f"    {sink_target}\n"
    )
    if guard is None:
        return loop
    return f"if {guard}:\n" + _indent(loop, 4)


def _cpu_loop(body, sink_target, guard=None):
    loop = (
        "probe_sum = 0.0\n"
        "for amp_k in range({k}):\n" + _indent(body, 4) + f"if probe_sum == {SINK}:\n"
        f"    {sink_target}\n"
    )
    if guard is None:
        return loop
    return f"if {guard}:\n" + _indent(loop, 4)


CUDA_RATE_LOOP = (
    "probe_log = log_E_eV - np.float64(1e-13) * amp_k\n"
    "probe_el = I32_ZERO\n"
    "while probe_el < n_elements:\n" + _indent(CUDA_RATE_TERM, 4) + "    probe_el += I32_ONE\n"
)

CUDA_MID_BODY = """\
probe_E = {energy} * {perturb}
if sbethe_on == I32_ONE:
    probe_sum += {sbethe}
else:
    probe_sum += {packed}
if rad_on == I32_ONE:
    probe_sum += {rad}
"""


def _cuda_mid(energy):
    return CUDA_MID_BODY.format(
        energy=energy,
        perturb=CUDA_PERTURB,
        sbethe=CUDA_SBETHE.format(energy="probe_E"),
        packed=CUDA_PACKED.format(energy="probe_E"),
        rad=_rad_cuda("RAD_SOFT", "probe_E * F64_EV_PER_KEV"),
    )


CUDA_ANGLE_BODY = """\
probe_row = el_start[element_row_offset + sel]
probe_count = el_len[element_row_offset + sel]
probe_log = xp.log(E_keV[electron_index] * np.float64(1e3)) - np.float64(1e-13) * amp_k
probe_lo = _log_grid_lower(el_logE, probe_row, probe_count, probe_log)
probe_f = _log_grid_fraction(el_logE, probe_row, probe_count, probe_log, probe_lo)
probe_sum += _elsepa_quantile_mu(
    el_cdf, el_pdf, el_mu, el_n_mu, el_guide, el_guide_cells, probe_row + probe_lo, probe_f,
    R_ang * (F64_ONE - np.float64(1e-13) * amp_k),
)
"""

CPU_PERTURB = "(1.0 - (amp_k + 1) * 1e-13)"
CPU_SBETHE = "_dEds_sbethe_packed_scalar(L_sbethe_logE, L_sbethe_logS, L, L_sbethe_n[L], {energy})"
CPU_RATE = "_elsepa_rate_scalar({energy}, el_logE, el_log_rate, el_start[L, probe_el], el_len[L, probe_el])"

CPU_MID_BODY = """\
probe_E = {energy} * {perturb}
if sbethe_on and not lut:
    probe_sum += {sbethe}
if radiative:
    probe_soft, probe_hard = {rad}
    probe_sum += probe_soft
"""


def _cpu_mid(energy):
    return CPU_MID_BODY.format(
        energy=energy,
        perturb=CPU_PERTURB,
        sbethe=CPU_SBETHE.format(energy="probe_E"),
        rad=_rad_cpu("probe_E * 1e3"),
    )


# component -> core -> (text before the insertion point, text after it, indent, block)
COMPONENTS = {
    "rate": {
        "cuda": (
            "",
            "        mean_free_path_ang = np.float64(1e8) / total_rate\n",
            8,
            _cuda_loop(CUDA_RATE_LOOP, "total_rate += F64_ONE", "elastic_model_code == I32_TWO"),
        ),
        "per-electron": (
            "",
            "            lam_ang = 1e8 / total_rate\n",
            12,
            _cpu_loop(
                f"probe_E = E_j * {CPU_PERTURB}\n"
                "for probe_el in range(n_el):\n"
                f"    probe_sum += {CPU_RATE.format(energy='probe_E')}\n",
                "total_rate += 1.0",
                "elastic_model_code == 2 and not lut",
            ),
        ),
    },
    "moments": {
        "cuda": (
            "",
            "        mean_free_path_ang = np.float64(1e8) / total_rate\n",
            8,
            _cuda_loop(
                f"probe_eV = energy_keV * F64_EV_PER_KEV * {CUDA_PERTURB}\n"
                f"probe_sum += {_rad_cuda('RAD_SOFT', 'probe_eV')}\n"
                f"probe_sum += {_rad_cuda('RAD_HARD_RATE', 'probe_eV')}\n",
                "total_rate += F64_ONE",
                "rad_on == I32_ONE",
            ),
        ),
        "per-electron": (
            "",
            "            lam_ang = 1e8 / total_rate\n",
            12,
            _cpu_loop(
                f"probe_soft, probe_hard = {_rad_cpu(f'E_j * 1e3 * {CPU_PERTURB}')}\n"
                "probe_sum += probe_soft + probe_hard\n",
                "total_rate += 1.0",
                "radiative",
            ),
        ),
    },
    "stopping": {
        "cuda": (
            "",
            "        rad_dEds = F64_ZERO\n",
            8,
            _cuda_loop(
                f"probe_sum += {CUDA_SBETHE.format(energy=f'energy_keV * {CUDA_PERTURB}')}\n",
                "dEds += F64_ONE",
                "sbethe_on == I32_ONE",
            ),
        ),
        "per-electron": (
            "",
            "            rad_dEds = 0.0\n",
            12,
            _cpu_loop(
                f"probe_sum += {CPU_SBETHE.format(energy=f'E_j * {CPU_PERTURB}')}\n",
                "dEds += 1.0",
                "sbethe_on and not lut",
            ),
        ),
    },
    "cutoff": {
        "cuda": (
            "",
            "                cutoff_distance = (cutoff_energy_keV - energy_keV) / cutoff_rate\n",
            16,
            _cuda_loop(_cuda_mid("cutoff_energy"), "cutoff_rate += F64_ONE"),
        ),
        "per-electron": (
            "",
            "                    cutoff_distance = (E_cut_e - E_j) / cutoff_rate\n",
            20,
            _cpu_loop(_cpu_mid("0.5 * (E_j + E_cut_e)"), "cutoff_rate += 1.0"),
        ),
    },
    "midpoint": {
        "cuda": (
            "",
            "                    end_energy_keV = energy_keV + step_length_ang * midpoint_rate\n",
            20,
            _cuda_loop(_cuda_mid("midpoint_energy"), "midpoint_rate += F64_ONE"),
        ),
        "per-electron": (
            "",
            "                        E_end_j = E_j + step_j * mid_rate\n",
            24,
            _cpu_loop(_cpu_mid("0.5 * (E_j + E_pred)"), "mid_rate += 1.0"),
        ),
    },
    "clock": {
        "cuda": (
            "",
            "        end_time_ang = clock[electron_index] + step_length_ang / segment_beta\n",
            8,
            _cuda_loop(
                f"probe_sum += _beta_from_keV(energy_keV * {CUDA_PERTURB})\n",
                "segment_beta += F64_ONE",
            ),
        ),
        "per-electron": (
            "",
            "                        t_end_j = clock[e] + step_j / beta_from_keV_scalar("
            "0.5 * (E_j + E_end_j))\n",
            24,
            _cpu_loop(
                f"probe_sum += beta_from_keV_scalar(0.5 * (E_j + E_end_j) * {CPU_PERTURB})\n",
                "E_end_j += 1.0",
            ),
        ),
    },
    "select": {
        "cuda": (
            "",
            "                        alpha = F64_ZERO\n",
            24,
            _cuda_loop(
                CUDA_RATE_LOOP,
                "sel = I32_ZERO",
                "elastic_model_code == I32_TWO and n_elements > I32_ONE",
            ),
        ),
        "per-electron": (
            "",
            "                alpha = 0.0\n",
            16,
            _cpu_loop(
                f"probe_E = E_j * {CPU_PERTURB}\n"
                "probe_cum = 0.0\n"
                "for probe_el in range(n_el):\n"
                f"    probe_cum += {CPU_RATE.format(energy='probe_E')}\n"
                "    if probe_cum > u:\n"
                "        break\n"
                "probe_sum += probe_cum\n",
                "i_el = 0",
                "elastic_model_code == 2 and not lut",
            ),
        ),
    },
    "angle": {
        "cuda": (
            "",
            "                        phi = (\n",
            24,
            _cuda_loop(CUDA_ANGLE_BODY, "cos_theta = F64_ONE", "elastic_model_code == I32_TWO"),
        ),
        "per-electron": (
            "                    cos_t = _sample_cos_theta_from_alpha(alpha, cos_u)\n",
            "                dirs[e, 0], dirs[e, 1], dirs[e, 2] = _rotate_direction_scalar(\n",
            16,
            _cpu_loop(
                "probe_sum += _sample_cos_theta_elsepa(\n"
                f"    E_keV[e] * {CPU_PERTURB},\n"
                "    cos_u * (1.0 - (amp_k + 1) * 1e-13),\n"
                "    el_logE, el_cdf, el_pdf, el_mu, el_guide, el_start[L, i_el], el_len[L, i_el],\n"
                ")\n",
                "cos_t = 1.0",
                "elastic_model_code == 2 and not lut",
            ),
        ),
    },
}
MODULES = {"cuda": "_jit_kernel", "per-electron": "cores"}
PRECISIONS = ("exact", "mixed", "mixed_math")
PRECISION_VARIANTS = HERE.parent / "fp32_speed_of_light/precision_variants.py"


def transform(source, component, core, k):
    """``source`` with the component's ``K``-fold probe at its one anchor."""
    before, after, indent, block = COMPONENTS[component][core]
    anchor = before + after
    if source.count(anchor) != 1:
        raise ValueError(f"{MODULES[core]}: {component} anchor inventory changed")
    if k == 0:
        return source
    probe = _indent(block.replace("{k}", str(int(k))), indent)
    return source.replace(anchor, before + probe + after)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.SourceLoader):
    """Serve one transformed transport module from a recorded file."""

    def __init__(self, name, path):
        self.name, self.path = name, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        return importlib.util.spec_from_file_location(fullname, self.path, loader=self)

    def get_filename(self, fullname):
        return str(self.path)

    def get_data(self, path):
        return Path(path).read_bytes()


def install(core, component, k, audit_dir):
    """Transform the core source before the transport package imports it."""
    if any(name.startswith(PACKAGE) for name in sys.modules):
        raise RuntimeError("install the probe before importing transport")
    spec = importlib.util.find_spec("pyrite.montecarlo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyrite.montecarlo is not importable")
    module = MODULES[core]
    original = (Path(spec.submodule_search_locations[0]) / "transport" / f"{module}.py").read_text()
    source = transform(original, component, core, k)
    destination = audit_dir / f"{module}-{component}-k{k}.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    sys.meta_path.insert(0, _Finder(f"{PACKAGE}.{module}", destination))
    return {
        "component": component,
        "amplification": int(k),
        "module": module,
        "reference_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def install_precision(variant):
    """Compose a #831 FP32 timing variant with the amplified source (CUDA only).

    ``precision_variants.InstrumentedKernel`` rewrites the #827 variant's
    generated source (here, of the probe-transformed template) and passes
    float32 table copies, cached per device pointer; the cache is cleared at
    every request, as the #831 harness does. A timing proxy for the #838
    mixed-precision kernel, not that kernel: its physics is not validated.
    """
    precision = runpy.run_path(str(PRECISION_VARIANTS))
    import pyrite.montecarlo.transport as transport
    from pyrite.montecarlo.transport import _jit_launch

    stats = {"variant": variant, "kernels": {}, "_events": {}, "_seed": None, "_f32_cache": {}}
    original = _jit_launch.transport_kernel_variant
    wrapped = {}

    def variant_kernel(switches):
        if switches not in wrapped:
            wrapped[switches] = precision["InstrumentedKernel"](
                original(switches), variant, stats, None
            )
        return wrapped[switches]

    _jit_launch.transport_kernel_variant = variant_kernel
    simulate = transport.simulate_trajectories

    def fresh_tables(*a, **kw):
        stats["_f32_cache"].clear()
        stats["_events"].clear()
        return simulate(*a, **kw)

    transport.simulate_trajectories = fresh_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=sorted(COMPONENTS))
    parser.add_argument("amplification", type=int)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--precision", choices=PRECISIONS, default="exact")
    args, baseline_args = parser.parse_known_args()
    if args.amplification < 0:
        parser.error("amplification must be nonnegative")
    core = baseline_args[baseline_args.index("--core") + 1]
    # Separate Numba caches per transformed source keep timings honest.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(args.audit_dir / f"numba-{args.component}-k{args.amplification}"),
    )
    if args.precision != "exact" and core != "cuda":
        parser.error("--precision variants are CUDA timing proxies")
    probe = install(core, args.component, args.amplification, args.audit_dir)
    probe["precision"] = args.precision
    if args.precision != "exact":
        install_precision(args.precision)
    harness = runpy.run_path(str(HERE / "baseline_803.py"))
    harness["main"](baseline_args)
    output = Path(baseline_args[baseline_args.index("--output") + 1])
    record = json.loads(output.read_text())
    record.update(schema="pyrite.lut-probe-803.sample.v1", probe=probe)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
