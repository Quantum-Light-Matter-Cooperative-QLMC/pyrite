"""Shaped-electron (momentum-superposition) CXR interference reference for #1102.

Isolated research calculation; no production code uses it. It evaluates
Shi et al., Nat. Commun. 15, 7803 (2024), Eq. (3) / SI Eq. (S17) for a
density matrix ``rho`` over a few plane-wave components of one electron, in a
laterally infinite static Si [001] slab of thickness ``L``:

    d2N/(dw dOmega) ~ w * sum_sigma Tr(A rho A^dagger),
    A[f, i] = sum_{g : q_i,perp + g_perp = Q_f} M_sigma(q_i, g) F(Delta_ig).

The final electron state ``f`` is labelled by its transverse wavevector
``Q_f - k_perp`` at energy ``E - hbar w`` (energy and transverse momentum are
conserved by the static, laterally periodic slab), so the trace over
unobserved final electrons is the explicit sum over ``f``. ``F`` is the slab
formation amplitude of ``quantum._longitudinal.slab_formation_amplitude`` at
the longitudinal mismatch ``Delta = q_iz + g_z - k_z - q_fz`` (forward final
branch). ``M`` is Shi's spinless PXR + CB element (SI Eqs. (S14)-(S15) with
``u_f = u_i``) with Shi's Salvat-Yukawa Si susceptibility and potential (SI
Eq. (S3)). Shi's reciprocal label is used internally (``q_f = q_i + g - k``);
``materials.recoil.recoil_resonance_eV`` takes ``-g``. Only ratios are
reported, so the common prefactor is dropped.

Parts:
1. plane-wave limit against the #527 incoherent per-reflection sum, with
   the line centres on ``recoil_resonance_eV``;
2. Shi Fig. 5 / SI S7 two-component state (30 keV, 30 nm, theta = 110 deg):
   coherent versus mixture, relative-phase scan, final-state selection rule;
3. momentum-offset scan (excitation error) and non-lattice offsets;
4. impact-position (registration) averaging;
5. two-beam electron Pendelloesung distance for the coupling vector.

Run:  uv run python checks/shaped_electron_1102.py

Validation: quantum-shaped-cxr-interference
"""

import numpy as np

from pyrite.materials.recoil import recoil_resonance_eV
from pyrite.quantum._longitudinal import slab_formation_amplitude

HBARC = 1973.269804  # eV Ang
MC2 = 510998.95  # eV
R_E = 2.8179403262e-5  # Ang
E2_4PIEPS0 = 14.3996454  # eV Ang
A0 = 0.529177211  # Ang
A_SI = 5.431  # Ang
B0 = 2 * np.pi / A_SI
V_CELL = A_SI**3
Z_SI = 14
YUKAWA = ((0.5160, 5.8492), (0.4840, 1.1732))  # Shi SI S1 (C_k, mu_k)
BASIS = np.array(
    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    + [[0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]
)
KEY = 1e-6  # Ang^-1 rounding for final-state labels


# --- crystal -----------------------------------------------------------------


def _structure(hkl):
    return np.exp(-2j * np.pi * (np.asarray(hkl, float) @ BASIS.T)).sum(axis=-1)


def chi_g(hkl, photon_eV):
    """Shi Eq. (S3) X-ray susceptibility Fourier coefficient (no Debye-Waller)."""
    g2 = np.sum((np.asarray(hkl, float) * B0) ** 2, axis=-1)
    form = sum(c * (mu / A0) ** 2 / ((mu / A0) ** 2 + g2) for c, mu in YUKAWA)
    lam = 2 * np.pi * HBARC / photon_eV
    return -R_E * lam**2 / (np.pi * V_CELL) * Z_SI * form * _structure(hkl)


def potential_g(hkl):
    """Electron potential-energy Fourier coefficient U_g in eV (attractive < 0)."""
    g2 = np.sum((np.asarray(hkl, float) * B0) ** 2, axis=-1)
    screen = sum(c / ((mu / A0) ** 2 + g2) for c, mu in YUKAWA)
    return -4 * np.pi * E2_4PIEPS0 * Z_SI / V_CELL * screen * _structure(hkl)


def reflections(hmax=7, lmax=6):
    span = np.arange(-hmax, hmax + 1)
    hh, kk, ll = np.meshgrid(span, span, np.arange(-lmax, lmax + 1), indexing="ij")
    hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=-1)
    par = hkl % 2
    same = (par == par[:, :1]).all(axis=1)
    allowed = same & ((par[:, 0] == 1) | (hkl.sum(axis=1) % 4 == 0))
    allowed &= np.any(hkl != 0, axis=1)
    return hkl[allowed]


# --- kinematics ----------------------------------------------------------------


def k_of(kinetic_eV):
    return np.sqrt(kinetic_eV * (kinetic_eV + 2 * MC2)) / HBARC


def component(kinetic_eV, q_perp_b0):
    """On-shell wavevector with transverse part ``q_perp_b0 * 2pi/a``."""
    qp = np.asarray(q_perp_b0, float) * B0
    return np.array([qp[0], qp[1], np.sqrt(k_of(kinetic_eV) ** 2 - qp @ qp)])


def energy(q):
    return np.sqrt((HBARC * np.linalg.norm(q, axis=-1)) ** 2 + MC2**2)


def photon_frame(theta_deg, phi_deg=0.0):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    n = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    e1 = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)])
    return n, (e1, np.cross(n, e1))


# --- amplitudes ------------------------------------------------------------------


def pathway_amplitudes(q, photon_eV, n, pols, hkl, thickness, kz_shift=0.0, exclude=()):
    """Return final labels, ``M_sigma F`` per (sigma, reflection), and mismatch.

    ``q`` is one on-shell 3D initial wavevector; ``hkl`` uses Shi's sign
    (``q_f = q + g - k``). Evanescent finals are dropped (amplitude 0).
    ``kz_shift`` adds a Bloch-branch wavevector shift to the mismatch only.
    CB terms whose lattice scatter is a reciprocal vector in ``exclude`` (the
    coupling between the shaped components) are dropped: near the electron
    Bragg condition the kinematic intermediate ``1/(E_q - E_{q+g})`` is a pole,
    and that scattering belongs to the electron state (two-beam Bloch waves,
    Parts 3 and 5), not to the emission vertex.
    """
    g = hkl * B0
    kvec = photon_eV / HBARC * n
    e_i = energy(q)
    beta = HBARC * q / e_i
    qf_perp = q[:2] + g[:, :2] - kvec[:2]
    kf2 = ((e_i - photon_eV) ** 2 - MC2**2) / HBARC**2
    with np.errstate(invalid="ignore"):
        qfz = np.sqrt(kf2 - np.sum(qf_perp**2, axis=1))
    ok = np.isfinite(qfz)
    qfz = np.where(ok, qfz, 0.0)
    delta = q[2] + kz_shift + g[:, 2] - kvec[2] - qfz
    form = np.where(ok, slab_formation_amplitude(delta, thickness), 0.0)
    kg = kvec - g
    denom = np.sum(kg**2, axis=1) - kvec @ kvec
    chi = chi_g(hkl, photon_eV)
    pot = potential_g(hkl)
    drop = np.zeros(len(hkl), bool)
    for ex in exclude:
        drop |= np.all(hkl == np.asarray(ex), axis=1)
    gap = e_i - energy(q + g)
    d1 = np.where(drop, 0.0, 1.0 / np.where(drop, 1.0, gap))
    d2 = 1.0 / ((e_i - photon_eV) - energy(q - kvec))
    out = []
    for e in pols:
        # beta . [k^2 I - (k-g)(k-g)] . e
        pxr = (kvec @ kvec) * (beta @ e) - (kg @ beta) * (kg @ e)
        amp = chi * pxr / denom + pot * (beta @ e) * (d1 + d2)
        out.append(amp * form)
    labels = np.round((q[:2] + g[:, :2]) / KEY).astype(np.int64)
    return labels, np.array(out), np.where(ok, delta, np.nan)


def amplitude_matrix(states, photon_eV, n, pols, hkl, thickness, kz_shifts=None, exclude=()):
    """Return ``A[sigma, f, i]`` with the explicit final-state bookkeeping."""
    shifts = np.zeros(len(states)) if kz_shifts is None else kz_shifts
    per = [
        pathway_amplitudes(q, photon_eV, n, pols, hkl, thickness, dz, exclude)
        for q, dz in zip(states, shifts, strict=True)
    ]
    keys = {}
    for labels, _, _ in per:
        for lab in map(tuple, labels):
            keys.setdefault(lab, len(keys))
    mat = np.zeros((len(pols), len(keys), len(states)), complex)
    for i, (labels, amps, _) in enumerate(per):
        rows = np.array([keys[tuple(lab)] for lab in labels])
        for s in range(len(pols)):
            np.add.at(mat[s, :, i], rows, amps[s])
    return mat


def rate(states, rho, photon_eV, n, pols, hkl, thickness):
    """``w * sum_sigma Tr(A rho A^dagger)``, arbitrary units."""
    mat = amplitude_matrix(states, photon_eV, n, pols, hkl, thickness)
    return photon_eV * np.einsum("sfi,ij,sfj->", mat, rho, mat.conj()).real


def incoherent_pathway_sum(q, photon_eV, n, pols, hkl, thickness, exclude=()):
    """#527 plane-wave form: every reflection as its own incoherent line."""
    _, amps, _ = pathway_amplitudes(q, photon_eV, n, pols, hkl, thickness, 0.0, exclude)
    return photon_eV * np.sum(np.abs(amps) ** 2)


# --- parts -------------------------------------------------------------------------

T30 = 30e3
L30 = 300.0
HKL = reflections()
N110, POL110 = photon_frame(110.0)
GRID = np.linspace(1850.0, 2100.0, 1001)
PHASES = np.radians(np.arange(0, 360, 30))


class Spectra:
    """Amplitude matrices on a photon grid, reused for any density matrix."""

    def __init__(self, states, shifts=None, exclude=(), grid=GRID, n=N110, pols=POL110):
        self.grid = grid
        self.mats = [amplitude_matrix(states, w, n, pols, HKL, L30, shifts, exclude) for w in grid]

    def __call__(self, rho):
        return np.array(
            [
                w * np.einsum("sfi,ij,sfj->", m, rho, m.conj()).real
                for w, m in zip(self.grid, self.mats, strict=True)
            ]
        )

    def visibility(self):
        """``|cross| / mean direct`` per frequency (two states; phase-origin free)."""
        out = []
        for m in self.mats:
            cross = abs(np.einsum("sf,sf->", m[:, :, 0], m[:, :, 1].conj()))
            direct = 0.5 * np.einsum("sfi,sfi->", m, m.conj()).real
            out.append(cross / direct if direct > 0 else 0.0)
        return np.array(out)


def two_state_rho(phase, coherence=1.0):
    """Density matrix of ``(|q1> + exp(i phase)|q2>)/sqrt(2)``, dephased by ``coherence``."""
    off = 0.5 * coherence * np.exp(-1j * phase)
    return np.array([[0.5, off], [np.conj(off), 0.5]])


def coupling(q1, q2):
    """Reciprocal vector (Miller, Shi sign) taking q2 to q1, and its negative."""
    hkl = np.rint((q1 - q2) / B0).astype(int)
    return hkl, (tuple(hkl), tuple(-hkl))


class TwoBeam:
    """Two-beam Bloch waves of the pair (q1, q2) at fixed energy.

    ``i d psi/dz = K psi`` with ``K = diag(q1z, q2z) - [[0, U], [U*, 0]]/(hbar v_z)``,
    ``U = U_{q1-q2}``; entry continuity ``psi(0) = psi_in``. Each branch b is
    emitted from with its own wavevector ``kappa_b`` (mismatch shift
    ``kappa_b - q_iz``), coherently over branches and components.
    """

    def __init__(self, q1, q2, grid=GRID):
        hkl, self.exclude = coupling(q1, q2)
        self.u = complex(potential_g(hkl))
        self.hv = HBARC**2 * 0.5 * (q1[2] + q2[2]) / energy(q1)
        k = np.diag([q1[2], q2[2]]).astype(complex)
        k -= np.array([[0, self.u], [np.conj(self.u), 0]]) / self.hv
        self.kappa, self.vec = np.linalg.eigh(k)
        shifts = [self.kappa[b] - q[2] for b in range(2) for q in (q1, q2)]
        self.spec = Spectra([q1, q2, q1, q2], np.array(shifts), self.exclude, grid)
        self.xi = np.pi * self.hv / abs(self.u)

    def __call__(self, psi):
        c = self.vec.conj().T @ psi
        w = np.concatenate([c[b] * self.vec[:, b] for b in range(2)])
        return self.spec(np.outer(w, w.conj()))

    def mixture(self):
        return 0.5 * (self(np.array([1.0, 0.0])) + self(np.array([0.0, 1.0])))


def psi_phase(phase):
    return np.array([1.0, np.exp(1j * phase)]) / np.sqrt(2)


def part1_plane_wave():
    print("Part 1: plane-wave limit (30 keV, 30 nm Si [001], theta = 110 deg)")
    # (1, 2) alone already satisfies the (040) electron Bragg condition: drop that CB pole.
    pole = ((0, 4, 0), (0, -4, 0))
    for qperp, ex in (((0, 0), ()), ((1, 2), pole)):
        q = component(T30, qperp)
        exact = Spectra([q], exclude=ex)(np.ones((1, 1)))
        inc = np.array([incoherent_pathway_sum(q, w, N110, POL110, HKL, L30, ex) for w in GRID])
        top = exact > 0.05 * exact.max()
        dev = np.max(np.abs(exact[top] / inc[top] - 1))
        local = (exact[1:-1] > exact[:-2]) & (exact[1:-1] >= exact[2:])
        local &= exact[1:-1] > 0.2 * exact.max()
        at_peaks = np.max(np.abs(exact[1:-1][local] / inc[1:-1][local] - 1))
        u = q / np.linalg.norm(q)
        roots = recoil_resonance_eV(T30, u, -HKL * B0, N110)
        sel = np.isfinite(roots) & (roots > GRID[0]) & (roots < GRID[-1])
        worst = 0.0
        for hkl, w in zip(HKL[sel], roots[sel], strict=True):
            _, _, d = pathway_amplitudes(q, w, N110, POL110, hkl[None], L30)
            worst = max(worst, abs(d[0]))
        print(
            f"  q_perp={qperp}: exact (coherent over g_z at fixed g_perp) vs #527 incoherent"
            f" per-g sum, max |ratio-1| at peaks >20% = {at_peaks:.1e}, where >5% = {dev:.1e};"
            f" {sel.sum()} roots in window, max |Delta| at root = {worst:.1e} 1/Ang"
        )


def part2_shi_fig5():
    print("Part 2: Shi Fig. 5 state q1=(1,2,qz), q2=(1,-2,qz) [2pi/a], kinematic electron")
    q1, q2 = component(T30, (1, 2)), component(T30, (1, -2))
    spec = Spectra([q1, q2], exclude=coupling(q1, q2)[1])
    s1, s2 = spec(np.diag([1.0, 0.0])), spec(np.diag([0.0, 1.0]))
    mix = spec(two_state_rho(0.0, 0.0))
    peak = mix.max()
    print(
        f"   mixture == (R1+R2)/2: max |diff|/peak = {np.max(np.abs(mix - 0.5 * (s1 + s2))) / peak:.1e}"
    )
    cells = []
    for ph in PHASES:
        coh = spec(two_state_rho(ph))
        integ = np.trapezoid(coh - mix, GRID) / np.trapezoid(mix, GRID)
        cells.append(f"{np.degrees(ph):3.0f}: {coh.max() / peak:4.2f} {integ:+.2f}")
    print("   phase deg: peak coherent/mixture, window-integrated (coh-mix)/mix")
    print("   " + "; ".join(cells))
    avg = np.mean([spec(two_state_rho(ph)) for ph in PHASES], axis=0)
    print(
        f"   phase-averaged coherent vs mixture: max |diff|/peak = {np.max(np.abs(avg - mix)) / peak:.1e}"
    )
    vis = spec.visibility()
    strong = mix > 0.2 * peak
    print(
        f"   visibility |cross|/mean-direct where mixture > 20% of peak: {vis[strong].min():.2f}-{vis[strong].max():.2f}"
    )
    print(f"   mixture peak at {GRID[np.argmax(mix)]:.1f} eV")
    return q1, q2


def part3_offsets(q1, q2):
    print("Part 3: excitation-error scan; both components shifted by d [2pi/a] along y")
    print("   (two-beam Bloch electron; kinematic value in brackets, valid only for |s| >> pi/xi)")
    grid = GRID[::2]
    for d in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        a, b = component(T30, (1, 2 + d)), component(T30, (1, -2 + d))
        dyn = TwoBeam(a, b, grid)
        dmix = dyn.mixture()
        kin = Spectra([a, b], exclude=dyn.exclude, grid=grid)
        kmix = kin(two_state_rho(0.0, 0.0))
        best_d = max(np.max(np.abs(dyn(psi_phase(ph)) - dmix)) for ph in PHASES[:6]) / dmix.max()
        best_k = (
            max(np.max(np.abs(kin(two_state_rho(ph)) - kmix)) for ph in PHASES[:6]) / kmix.max()
        )
        s = a[2] - b[2]
        print(
            f"   d = {d:4.2f}: s = q1z-q2z = {s:+.5f} 1/Ang, |s|L = {abs(s) * L30:5.2f},"
            f" |s|xi/pi = {abs(s) * dyn.xi / np.pi:5.2f}; max_phase |coh-mix|/peak = {best_d:.2f} [{best_k:.2f}]"
        )
    for eps in (1e-3, 0.05):
        b = component(T30, (1, -2 - eps))
        m = amplitude_matrix([q1, b], 1950.0, N110, POL110, HKL, L30, None, coupling(q1, q2)[1])
        shared = np.sum(np.any(np.abs(m[..., 0]) > 0, 0) & np.any(np.abs(m[..., 1]) > 0, 0))
        print(
            f"   q2 offset by eps = {eps} [2pi/a] off the lattice: shared final states = {shared}"
        )


def part4_registration(q1, q2):
    print("Part 4: lateral registration jitter sigma (interference factor exp(-|G|^2 sigma^2/2))")
    gvec = q1[:2] - q2[:2]
    rng = np.random.default_rng(1102)
    for sigma in (0.05, 0.1, 0.2, 0.3, 0.5):
        shift = rng.normal(0.0, sigma, size=(200000, 2))
        # translating the electron by d multiplies phi(q) by exp(-i q.d): rho_12 *= exp(-i(q1-q2).d)
        mc = abs(np.mean(np.exp(-1j * shift @ gvec)))
        print(
            f"   sigma {sigma:.2f} Ang: sampled {mc:.3f}, analytic {np.exp(-0.5 * (gvec @ gvec) * sigma**2):.3f}"
        )
    print(
        f"   |G_perp| = {np.linalg.norm(gvec):.3f} 1/Ang; uniform registration over a cell averages it to 0"
    )


def part5_dynamical(q1, q2):
    print(
        "Part 5: two-beam electron diffraction by G = q1 - q2 (Pendelloesung xi = pi hbar v_z/|U_G|)"
    )
    for hkl in ((0, 4, 0), (2, 2, 0)):
        u = abs(potential_g(np.array(hkl)))
        xis = []
        for t in (30e3, 100e3, 200e3, 300e3):
            beta = HBARC * k_of(t) / (t + MC2)
            xis.append(f"{t / 1e3:.0f} keV {np.pi * HBARC * beta / u / 10:.0f} nm")
        print(f"   {hkl}: |U_G| = {u:.2f} eV (Shi Yukawa, no Debye-Waller); xi: " + ", ".join(xis))
    dyn = TwoBeam(q1, q2)
    print(
        f"   Fig. 5: U_040 = {dyn.u.real:+.2f} eV, xi = {dyn.xi / 10:.1f} nm, L/xi = {L30 / dyn.xi:.2f};"
        f" single-component entry transfers sin^2(pi L/xi) = {np.sin(np.pi * L30 / dyn.xi) ** 2:.2f}"
    )
    print(
        "   Bloch eigenstates: phi = 0 (lower branch, density on (040) atom planes), phi = 180 deg (upper)"
    )
    kin = Spectra([q1, q2], exclude=dyn.exclude)
    kmix = kin(two_state_rho(0.0, 0.0))
    dmix = dyn.mixture()
    print(f"   dynamical mixture peak = {dmix.max() / kmix.max():.2f} x kinematic mixture peak")
    cells = []
    for ph in PHASES:
        kc, dc = kin(two_state_rho(ph)), dyn(psi_phase(ph))
        cells.append(
            f"{np.degrees(ph):3.0f}: {kc.max() / kmix.max():4.2f}/{dc.max() / dmix.max():4.2f}"
        )
    print("   phase deg: peak coherent/mixture, kinematic / two-beam dynamical")
    print("   " + "; ".join(cells))


if __name__ == "__main__":
    part1_plane_wave()
    q1, q2 = part2_shi_fig5()
    part3_offsets(q1, q2)
    part4_registration(q1, q2)
    part5_dynamical(q1, q2)
