# Additional terms for the `svml` build

This notice covers the pixi `svml` environment defined in `pyproject.toml`
(`[tool.pixi.feature.svml]`, #1459/#1462), and anything built or run from
it: the installed environment, Numba caches compiled with Intel SVML, and
results produced while studying the SVML kernels. These terms are **more
restrictive than `LICENSE.txt`**. They apply on top of it and do not replace
it.

They do not apply to the PyPI package (`pip install pyrite-mc`) or to the
`default`, `py314`, `docs`, `nvidia`, `amd` and `intel` environments. Those
environments use only conda-forge and PyPI and are covered by `LICENSE.txt`
and `THIRD-PARTY-NOTICES.md` alone.

## Why the `svml` build is different

The `svml` environment is the only one that installs from the Anaconda
`pkgs/main` channel (`https://repo.anaconda.com/pkgs/main`), because only
that channel ships the SVML-patched `llvmlite` that Numba needs to call Intel's
Short Vector Math Library. Two of its packages carry terms that the rest of
the stack does not:

| Component | Source | Terms |
|---|---|---|
| `numba` 0.61, `llvmlite` 0.44 (SVML builds), `numpy` 2.2, Python, and the other `pkgs/main` packages | Anaconda `pkgs/main` | [Anaconda Terms of Service](https://legal.anaconda.com/policies/en/?name=terms-of-service). Free use by accredited universities is limited to non-commercial, educational purposes; the terms say nothing about grant-funded research. |
| `intel-cmplr-lib-rt` (the SVML runtime, `libsvml.so`) | Anaconda `pkgs/main` | Intel End User License Agreement for Developer Tools (proprietary). It may be depended on, but never vendored. |

## Terms for the `svml` build

1. **Who may use it.** Use is limited to academic or nonprofit researchers at
   an accredited educational institution, for non-commercial, educational or
   academic research purposes. Before relying on the build for grant-funded or
   sponsored work, confirm that the work fits Anaconda's educational-use terms.
   Commercial use requires your own Anaconda commercial licence. It is not
   covered by `LICENSE.txt` or by this notice.
2. **No redistribution.** Do not redistribute the installed `svml`
   environment, its conda packages, `libsvml.so` or any other part of
   `intel-cmplr-lib-rt`, or Numba cache files compiled against SVML. That
   includes container images, wheels, tarballs and shared caches outside your
   institution. Share the lock (`pixi.lock`) instead; each user installs from
   the upstream channel under their own acceptance of its terms.
3. **No vendoring.** The repository must never contain copies of
   `pkgs/main` packages or of `intel-cmplr-lib-rt`; it only names them as
   dependencies.
4. **Results.** Evidence produced with the `svml` build records that it came
   from the `svml` environment, for example through `numba -s` output, and it
   is never used as the identity pin or default physics baseline. Those come
   from the `default` environment.
5. **CI.** Continuous integration builds and tests the `svml` environment only
   to check that the SVML kernels still work. No CI artifact built from it is
   published.
6. **Upstream terms govern.** This notice is a summary. If it conflicts with the
   Anaconda Terms of Service or the Intel EULA, those upstream terms win.
   Accepting `LICENSE.txt` does not grant any right under them.

Questions about these terms: see the contact in `LICENSE.txt`.
