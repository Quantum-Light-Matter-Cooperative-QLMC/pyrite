# tests/montecarlo/test_imports.py

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_cpu_import_path_does_not_require_cupy() -> None:
    """The normal CPU import path must work when CuPy is unavailable."""

    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        import numpy as np


        class RejectCupyImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "cupy"
                    or fullname.startswith("cupy.")
                    or fullname == "cupyx"
                    or fullname.startswith("cupyx.")
                ):
                    raise ModuleNotFoundError(
                        "No module named 'cupy'",
                        name="cupy",
                    )

                return None


        sys.meta_path.insert(0, RejectCupyImports())

        # Reproduce the import chain that originally failed:
        #
        # config
        # -> montecarlo
        # -> spectrum_jit_kernel
        #
        # The import should now finish using the CPU backend.
        importlib.import_module("cxr_mc.config")

        from cxr_mc.montecarlo import _backend
        from cxr_mc.montecarlo import runner
        from cxr_mc.montecarlo import spectrum
        from cxr_mc.montecarlo import transport

        assert _backend._GPU is False
        assert spectrum.xp is np
        assert runner._GPU is False

        # CUDA-only implementation modules must remain unloaded on CPU.
        cuda_modules = {
            "cxr_mc.montecarlo.spectrum_jit_kernel",
            "cxr_mc.montecarlo.coherent_jit_kernel",
            "cxr_mc.montecarlo.brem_jit_kernel",
        }

        assert cuda_modules.isdisjoint(sys.modules)
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"CPU import path unexpectedly required CuPy:\n{result.stderr}"


def test_sincsq_lineshape_runs_on_cpu() -> None:
    import numpy as np

    from cxr_mc.montecarlo import spectrum

    result = spectrum._sincsq_lineshape(
        np.array([1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0]),
    )

    assert isinstance(result, np.ndarray)
    assert np.all(np.isfinite(result))
    assert result[1] == 1.0


# tests/montecarlo/test_init_imports.py


def test_config_import_does_not_require_cupy() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class RejectCupyImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "cupy"
                    or fullname.startswith("cupy.")
                    or fullname == "cupyx"
                    or fullname.startswith("cupyx.")
                ):
                    raise ModuleNotFoundError(
                        "No module named 'cupy'",
                        name="cupy",
                    )

                return None


        sys.meta_path.insert(0, RejectCupyImports())

        # This is the import chain that originally prevented CLI startup
        # and pytest discovery.
        import cxr_mc.config

        from cxr_mc.montecarlo._backend import BACKEND

        assert BACKEND.name not in {"cuda", "rocm"}

        assert not any(
            name == "cupy"
            or name.startswith("cupy.")
            or name == "cupyx"
            or name.startswith("cupyx.")
            for name in sys.modules
        )
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
