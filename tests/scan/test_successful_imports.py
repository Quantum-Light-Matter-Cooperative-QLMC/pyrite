import subprocess
import sys


def test_api_imports_in_clean_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", "import pyrite.api"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
