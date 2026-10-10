"""The catalog as the suite sees it: bundled demos plus the fixture user layer.

The research campaign profiles the suite pins live in the tracked fixture user
layer (``tests/data/user_catalog``, see ``tests/conftest.py``), read over the
bundled catalog. A test that needs a standalone copy of that view -- an
external catalog, or a writable catalog behind ``_catalog_io._CATALOG_PATH`` --
copies both.
"""

import shutil
from pathlib import Path

from pyrite._catalog_layout import bundled_catalog

USER_CATALOG_FIXTURE = Path(__file__).parents[1] / "data" / "user_catalog"


def copy_full_catalog(destination: Path) -> Path:
    """Copy the bundled catalog and the fixture user profiles to ``destination``."""
    shutil.copytree(bundled_catalog(), destination, dirs_exist_ok=True)
    shutil.copytree(USER_CATALOG_FIXTURE / "profiles", destination / "profiles", dirs_exist_ok=True)
    return destination
