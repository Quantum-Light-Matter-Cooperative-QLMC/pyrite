"""Preserve process-global compatibility registries around authoring tests."""

import pytest


@pytest.fixture(autouse=True)
def preserve_catalog_registry():
    from pyrite.materials.catalog import _get_default_catalog, _refresh_default_catalog

    previous = _get_default_catalog()
    try:
        yield
    finally:
        _refresh_default_catalog(previous)
