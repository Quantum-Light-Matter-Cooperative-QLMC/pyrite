"""Ordered download locations for one pinned archive.

A pinned archive may be published in several places: upstream, PyRITE's own
GitHub Release, later a Zenodo record. Fetching tries them in order and the
pinned SHA-256 decides what installs, so adding a mirror cannot change what
gets installed. Release indexes record the list as ``archive.urls``; the
single ``archive.url`` of earlier indexes is still read.
"""

from collections.abc import Iterable, Mapping
from typing import Any


def as_urls(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize ``None``, one URL, or several URLs to an ordered tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    urls = tuple(value)
    if not all(isinstance(url, str) and url for url in urls):
        raise ValueError(f"download locations must be non-empty strings, got {urls!r}")
    return urls


def archive_urls(archive: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered URLs of an index's ``archive`` record."""
    if "urls" in archive:
        urls = archive["urls"]
        if isinstance(urls, str):
            raise ValueError("archive.urls must be a list of URLs, not a string")
        return as_urls(urls)
    return as_urls(archive.get("url"))


__all__ = ["archive_urls", "as_urls"]
