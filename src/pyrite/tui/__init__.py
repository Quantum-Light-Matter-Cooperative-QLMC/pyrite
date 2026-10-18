"""Optional terminal editors over the shared catalog configuration services.

Only :mod:`pyrite.tui.detector_app` imports Textual (the ``tui`` extra). The
form model in :mod:`pyrite.tui.detector_form` is UI-independent: it stages edits
through :class:`pyrite.catalog_edit.CatalogDraft` and
:mod:`pyrite.detectors.editing`, exactly as ``pyrite detector`` does, and never
imports the Click CLI.
"""
