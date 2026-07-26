"""Shared anywidget controls for the marimo apps.

Imported by ``analysis_app.py`` and ``trace_app.py`` the same way ``_design``
is: marimo runs apps with ``notebooks/`` on ``sys.path``, so a plain
``from _widgets import MaterialSelect`` works in both.
"""

import traitlets
from anywidget import AnyWidget


class MaterialSelect(AnyWidget):
    """Native ``<select>`` with per-row disabled flags and a bold label."""

    _esm = r"""
    function render({ model, el }) {
      const label = document.createElement("label");
      const labelText = document.createElement("strong");
      labelText.textContent = model.get("label");
      label.appendChild(labelText);
      const select = document.createElement("select");
      select.setAttribute("aria-label", model.get("label"));
      for (const row of model.get("options")) {
        const option = document.createElement("option");
        option.value = row.value;
        option.textContent = row.label;
        option.disabled = row.disabled;
        select.appendChild(option);
      }
      select.value = model.get("value") ?? "";
      select.disabled = model.get("disabled");
      select.addEventListener("change", () => {
        model.set("value", select.value);
        model.save_changes();
      });
      label.appendChild(select);
      el.replaceChildren(label);
    }
    export default { render };
    """
    options = traitlets.List().tag(sync=True)
    value = traitlets.Unicode(allow_none=True, default_value=None).tag(sync=True)
    label = traitlets.Unicode().tag(sync=True)
    disabled = traitlets.Bool().tag(sync=True)
