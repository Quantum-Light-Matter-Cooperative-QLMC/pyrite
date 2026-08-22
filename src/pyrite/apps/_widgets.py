"""Shared anywidget controls for the PyRITE marimo apps."""

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
      select.style.backgroundColor = "var(--cxr-surface, #FFFFFF)";
      select.style.color = "var(--cxr-text, #172832)";
      select.style.borderColor = "var(--cxr-rule, #C6D2D8)";
      select.style.colorScheme = "light dark";
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


class ThemeSelect(AnyWidget):
    """Shared System/Light/Dark selector synchronized across PyRITE browser tabs.

    The preference is stored in browser localStorage.  The resolved light/dark
    value is synchronized back to Python so figures that need explicit styling
    (notably Plotly 3D scenes) can react without tying theme changes to transport.
    """

    _esm = r"""
    const STORAGE_KEY = "pyrite:appearance";
    const VALID = new Set(["system", "light", "dark"]);

    const COOKIE_KEY = "pyrite_appearance";

    function cookieRead() {
      const prefix = `${COOKIE_KEY}=`;
      for (const part of document.cookie.split(";")) {
        const item = part.trim();
        if (item.startsWith(prefix)) {
          const value = decodeURIComponent(item.slice(prefix.length));
          return VALID.has(value) ? value : null;
        }
      }
      return null;
    }

    function safeRead() {
      const cookieValue = cookieRead();
      if (cookieValue) return cookieValue;
      try {
        const value = localStorage.getItem(STORAGE_KEY);
        return VALID.has(value) ? value : null;
      } catch (_) {
        return null;
      }
    }

    function safeWrite(value) {
      try {
        localStorage.setItem(STORAGE_KEY, value);
      } catch (_) {
        // Storage may be disabled (e.g. hardened/private browser settings).
      }
      document.cookie = `${COOKIE_KEY}=${encodeURIComponent(value)}; Path=/; Max-Age=31536000; SameSite=Lax`;
    }

    function resolvedTheme(preference, media) {
      return preference === "system" ? (media.matches ? "dark" : "light") : preference;
    }

    function syncMarimoShadowThemes(resolved) {
      for (const element of document.querySelectorAll("*")) {
        const shadow = element.shadowRoot;
        if (!shadow) continue;
        for (const wrapper of shadow.querySelectorAll(".marimo > .contents")) {
          wrapper.classList.remove("light", "dark");
          wrapper.classList.add(resolved);
        }
      }
    }

    function render({ model, el }) {
      const media = window.matchMedia("(prefers-color-scheme: dark)");

      const wrapper = document.createElement("label");
      wrapper.className = "cxr-theme-select";

      const caption = document.createElement("span");
      caption.textContent = "Appearance";
      caption.className = "cxr-theme-select__label";

      const select = document.createElement("select");
      select.className = "cxr-theme-select__control";
      select.setAttribute("aria-label", "Appearance");
      select.style.backgroundColor = "var(--cxr-surface, #FFFFFF)";
      select.style.color = "var(--cxr-text, #172832)";
      select.style.border = "1px solid var(--cxr-rule, #C6D2D8)";
      select.style.borderRadius = "3px";
      select.style.padding = ".25rem 1.8rem .25rem .55rem";
      for (const [value, label] of [
        ["system", "System"],
        ["light", "Light"],
        ["dark", "Dark"],
      ]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        select.appendChild(option);
      }

      wrapper.append(caption, select);
      el.replaceChildren(wrapper);

      function apply(preference, { persist = false } = {}) {
        if (!VALID.has(preference)) preference = "light";
        const resolved = resolvedTheme(preference, media);

        select.value = preference;
        document.documentElement.dataset.theme = resolved;
        document.documentElement.dataset.pyriteTheme = resolved;
        document.documentElement.dataset.pyriteThemePreference = preference;
        document.documentElement.style.colorScheme = resolved;
        if (document.body) {
          // marimo exposes its resolved mode on body[data-theme].  Updating the
          // same attribute lets marimo controls and third-party outputs consume
          // the same resolved PyRITE appearance.
          document.body.dataset.theme = resolved;
          document.body.dataset.pyriteTheme = resolved;
        }
        wrapper.style.color = "var(--cxr-muted, #526B76)";
        syncMarimoShadowThemes(resolved);
        window.requestAnimationFrame(() => syncMarimoShadowThemes(resolved));

        if (persist) safeWrite(preference);

        let changed = false;
        if (model.get("preference") !== preference) {
          model.set("preference", preference);
          changed = true;
        }
        if (model.get("resolved") !== resolved) {
          model.set("resolved", resolved);
          changed = true;
        }
        if (changed) model.save_changes();

        window.dispatchEvent(
          new CustomEvent("pyrite-theme-change", {
            detail: { preference, resolved },
          }),
        );
      }

      const stored = safeRead();
      const initial =
        stored ??
        (VALID.has(model.get("preference")) ? model.get("preference") : "light");
      apply(initial);

      const onSelect = () => apply(select.value, { persist: true });
      const onStorage = (event) => {
        if (event.key !== STORAGE_KEY) return;
        apply(VALID.has(event.newValue) ? event.newValue : "system");
      };
      const onSystemTheme = () => apply(select.value);
      const syncTimer = window.setInterval(() => {
        const shared = cookieRead();
        if (shared && shared !== select.value) {
          apply(shared);
        } else {
          syncMarimoShadowThemes(resolvedTheme(select.value, media));
        }
      }, 750);
      const bodyObserver = new MutationObserver(() => {
        const expected = resolvedTheme(select.value, media);
        if (document.body?.dataset.theme !== expected) {
          apply(select.value);
        } else {
          window.requestAnimationFrame(() => syncMarimoShadowThemes(expected));
        }
      });

      select.addEventListener("change", onSelect);
      window.addEventListener("storage", onStorage);
      media.addEventListener("change", onSystemTheme);
      if (document.body) {
        bodyObserver.observe(document.body, {
          attributes: true,
          childList: true,
          subtree: true,
          attributeFilter: ["data-theme"],
        });
      }

      return () => {
        select.removeEventListener("change", onSelect);
        window.removeEventListener("storage", onStorage);
        media.removeEventListener("change", onSystemTheme);
        window.clearInterval(syncTimer);
        bodyObserver.disconnect();
      };
    }

    export default { render };
    """

    preference = traitlets.Enum(
        values=("system", "light", "dark"), default_value="light"
    ).tag(sync=True)
    resolved = traitlets.Enum(values=("light", "dark"), default_value="light").tag(sync=True)
