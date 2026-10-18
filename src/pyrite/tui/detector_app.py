"""Textual detector editor (prototype) over :class:`DetectorSession`.

Requires the optional ``tui`` extra. Every screen is keyboard-operable and
legible without color; errors carry an ``error:`` prefix. Typing, cancelling
and failed validation never write: only Save on the review screen does.
"""

from collections.abc import Callable
from itertools import groupby
from shlex import join

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from .detector_form import BY_KEY, FIELDS, KINDS, LABEL, DetectorSession, FormError, parse_field

HELP = """\
PyRITE detector editor (prototype)

Browse:  up/down select   enter or e edit   n new   c clone   ? help   q quit
Editor:  tab/shift+tab move between fields   ctrl+r review the diff   escape cancel
Review:  ctrl+s save   escape back to the editor

Bundled detectors are read-only: clone one (c) into the writable layer.
Fields match `pyrite detector create scalar|planar` options; each label shows
its unit and CLI flag. Pairs take two numbers ("0.055 0.055"); measured edges
are comma-separated. Leave a field blank to use the model default.

Nothing is written until you save a reviewed diff. Clearing an existing
optional section is not supported here: use `pyrite detector reset`.
Plotting, runs and other catalog objects stay in the CLI and marimo apps.
"""

_TITLES = {"create": "New detector", "clone": "Clone {source}", "edit": "Edit {name}"}


def _plain(text) -> Content:
    return Content(str(text))


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,q,question_mark", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help"):
            yield Static(HELP, markup=False)


class ReviewScreen(Screen[bool]):
    """Show the canonical diff and equivalent CLI call; save only on request."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, session: DetectorSession, staged) -> None:
        super().__init__()
        self.session = session
        self.staged = staged

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Proposed catalog change (nothing saved yet)", classes="section")
        with VerticalScroll(id="diff"):
            yield Static(self.staged.diff() or "(no textual change)", markup=False)
        commands = "\n  ".join(join(["pyrite", *args]) for args in self.staged.cli_commands())
        yield Static(f"CLI equivalent:\n  {commands}", classes="cli", markup=False)
        yield Static("", id="message", markup=False)
        with Horizontal(classes="buttons"):
            yield Button("Save (ctrl+s)", id="save", variant="primary")
            yield Button("Back (esc)", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        else:
            self.action_back()

    def action_save(self) -> None:
        try:
            self.session.save(self.staged)
        except FormError as exc:
            self.query_one("#message", Static).update(f"error: {exc}")
            return
        self.dismiss(True)

    def action_back(self) -> None:
        self.dismiss(False)


class EditorScreen(Screen[str | None]):
    """Conditional scalar/planar form; hidden fields keep their typed values."""

    BINDINGS = [
        Binding("ctrl+r", "review", "Review diff", priority=True),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, session, mode, *, name="", source=None) -> None:
        super().__init__()
        self.session = session
        self.mode = mode
        self.source = source
        base = source if mode == "clone" else name
        self.initial = session.form(base) if base else {}
        self.kind = "scalar"
        if base:
            self.kind = next(e.kind for e in session.entries() if e.name == base)
        self.default_name = f"{source}_copy" if mode == "clone" else name

    def compose(self) -> ComposeResult:
        yield Header()
        title = _TITLES[self.mode].format(source=self.source, name=self.default_name)
        yield Label(f"{title} ({self.kind if self.mode != 'create' else 'choose type'})")
        with VerticalScroll(id="form"):
            yield Label("Name", markup=False)
            yield Input(self.default_name, id="f-name", disabled=self.mode == "edit")
            if self.mode == "create":
                yield Label("Type", markup=False)
                yield Select([(kind, kind) for kind in KINDS], value=self.kind,
                             allow_blank=False, id="f-kind")  # fmt: skip
            yield Label("Label (display only)", markup=False)
            yield Input(self.initial.get(LABEL, ""), id=f"f-{LABEL}")
            for (kind, section), fields in groupby(FIELDS, lambda f: (f.kind, f.section)):
                with Vertical(classes=f"group kind-{kind}"):
                    yield Label(section, classes="section", markup=False)
                    for spec in fields:
                        yield from self._field(spec)
        yield Static("", id="message", markup=False)
        with Horizontal(classes="buttons"):
            yield Button("Review diff (ctrl+r)", id="review", variant="primary")
            yield Button("Cancel (esc)", id="cancel")
        yield Footer()

    def _field(self, spec):
        unit = f" [{spec.unit}]" if spec.unit else ""
        yield Label(f"{spec.label}{unit}  {spec.flag}", markup=False)
        value = self.initial.get(spec.key, "")
        if spec.choices:
            yield Select([(c, c) for c in spec.choices], value=value or Select.NULL,
                         id=f"f-{spec.key}", tooltip=spec.help)  # fmt: skip
        else:
            yield Input(value, placeholder=spec.help, id=f"f-{spec.key}", tooltip=spec.help)
        yield Static("", id=f"e-{spec.key}", classes="error", markup=False)

    def on_mount(self) -> None:
        self._show_kind()
        self.query_one("#f-name" if self.mode != "edit" else f"#f-{LABEL}").focus()

    def _show_kind(self) -> None:
        for group in self.query(".group"):
            group.display = group.has_class(f"kind-{self.kind}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "f-kind" and event.value in KINDS:
            self.kind = str(event.value)
            self._show_kind()

    def on_input_changed(self, event: Input.Changed) -> None:
        key = (event.input.id or "").removeprefix("f-")
        if key in BY_KEY:
            try:
                parse_field(key, event.value)
                message = ""
            except FormError as exc:
                message = f"error: {exc}"
            self.query_one(f"#e-{key}", Static).update(message)

    def values(self) -> dict[str, str]:
        values = {LABEL: self.query_one(f"#f-{LABEL}", Input).value}
        for spec in FIELDS:
            if spec.choices:
                value = self.query_one(f"#f-{spec.key}", Select).value
                values[spec.key] = "" if value is Select.NULL else str(value)
            else:
                values[spec.key] = self.query_one(f"#f-{spec.key}", Input).value
        return values

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review":
            self.action_review()
        else:
            self.action_cancel()

    def action_review(self) -> None:
        for error in self.query(".error").results(Static):
            error.update("")
        name = self.query_one("#f-name", Input).value
        try:
            staged = self.session.stage(
                self.mode,
                name,
                self.values(),
                kind=self.kind,
                source=self.source,
                initial=self.initial,
            )
        except FormError as exc:
            for key, message in exc.fields.items():
                self.query_one(f"#e-{key}", Static).update(f"error: {message}")
            self.query_one("#message", Static).update(f"error: {exc}")
            return
        self.query_one("#message", Static).update("")

        def reviewed(saved: bool | None) -> None:
            if saved:
                self.dismiss(staged.name)

        self.app.push_screen(ReviewScreen(self.session, staged), reviewed)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BrowseScreen(Screen[None]):
    """Detector list with kind, source layer and read-only state."""

    BINDINGS = [
        Binding("n", "new", "New"),
        Binding("c", "clone", "Clone"),
        Binding("e", "edit", "Edit"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "app.quit", "Quit"),
    ]

    def __init__(self, session: DetectorSession) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"catalog: {self.session.path}", markup=False)
        yield OptionList(id="detectors")
        yield Static("", id="details", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_list()

    def refresh_list(self, select: str | None = None) -> None:
        options = self.query_one("#detectors", OptionList)
        options.clear_options()
        entries = self.session.entries()
        for entry in entries:
            mode = "" if entry.writable else ", read-only"
            label = f"  {entry.label}" if entry.label else ""
            prompt = f"{entry.name}  ({entry.kind}, {entry.source}{mode}){label}"
            options.add_option(Option(_plain(prompt), id=entry.name))
        names = [entry.name for entry in entries]
        if names:
            options.highlighted = names.index(select) if select in names else 0
        options.focus()

    def selected(self):
        options = self.query_one("#detectors", OptionList)
        if options.highlighted is None:
            return None
        name = options.get_option_at_index(options.highlighted).id
        return next(entry for entry in self.session.entries() if entry.name == name)

    def on_option_list_option_highlighted(self, _event) -> None:
        entry = self.selected()
        if entry is None:
            return
        values = self.session.form(entry.name)
        lines = [f"{entry.name}: {entry.kind} detector from the {entry.source} layer"]
        for spec in FIELDS:
            if values.get(spec.key):
                lines.append(f"  {spec.label}: {values[spec.key]} {spec.unit}".rstrip())
        if not entry.writable:
            lines.append("read-only: press c to clone it into the writable layer")
        self.query_one("#details", Static).update("\n".join(lines))

    def on_option_list_option_selected(self, _event) -> None:
        self.action_edit()

    def _open(self, screen: EditorScreen, verb: str) -> None:
        def done(name: str | None) -> None:
            self.session.reload()
            self.refresh_list(select=name)
            if name is None:
                self.notify(f"{verb} cancelled; nothing written", markup=False)
            else:
                self.notify(f"saved detector {name}", markup=False)

        self.app.push_screen(screen, done)

    def action_new(self) -> None:
        self._open(EditorScreen(self.session, "create"), "create")

    def action_clone(self) -> None:
        entry = self.selected()
        if entry is not None:
            self._open(EditorScreen(self.session, "clone", source=entry.name), "clone")

    def action_edit(self) -> None:
        entry = self.selected()
        if entry is None:
            return
        if not entry.writable:
            self.notify(
                f"{entry.name} is bundled and read-only; press c to clone it",
                severity="warning",
                markup=False,
            )
            return
        self._open(EditorScreen(self.session, "edit", name=entry.name), "edit")

    def action_help(self) -> None:
        self.app.push_screen(HelpScreen())


class DetectorEditorApp(App[None]):
    TITLE = "PyRITE detector editor"
    SUB_TITLE = "prototype"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    #detectors { height: 1fr; min-height: 4; }
    #details { height: auto; max-height: 12; border-top: solid $secondary; }
    #form, #diff, #help { height: 1fr; }
    #help { background: $panel; padding: 1 2; }
    .section { text-style: bold; margin-top: 1; }
    .group { height: auto; }
    .error { color: $error; height: auto; }
    .buttons { height: auto; }
    Input, Select { width: 1fr; }
    """

    def __init__(self, path) -> None:
        super().__init__()
        self.session = DetectorSession(path)

    def on_mount(self) -> None:
        self.push_screen(BrowseScreen(self.session))


def run(path, *, app_factory: Callable = DetectorEditorApp) -> int:
    """Run the editor on ``path``; return ``130`` on interruption, else ``0``."""
    try:
        app_factory(path).run()
    except KeyboardInterrupt:
        return 130
    return 0
