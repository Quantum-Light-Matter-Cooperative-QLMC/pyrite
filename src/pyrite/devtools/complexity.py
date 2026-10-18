"""Source complexity inventory and high-CC ratchet behind ``pyrite-dev complexity``.

Radon supplies per-routine cyclomatic complexity (CC) and the informational
maintainability index (MI); ``scc`` optionally supplies a language-aware line
inventory, and ``git log`` optional churn. Only the CC ratchet gates: a routine
above :data:`CC_THRESHOLD` must match a baseline entry with an equal or higher
CC. See ``docs/repo-design/complexity.md`` for the policy and limitations.
"""

import ast
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CC_THRESHOLD = 20
"""Ratchet threshold: Radon ranks D-F start above CC 20."""

NEAR_BUDGET_LINES = 1_100
BASELINE_SCHEMA = 1
BASELINE_PATH = Path("docs/repo-design/complexity-baseline.json")
SCOPES = {"src": Path("src/pyrite"), "tests": Path("tests")}
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        ".worktrees",
        "__pycache__",
        "_build",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "vendor",
        "venv",
    }
)
SCC_ENV = "PYRITE_SCC"
_CC_RANKS = ((5, "A"), (10, "B"), (20, "C"), (30, "D"), (40, "E"))
_MISSING_RADON = (
    "radon is not installed; it is pinned in the `lint` dependency group. "
    "Run `uv sync` (default groups) or `uv sync --group lint`, then retry."
)


class ComplexityToolError(RuntimeError):
    """Raised for a missing tool, unreadable baseline, or unusable scan."""


@dataclass(frozen=True, order=True)
class Routine:
    """One function, method, or closure with its Radon cyclomatic complexity."""

    path: str
    lineno: int
    col: int
    name: str
    cc: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.name)

    def describe(self) -> str:
        return f"{self.path}:{self.lineno} {self.name} CC {self.cc}"


@dataclass(frozen=True)
class ParseFailure:
    path: str
    message: str


@dataclass
class Scan:
    """Deterministic per-scope scan result; lists are sorted by path/position."""

    scope: str
    routines: list[Routine] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    module_lines: dict[str, int] = field(default_factory=dict)
    maintainability: dict[str, float] = field(default_factory=dict)

    def hotspots(self, threshold: int = CC_THRESHOLD) -> list[Routine]:
        return sorted(
            (r for r in self.routines if r.cc > threshold), key=lambda r: (-r.cc, r.path, r.lineno)
        )


@dataclass
class Ratchet:
    """Comparison of a scan's hotspots against the committed baseline."""

    new: list[Routine] = field(default_factory=list)
    worsened: list[tuple[Routine, int]] = field(default_factory=list)
    improved: list[tuple[Routine, int]] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    unchanged: int = 0

    @property
    def failed(self) -> bool:
        return bool(self.new or self.worsened)


def radon_modules() -> tuple[str, Any, Any]:
    """Return Radon's version, ``ComplexityVisitor`` and ``mi_visit``, or fail actionably."""
    try:
        import radon
        from radon.metrics import mi_visit
        from radon.visitors import ComplexityVisitor
    except ImportError as exc:
        raise ComplexityToolError(_MISSING_RADON) from exc
    return radon.__version__, ComplexityVisitor, mi_visit


def cc_rank(cc: int) -> str:
    return next((rank for bound, rank in _CC_RANKS if cc <= bound), "F")


def mi_rank(mi: float) -> str:
    return "A" if mi > 19 else "B" if mi > 9 else "C"


def iter_sources(root: Path) -> list[Path]:
    """Return sorted Python files under ``root``, skipping vendor/build/env trees."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
    )


def routines_in_source(source: str, path: str, cc_of: Callable[[ast.AST], int]) -> list[Routine]:
    """Return every def (methods, closures, defs in nested classes) once by position.

    Names are dotted lexical scopes (``Class.method``, ``outer.inner``); a repeat
    of the same name in one module, such as a property setter or a conditional
    redefinition, gets ``#2``, ``#3`` in source order so baseline keys stay unique.
    """
    tree = ast.parse(source, filename=path)
    found: dict[tuple[int, int], tuple[str, int]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = (*scope, child.name)
                found.setdefault((child.lineno, child.col_offset), (".".join(name), cc_of(child)))
                visit(child, name)
            elif isinstance(child, ast.ClassDef):
                visit(child, (*scope, child.name))
            else:
                visit(child, scope)

    visit(tree, ())
    seen: Counter[str] = Counter()
    routines = []
    for (lineno, col), (name, cc) in sorted(found.items()):
        seen[name] += 1
        unique = name if seen[name] == 1 else f"{name}#{seen[name]}"
        routines.append(Routine(path, lineno, col, unique, cc))
    return routines


def _radon_cc(visitor_cls: Any) -> Callable[[ast.AST], int]:
    def cc_of(node: ast.AST) -> int:
        visitor = visitor_cls(off=False)
        visitor.visit(node)
        return int(visitor.functions[0].complexity)

    return cc_of


def scan(root: Path, scope: str = "src", *, maintainability: bool = True) -> Scan:
    """Scan one scope of a checkout with Radon; parse failures are recorded, not raised."""
    _, visitor_cls, mi_visit = radon_modules()
    cc_of = _radon_cc(visitor_cls)
    scope_root = root / SCOPES[scope]
    if not scope_root.is_dir():
        raise ComplexityToolError(f"scope {scope!r} root {scope_root} does not exist")
    result = Scan(scope=SCOPES[scope].as_posix())
    for source_path in iter_sources(scope_root):
        rel = source_path.relative_to(root).as_posix()
        try:
            text = source_path.read_text(encoding="utf-8")
            result.routines.extend(routines_in_source(text, rel, cc_of))
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            result.failures.append(ParseFailure(rel, f"{type(exc).__name__}: {exc}"))
            continue
        result.module_lines[rel] = len(text.splitlines())
        if maintainability:
            result.maintainability[rel] = round(float(mi_visit(text, multi=True)), 2)
    result.routines.sort()
    return result


def load_baseline(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ComplexityToolError(
            f"complexity baseline {path} is missing; create it with `pyrite-dev complexity --write`"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ComplexityToolError(f"complexity baseline {path} is not valid JSON: {exc}") from exc
    if data.get("schema") != BASELINE_SCHEMA:
        raise ComplexityToolError(
            f"complexity baseline {path} has schema {data.get('schema')!r}; "
            f"expected {BASELINE_SCHEMA}"
        )
    return data


def compare(baseline: dict[str, Any], current: Scan) -> Ratchet:
    """Classify current hotspots against baseline entries keyed by (path, name)."""
    threshold = int(baseline["threshold"])
    entries = {(e["path"], e["name"]): e for e in baseline["routines"]}
    ratchet = Ratchet()
    matched = set()
    for routine in sorted(current.hotspots(threshold)):
        entry = entries.get(routine.key)
        if entry is None:
            ratchet.new.append(routine)
            continue
        matched.add(routine.key)
        if routine.cc > entry["cc"]:
            ratchet.worsened.append((routine, entry["cc"]))
        elif routine.cc < entry["cc"]:
            ratchet.improved.append((routine, entry["cc"]))
        else:
            ratchet.unchanged += 1
    ratchet.resolved = [e for key, e in sorted(entries.items()) if key not in matched]
    return ratchet


def move_candidates(routine: Routine, resolved: Iterable[dict[str, Any]]) -> list[str]:
    """Name resolved baseline entries a new hotspot may have been moved or renamed from."""
    leaf = routine.name.split("#")[0].rsplit(".", 1)[-1]
    return [
        f"{e['path']}::{e['name']}"
        for e in resolved
        if e["name"].split("#")[0].rsplit(".", 1)[-1] == leaf or e["cc"] == routine.cc
    ]


def ratchet_problems(baseline: dict[str, Any], current: Scan, radon_version: str) -> list[str]:
    """Return actionable gate failures; an empty list means the ratchet passes."""
    problems = [f"parse failure: {f.path}: {f.message}" for f in current.failures]
    if baseline.get("tool", {}).get("radon") != radon_version:
        problems.append(
            f"baseline was produced with radon {baseline.get('tool', {}).get('radon')}, "
            f"installed radon is {radon_version}; run `uv sync` to restore the pinned version"
        )
    if baseline.get("scope") != current.scope:
        problems.append(f"baseline scope {baseline.get('scope')!r} is not {current.scope!r}")
    if problems:
        return problems
    ratchet = compare(baseline, current)
    for routine in ratchet.new:
        moved = move_candidates(routine, ratchet.resolved)
        hint = f"; possibly moved/renamed from {', '.join(moved)}" if moved else ""
        problems.append(f"new hotspot {routine.describe()} > {baseline['threshold']}{hint}")
    for routine, old in ratchet.worsened:
        problems.append(f"worsened hotspot {routine.describe()} (baseline {old})")
    return problems


def baseline_document(
    current: Scan, radon_version: str, previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the canonical baseline, keeping documented exceptions for surviving keys."""
    notes = {
        (e["path"], e["name"]): e["exception"]
        for e in (previous or {}).get("routines", [])
        if e.get("exception")
    }
    routines = []
    for routine in sorted(current.hotspots(), key=lambda r: r.key):
        entry: dict[str, Any] = {"path": routine.path, "name": routine.name, "cc": routine.cc}
        if routine.key in notes:
            entry["exception"] = notes[routine.key]
        routines.append(entry)
    return {
        "schema": BASELINE_SCHEMA,
        "scope": current.scope,
        "threshold": CC_THRESHOLD,
        "tool": {"radon": radon_version},
        "routines": routines,
    }


def write_baseline(path: Path, current: Scan, radon_version: str) -> list[str]:
    """Rewrite ``path`` from ``current``; return the change summary for review."""
    if current.failures:
        raise ComplexityToolError(
            "refusing to write a baseline with parse failures: "
            + ", ".join(f"{f.path} ({f.message})" for f in current.failures)
        )
    previous = load_baseline(path) if path.exists() else None
    document = baseline_document(current, radon_version, previous)
    changes = []
    if previous is not None:
        ratchet = compare(previous, current)
        changes += [f"added {r.describe()}" for r in ratchet.new]
        changes += [f"raised {r.describe()} (was {old})" for r, old in ratchet.worsened]
        changes += [f"lowered {r.describe()} (was {old})" for r, old in ratchet.improved]
        changes += [
            f"removed {e['path']}::{e['name']} (was CC {e['cc']})" for e in ratchet.resolved
        ]
        lost = [e for e in ratchet.resolved if e.get("exception")]
        changes += [f"dropped exception note for {e['path']}::{e['name']}" for e in lost]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return changes


def scc_inventory(root: Path) -> dict[str, Any]:
    """Return scc's Python line inventory, or ``{"error": ...}`` when unavailable."""
    binary = os.environ.get(SCC_ENV) or shutil.which("scc")
    if not binary:
        return {"error": f"scc not found on PATH; set {SCC_ENV}=/path/to/scc to add it"}
    try:
        version = (
            subprocess.run(
                [binary, "--version"], capture_output=True, text=True, check=True, timeout=60
            )
            .stdout.strip()
            .removeprefix("scc version ")
        )
        proc = subprocess.run(
            [
                binary,
                "--format",
                "json",
                "--no-cocomo",
                "--include-ext",
                "py",
                "--exclude-dir",
                ",".join(sorted(EXCLUDED_PARTS)),
                str(root),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        languages = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        first = detail.strip().splitlines()[0] if detail.strip() else type(exc).__name__
        return {
            "error": f"{binary} failed ({first}); point {SCC_ENV} at the installed scc "
            "binary rather than a launcher wrapper"
        }
    fields = ("Count", "Lines", "Code", "Comment", "Blank", "Complexity")
    python = next((lang for lang in languages if lang.get("Name") == "Python"), {})
    return {"version": version, **{name.lower(): int(python.get(name, 0)) for name in fields}}


def git_churn(root: Path, scope_root: Path, commits: int) -> dict[str, Any]:
    """Return added+deleted lines per file over the last ``commits`` commits in scope."""
    try:
        proc = subprocess.run(
            ["git", "log", f"-n{commits}", "--no-renames", "--numstat", "--format=", "--"]
            + [scope_root.relative_to(root).as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"git log failed: {exc}"}
    churn: Counter[str] = Counter()
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            churn[parts[2]] += int(parts[0]) + int(parts[1])
    top = sorted(churn.items(), key=lambda item: (-item[1], item[0]))
    return {"commits": commits, "files": dict(top)}


def _git_head(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def build_report(
    root: Path, current: Scan, radon_version: str, *, top: int, churn: int | None = None
) -> dict[str, Any]:
    """Assemble the deterministic report dictionary rendered as text or JSON."""
    ccs = [r.cc for r in current.routines]
    ranks = Counter(cc_rank(cc) for cc in ccs)
    mi_ranks = Counter(mi_rank(mi) for mi in current.maintainability.values())
    report: dict[str, Any] = {
        "provenance": {
            "scope": current.scope,
            "git_head": _git_head(root),
            "radon": radon_version,
            "python": platform.python_version(),
            "threshold": CC_THRESHOLD,
            "excluded_parts": sorted(EXCLUDED_PARTS),
        },
        "inventory": {
            "modules": len(current.module_lines),
            "physical_lines": sum(current.module_lines.values()),
            f"modules_at_or_over_{NEAR_BUDGET_LINES}_lines": sum(
                n >= NEAR_BUDGET_LINES for n in current.module_lines.values()
            ),
            "scc": scc_inventory(root / current.scope),
        },
        "cc": {
            "routines": len(ccs),
            "mean": round(sum(ccs) / len(ccs), 2) if ccs else 0.0,
            "ranks": {rank: ranks.get(rank, 0) for rank in "ABCDEF"},
            "hotspots": [r.describe() for r in current.hotspots()[:top]],
        },
        "mi": {
            "ranks": {rank: mi_ranks.get(rank, 0) for rank in "ABC"},
            "lowest": [
                f"{path} MI {mi:g}"
                for path, mi in sorted(current.maintainability.items(), key=lambda i: (i[1], i[0]))[
                    :top
                ]
            ],
        },
        "parse_failures": [f"{f.path}: {f.message}" for f in current.failures],
    }
    if churn:
        history = git_churn(root, root / current.scope, churn)
        if "files" in history:
            history["files"] = dict(list(history["files"].items())[:top])
        report["churn"] = history
    return report


def render_report(report: dict[str, Any]) -> str:
    prov, inv, cc, mi = report["provenance"], report["inventory"], report["cc"], report["mi"]
    lines = [
        f"scope {prov['scope']} @ {prov['git_head'] or 'unknown'}; radon {prov['radon']}, "
        f"python {prov['python']}",
        f"modules {inv['modules']}, physical lines {inv['physical_lines']}, "
        f">= {NEAR_BUDGET_LINES} lines: {inv[f'modules_at_or_over_{NEAR_BUDGET_LINES}_lines']}",
    ]
    scc = inv["scc"]
    if "error" in scc:
        lines.append(f"scc: unavailable ({scc['error']})")
    else:
        lines.append(
            f"scc {scc['version']}: {scc['count']} files, {scc['code']} code / "
            f"{scc['comment']} comment / {scc['blank']} blank lines, complexity {scc['complexity']}"
        )
    ranks = " ".join(f"{k} {v}" for k, v in cc["ranks"].items())
    lines.append(f"CC: {cc['routines']} routines, mean {cc['mean']}; {ranks}")
    lines.append(f"top hotspots (CC > {prov['threshold']}):")
    lines += [f"  {item}" for item in cc["hotspots"]] or ["  none"]
    lines.append("MI (informational): " + " ".join(f"{k} {v}" for k, v in mi["ranks"].items()))
    lines += [f"  {item}" for item in mi["lowest"]]
    if "churn" in report:
        history = report["churn"]
        if "error" in history:
            lines.append(f"churn: unavailable ({history['error']})")
        else:
            lines.append(f"churn over {history['commits']} commits (informational):")
            lines += [f"  {path} {count}" for path, count in history["files"].items()]
    lines.append(f"parse failures: {len(report['parse_failures'])}")
    lines += [f"  {item}" for item in report["parse_failures"]]
    return "\n".join(lines)


def run(
    root: Path,
    *,
    scope: str = "src",
    check: bool = False,
    write: bool = False,
    as_json: bool = False,
    top: int = 15,
    churn: int | None = None,
    baseline: Path | None = None,
) -> int:
    """Execute one ``pyrite-dev complexity`` invocation; return its exit status."""
    radon_version = radon_modules()[0]
    baseline_path = root / (baseline or BASELINE_PATH)
    if (check or write) and scope != "src":
        raise ComplexityToolError("the ratchet covers --scope src only; tests are report-only")
    current = scan(root, scope, maintainability=not (check or write))
    if write:
        changes = write_baseline(baseline_path, current, radon_version)
        print(f"wrote {baseline_path.relative_to(root)} ({len(current.hotspots())} hotspots)")
        print("\n".join(changes) if changes else "no baseline changes")
        return 0
    if check:
        problems = ratchet_problems(load_baseline(baseline_path), current, radon_version)
        if problems:
            print("complexity ratchet failed:")
            print("\n".join(f"  {problem}" for problem in problems))
            print(
                "Simplify the routine, or record it with `pyrite-dev complexity --write` and "
                "justify the baseline diff in review (docs/repo-design/complexity.md)."
            )
            return 1
        ratchet = compare(load_baseline(baseline_path), current)
        stale = len(ratchet.improved) + len(ratchet.resolved)
        print(f"complexity ratchet passed ({ratchet.unchanged} unchanged hotspots)")
        if stale:
            print(f"{stale} baseline entries improved or resolved; lock in with --write")
        return 0
    report = build_report(root, current, radon_version, top=top, churn=churn)
    print(json.dumps(report, indent=2) if as_json else render_report(report))
    return 1 if current.failures else 0
