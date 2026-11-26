"""Find duplicated function bodies across tracked Python files.

Compares functions by AST after normalizing the function name, docstrings,
parameters and local variable names, so copies with renamed locals still match.
Globals, attributes and called names are kept, so functions doing the same
thing against different collaborators stay distinct.

Usage (from the repository root):
    pixi run python checks/code_duplication.py [ROOTS...] [--min-nodes N]
        [--include-marimo] [--include-tests]

Defaults: ROOTS=src, min-nodes=40, marimo cells and lazy ``__getattr__``
shims skipped. Report-only engineering diagnostic (#1467): always exits 0,
no physics claim. For copied blocks inside functions, pair it with
``pixi exec --spec pylint symilar -d 12 -i --ignore-imports --ignore-signatures``
over ``git ls-files -z 'src/*.py'``.
"""

import argparse
import ast
import copy
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

SKIP_NAMES = {"__getattr__", "__dir__"}


def tracked_files(roots):
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots],
        check=True,
        capture_output=True,
    ).stdout.decode()
    return [Path(p) for p in out.split("\0") if p.endswith(".py")]


def is_marimo_cell(node):
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr in {"cell", "function"}:
            if isinstance(target.value, ast.Name) and target.value.id == "app":
                return True
    return False


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)


class LocalRenamer(ast.NodeTransformer):
    """Rename parameters and assigned locals to positional placeholders."""

    def __init__(self, local_names):
        self.mapping = {n: f"v{i}" for i, n in enumerate(local_names)}

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        node.annotation = None
        return node


def local_names(func):
    names = [a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)]
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
            names.append(node.name)
    seen = {}
    for n in names:
        seen.setdefault(n, None)
    return list(seen)


def fingerprint(func):
    func = copy.deepcopy(func)
    func.name = "f"
    func.decorator_list = []
    func.returns = None
    strip_docstrings(func)
    func = LocalRenamer(local_names(func)).visit(func)
    return ast.dump(func, annotate_fields=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["src"])
    ap.add_argument("--min-nodes", type=int, default=40)
    ap.add_argument("--include-marimo", action="store_true")
    ap.add_argument("--include-tests", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(list)
    for path in tracked_files(args.roots):
        if not args.include_tests and path.parts[0] == "tests":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in SKIP_NAMES:
                continue
            if not args.include_marimo and is_marimo_cell(node):
                continue
            size = sum(1 for _ in ast.walk(node))
            if size < args.min_nodes:
                continue
            groups[fingerprint(node)].append((size, f"{path}:{node.lineno} ({node.name})"))

    dups = [v for v in groups.values() if len(v) > 1]
    dups.sort(key=lambda v: -v[0][0] * (len(v) - 1))
    total = 0
    for locs in dups:
        size = locs[0][0]
        total += size * (len(locs) - 1)
        print(f"[{size} nodes x{len(locs)}]")
        for _, loc in locs:
            print(f"  - {loc}")
    print(f"\n{len(dups)} groups, ~{total} redundant AST nodes")


if __name__ == "__main__":
    main()
