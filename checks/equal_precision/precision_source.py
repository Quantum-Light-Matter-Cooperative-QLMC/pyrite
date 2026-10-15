"""Generate isolated radiative helpers for opt-in precision diagnostics."""

import ast

VARIANTS = (
    "fp64",
    "rad-fp32-52",
    "rad-fp32-24",
    "rad-cell64-24",
    "rad-stable32-24",
    "rad-boundary64-24",
    "rad-quotient32-24",
)


def _cast_parameters(function, names, dtype):
    statements = []
    for argument in function.args.args:
        if argument.arg in names:
            name = argument.arg
            argument.arg += "_input"
            statements.extend(ast.parse(f"{name} = np.{dtype}({name}_input)").body)
    offset = (
        1
        if isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
        else 0
    )
    function.body[offset:offset] = statements


def precision_source(source, variant):
    """Return an experimental source copy; never mutate the reference module."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown precision variant: {variant}")
    if variant == "fp64":
        return source
    if variant == "rad-quotient32-24":
        return _quotient_source(source)
    if variant == "rad-boundary64-24":
        return _boundary_source(source)
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    expected = {"_rad_moment", "_rad_layer_moment", "_rad_sample_photon_eV", "_rad_cell_rate"}
    if not expected <= functions.keys():
        raise ValueError("radiative source no longer matches the precision experiment")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr == "float64"
        ):
            node.attr = "float32"
    # This exported converter is also imported by the FP64 transport kernel.
    # Keep that interface double; only the radiative helpers use its FP32 twin.
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "F64_KEV_PER_EV" for t in node.targets
        ):
            node.targets[0].id = "F32_KEV_PER_EV"
            tree.body[index + 1 : index + 1] = ast.parse("F64_KEV_PER_EV = np.float64(1e-3)").body
            break
    for function in functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and node.id == "F64_KEV_PER_EV":
                node.id = "F32_KEV_PER_EV"
    shared = {"F64_HALF": "0.5", "F64_INF": "np.inf", "F64_ONE": "1.0", "F64_ZERO": "0.0"}
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "_jit_device":
            imported = {a.name for a in node.names}
            if not shared.keys() <= imported:
                raise ValueError("radiative shared constants changed")
            node.names = [a for a in node.names if a.name not in shared]
            tree.body[index + 1 : index + 1] = ast.parse(
                "\n".join(f"{name} = np.float32({value})" for name, value in shared.items())
            ).body
            break
    else:
        raise ValueError("radiative shared constants import missing")
    _cast_parameters(functions["_rad_moment"], {"energy_eV", "cutoff_eV", "z"}, "float32")
    _cast_parameters(functions["_rad_layer_moment"], {"energy_eV", "cutoff_eV"}, "float32")
    _cast_parameters(
        functions["_rad_sample_photon_eV"], {"energy_eV", "cutoff_eV", "z", "uniform"}, "float32"
    )
    if variant != "rad-fp32-52":
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "I32_BISECTION_STEPS" for t in node.targets
            ):
                node.value = ast.parse("np.int32(24)", mode="eval").body
    if variant == "rad-stable32-24":
        # Same linear-cell integral, rearranged around log1p(width/left).
        # For x<=.01, the omitted term of x-log1p(x) is <=x**7/7,
        # below FP32 rounding of the correction; no physics/tail is truncated.
        functions["_rad_cell_rate"].body = ast.parse("""
if right <= left:
    return F64_ZERO
width = right - left
x = width / left
log_ratio = xp.log1p(x)
correction = x - log_ratio
if x <= np.float32(.01):
    correction = x*x*(np.float32(.5)+x*(np.float32(-1/3)+x*(np.float32(.25)+x*(np.float32(-.2)+x*np.float32(1/6)))))
slope = (chi_right-chi_left)/width
return chi_left*log_ratio + slope*left*correction
""").body
    if variant == "rad-cell64-24":
        cell = functions["_rad_cell_rate"]
        _cast_parameters(cell, {"left", "right", "chi_left", "chi_right"}, "float64")
        cell.body[-1].value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load()
            ),
            args=[cell.body[-1].value],
            keywords=[],
        )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _boundary_source(source):
    """Keep coordinates double; SDCS interpolation and moment/cell work single."""
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "F64_MB_CM2" in names:
                node.value = ast.parse("np.float32(_MB_CM2)", mode="eval").body
            if "I32_BISECTION_STEPS" in names:
                node.value = ast.parse("np.int32(24)", mode="eval").body
    _cast_parameters(functions["_rad_chi"], {"fraction"}, "float32")
    for name in ["_rad_moment", "_rad_sample_photon_eV"]:
        _cast_parameters(functions[name], {"z"}, "float32")
    _cast_parameters(functions["_rad_sample_photon_eV"], {"uniform"}, "float32")
    for function in functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "weight",
                    "scale",
                    "soft",
                    "cumulative",
                    "chi_mid",
                }:
                    node.value = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load()
                        ),
                        args=[node.value],
                        keywords=[],
                    )
            if (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "soft"
            ):
                node.value = ast.parse(
                    "scale * np.float32(.5) * (chi_left + chi_right) * np.float32(right-left)",
                    mode="eval",
                ).body
    # hard_rate is both an accumulator and a function result; initialize only
    # the moment accumulator in single precision, leaving call sites inferred.
    for node in ast.walk(functions["_rad_moment"]):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "hard_rate"
        ):
            node.value = ast.parse("np.float32(0)", mode="eval").body
    functions["_rad_cell_rate"].body = ast.parse("""
if right <= left:
    return np.float32(0)
x = np.float32((right-left)/left)
chi_left = np.float32(chi_left)
chi_right = np.float32(chi_right)
log_ratio = xp.log1p(x)
correction = x-log_ratio
if x <= np.float32(.01):
    correction = x*x*(np.float32(.5)+x*(np.float32(-1/3)+x*(np.float32(.25)+x*(np.float32(-.2)+x*np.float32(1/6)))))
return chi_left*log_ratio+(chi_right-chi_left)*(correction/x)
""").body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _quotient_source(source):
    """Cast precisely formed coordinate differences before selected divisions."""
    tree = ast.parse(_boundary_source(source))
    targets = {"_rad_row_chi": "weight", "_rad_cell_rate": "x", "_rad_sample_photon_eV": "chi_mid"}

    class Quotient(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Div):

                def single(value):
                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load()
                        ),
                        args=[value],
                        keywords=[],
                    )

                node.left, node.right = single(node.left), single(node.right)
            return node

    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in targets:
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == targets[function.name]
                ):
                    node.value = Quotient().visit(node.value)
                    # Drop the now-redundant outer cast to expose FP32 division.
                    if function.name == "_rad_cell_rate":
                        node.value = node.value.args[0]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
