"""Rational auxiliary functions for ``Si(t) - pi/2``, ``4 < t < 48`` (#1407).

``Si(t) - pi/2 = -f(t) cos(t) - g(t) sin(t)``. Cephes fits use
``z = 1/t**2`` and split at ``t = 8``. Evaluate the complement directly:
subtracting pi/2 from a rounded Si would lose relative tail accuracy.
The power series, asymptotic tail, and midpoint window remain unchanged.

Coefficients adapted from SciPy xsf Cephes ``sici.h`` at
https://github.com/scipy/xsf/blob/55c14c958e42b7457071978f5b8ab8097174c172/include/xsf/cephes/sici.h
Copyright 1984, 1987, 1989 Stephen L. Moshier; Copyright (c) 2024, SciPy.
BSD-3-Clause terms are reproduced in THIRD-PARTY-NOTICES.md.

Validation: sinc-bin-integration
"""

_FN4 = (
    4.23612862892216586994e0,
    5.45937717161812843388e0,
    1.62083287701538329132e0,
    1.67006611831323023771e-1,
    6.81020132472518137426e-3,
    1.08936580650328664411e-4,
    5.48900223421373614008e-7,
)

_FD4 = (
    1.0,
    8.16496634205391016773e0,
    7.30828822505564552187e0,
    1.86792257950184183883e0,
    1.78792052963149907262e-1,
    7.01710668322789753610e-3,
    1.10034357153915731354e-4,
    5.48900252756255700982e-7,
)

_GN4 = (
    8.71001698973114191777e-2,
    6.11379109952219284151e-1,
    3.97180296392337498885e-1,
    7.48527737628469092119e-2,
    5.38868681462177273157e-3,
    1.61999794598934024525e-4,
    1.97963874140963632189e-6,
    7.82579040744090311069e-9,
)

_GD4 = (
    1.0,
    1.64402202413355338886e0,
    6.66296701268987968381e-1,
    9.88771761277688796203e-2,
    6.22396345441768420760e-3,
    1.73221081474177119497e-4,
    2.02659182086343991969e-6,
    7.82579218933534490868e-9,
)

_FN8 = (
    4.55880873470465315206e-1,
    7.13715274100146711374e-1,
    1.60300158222319456320e-1,
    1.16064229408124407915e-2,
    3.49556442447859055605e-4,
    4.86215430826454749482e-6,
    3.20092790091004902806e-8,
    9.41779576128512936592e-11,
    9.70507110881952024631e-14,
)

_FD8 = (
    1.0,
    9.17463611873684053703e-1,
    1.78685545332074536321e-1,
    1.22253594771971293032e-2,
    3.58696481881851580297e-4,
    4.92435064317881464393e-6,
    3.21956939101046018377e-8,
    9.43720590350276732376e-11,
    9.70507110881952025725e-14,
)

_GN8 = (
    6.97359953443276214934e-1,
    3.30410979305632063225e-1,
    3.84878767649974295920e-2,
    1.71718239052347903558e-3,
    3.48941165502279436777e-5,
    3.47131167084116673800e-7,
    1.70404452782044526189e-9,
    3.85945925430276600453e-12,
    3.14040098946363334640e-15,
)

_GD8 = (
    1.0,
    1.68548898811011640017e0,
    4.87852258695304967486e-1,
    4.67913194259625806320e-2,
    1.90284426674399523638e-3,
    3.68475504442561108162e-5,
    3.57043223443740838771e-7,
    1.72693748966316146736e-9,
    3.87830166023954706752e-12,
    3.14040098946363335242e-15,
)


def _c_horner(coefficients):
    expression = repr(coefficients[0])
    for coefficient in coefficients[1:]:
        expression = f"({expression} * z + {coefficient!r})"
    return expression


SI_RATIONAL_COEFFICIENTS = ((_FN4, _FD4, _GN4, _GD4), (_FN8, _FD8, _GN8, _GD8))


def _device_source():
    lines = [
        "static __device__ double pyrite_si_rational_complement(double t) {",
        "const double z = 1.0 / (t * t);",
        "double f, g;",
        "if (t < 8.0) {",
    ]
    for index, (fn, fd, gn, gd) in enumerate(SI_RATIONAL_COEFFICIENTS):
        if index:
            lines.append("} else {")
        lines.append(f"f = {_c_horner(fn)} / (t * {_c_horner(fd)});")
        lines.append(f"g = z * {_c_horner(gn)} / {_c_horner(gd)};")
    lines.extend(("}", "return -f * cos(t) - g * sin(t);", "}"))
    return "\n".join(lines)


SI_RATIONAL_PREAMBLE = _device_source()
