"""Build and inspect a material without writing to a catalog or running transport."""

from pyrite.materials import build_medium


def compute():
    silica = build_medium("lab-silica", formula="SiO2", density_g_cm3=2.2)
    lines = [f"Bulk density: {silica.density_g_cm3:.3f} g/cm^3"]
    for element, density in silica.composition:
        lines.append(f"{element}: {density:.6f} atoms/angstrom^3")
    return {
        "density_g_cm3": silica.density_g_cm3,
        "composition": dict(silica.composition),
        "identity": silica.identity,
        "display_text": "\n".join(lines) + "\n",
    }


if __name__ == "__main__":
    print(compute()["display_text"], end="")
