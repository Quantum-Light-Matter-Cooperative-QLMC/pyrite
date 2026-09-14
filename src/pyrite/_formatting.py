def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"
