# /// script
# [tool.marimo.display]
# theme = "dark"
# ///


import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    from _design import page_title, status_badge, style_sheet

    from cxr_mc import check as check_support

    repo_dir = Path(__file__).resolve().parent.parent
    checks_dir = repo_dir / "checks"
    if str(checks_dir) not in sys.path:
        sys.path.insert(0, str(checks_dir))

    import anchor_figures as af

    def run_checks(filenames, force_cpu):
        """Run selected standalone checks and retain their complete console reports."""
        reports = []
        for filename in filenames:
            path = checks_dir / filename
            if force_cpu:
                command = [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,runpy,sys;"
                        "sys.modules['cupy']=None;"
                        "sys.path.insert(0,str(pathlib.Path(sys.argv[1]).resolve().parent));"
                        "runpy.run_path(sys.argv[1],run_name='__main__')"
                    ),
                    str(path),
                ]
            else:
                command = [sys.executable, str(path)]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            reports.append(
                {
                    "filename": filename,
                    "returncode": completed.returncode,
                    "seconds": time.perf_counter() - started,
                    "output": output.rstrip() or "(no output)",
                }
            )
        return reports

    check_authorities = {
        "mosaic_mc_check.py": "Anchor",
        "detector_solid_angle_check.py": "Anchor",
        "multilayer_validation_check.py": "Anchor",
        "multilayer_check.py": "Anchor",
        "multilayer_slice3_check.py": "Anchor",
        "dans_diffraction_oracle.py": "Optional oracle",
        "feranchuk_check_script.py": "Diagnostic",
        "feranchuk_vs_zhai_check.py": "Diagnostic",
        "kinematic_validity_check.py": "Diagnostic",
    }

    return (
        af,
        check_authorities,
        check_support,
        mo,
        page_title,
        run_checks,
        status_badge,
        style_sheet,
    )


@app.cell(hide_code=True)
def _(mo, page_title, status_badge, style_sheet):
    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "CXR physics validation",
                "Run evidence with explicit authority, inspect complete reports, and keep derivation provenance distinct from pass/fail claims.",
                eyebrow="Beamline control / validation",
            ),
            mo.hstack(
                [
                    status_badge(mo, "Ready"),
                    mo.md(
                        "**Evidence summary:** actions remain idle until their run button is pressed."
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    audit_rows = [
        {
            "artifact": "anchor_figures.py",
            "role": "anchor backend",
            "decision": "keep",
            "reason": "Unique Zhai line-energy, absolute-flux, and bulk/film anchors.",
        },
        {
            "artifact": "mosaic_mc_check.py",
            "role": "pass/fail anchor",
            "decision": "keep",
            "reason": "Validates the exact orientation average, convergence, and analytic limit.",
        },
        {
            "artifact": "detector_solid_angle_check.py",
            "role": "anchor + diagnostic",
            "decision": "keep",
            "reason": "Unique end-to-end wide-detector lineshape integration check.",
        },
        {
            "artifact": "multilayer_validation_check.py",
            "role": "pass/fail anchor",
            "decision": "keep",
            "reason": "Closed-form absorption and independent K-O transport-range comparison.",
        },
        {
            "artifact": "multilayer_check.py",
            "role": "pass/fail anchor",
            "decision": "keep",
            "reason": "Front/back escape and substrate backscatter exercise different behavior.",
        },
        {
            "artifact": "multilayer_slice3_check.py",
            "role": "pass/fail anchor",
            "decision": "keep",
            "reason": "Only check of per-crystalline-layer incoherent radiation summation.",
        },
        {
            "artifact": "dans_diffraction_oracle.py",
            "role": "optional oracle",
            "decision": "keep",
            "reason": "Independent crystallography implementation; optional dependency may skip.",
        },
        {
            "artifact": "feranchuk_check_script.py",
            "role": "diagnostic",
            "decision": "keep, relabel",
            "reason": "Reports the unresolved LiF absolute-flux discrepancy; not a passing anchor.",
        },
        {
            "artifact": "feranchuk_vs_zhai_check.py",
            "role": "diagnostic",
            "decision": "keep",
            "reason": "Compares idealized and transport pipelines in their valid/invalid regimes.",
        },
        {
            "artifact": "kinematic_validity_check.py",
            "role": "diagnostic",
            "decision": "keep",
            "reason": "Defines the model's applicability envelope; warnings depend on use case.",
        },
        {
            "artifact": "feranchuk_spence.py",
            "role": "reference backend",
            "decision": "keep",
            "reason": "Shared analytic implementation used by multiple independent comparisons.",
        },
        {
            "artifact": "cxr_analysis_feranchuk.ipynb/.md",
            "role": "provenance",
            "decision": "preserve",
            "reason": "Unique derivation narrative and paper-figure studies; not a pass/fail check.",
        },
        {
            "artifact": "mosaic_scoping_check.py",
            "role": "completed design study",
            "decision": "retire",
            "reason": "Its go/no-go question was answered when exact mosaic MC was implemented.",
        },
        {
            "artifact": "zhai_fig1c_check + zhai_fig1c_validation notebooks",
            "role": "duplicate wrappers",
            "decision": "retire",
            "reason": "Superseded by anchor_figures.py and this dashboard's Zhai section.",
        },
    ]
    provenance_audit = mo.accordion(
        {
            "Provenance: authority audit and unresolved methods": mo.ui.table(
                audit_rows,
                selection=None,
                page_size=20,
            )
        }
    )
    return (provenance_audit,)


@app.cell(hide_code=True)
def _(mo):
    provenance_findings = mo.callout(
        mo.md(r"""
        **Audit findings from the current implementations**

        - The LiF absolute-flux report remains **1.74 decades above** the paper value;
          it is an unresolved discrepancy, not a calibration or validation anchor.
        - The idealized Feranchuk line yield agrees with transport for the 29 nm film
          (ratio **1.00**) but overshoots the 1 mm bulk result (**2.57**) outside its
          kinematic validity regime.
        - The applicability audit intentionally emits warnings for recoil and long
          coherent lengths. Those warnings are results, not test failures.
        - The optional Dans_Diffraction oracle exits cleanly when the package is absent;
          read its report to distinguish a real comparison from a skip.
        """),
        kind="warn",
    )
    return (provenance_findings,)


@app.cell(hide_code=True)
def _(mo):
    anchors_intro = mo.md(r"""
    ## Standalone anchors and diagnostics

    Select related checks and run them in fresh child processes. CPU forcing is
    useful on machines that have a CuPy wheel but no usable CUDA device. A zero
    exit code means the program completed its own contract; diagnostic programs
    still require interpretation and are deliberately not presented as “passed”.
    """)
    return (anchors_intro,)


@app.cell
def _(mo):
    check_options = {
        "Mosaic: exact orientation average": "mosaic_mc_check.py",
        "Detector: solid-angle integration": "detector_solid_angle_check.py",
        "Multilayer: absorption + transport": "multilayer_validation_check.py",
        "Multilayer: escape + backscatter": "multilayer_check.py",
        "Multilayer: crystalline substrate radiation": "multilayer_slice3_check.py",
        "Oracle: Dans_Diffraction (optional)": "dans_diffraction_oracle.py",
        "Diagnostic: LiF absolute-flux discrepancy": "feranchuk_check_script.py",
        "Diagnostic: idealized vs transport": "feranchuk_vs_zhai_check.py",
        "Diagnostic: kinematic validity envelope": "kinematic_validity_check.py",
    }
    default_checks = [
        "Mosaic: exact orientation average",
        "Detector: solid-angle integration",
        "Multilayer: absorption + transport",
        "Multilayer: escape + backscatter",
        "Multilayer: crystalline substrate radiation",
    ]
    checks_ui = mo.ui.multiselect(
        check_options,
        value=default_checks,
        label="Checks to run",
    )
    force_cpu_ui = mo.ui.checkbox(value=True, label="Force CPU (disable CuPy in child processes)")
    run_checks_ui = mo.ui.run_button(label="Run selected checks")
    anchors_controls = mo.vstack([checks_ui, mo.hstack([force_cpu_ui, run_checks_ui])])
    return anchors_controls, checks_ui, force_cpu_ui, run_checks_ui


@app.cell
def _(checks_ui, force_cpu_ui, mo, run_checks, run_checks_ui):
    if not run_checks_ui.value:
        check_reports = None
    elif not checks_ui.value:
        check_reports = []
    else:
        with mo.status.spinner(
            title="Running validation programs",
            subtitle="The complete console report for each check will appear below.",
        ):
            check_reports = run_checks(checks_ui.value, force_cpu_ui.value)
    return (check_reports,)


@app.cell
def _(check_authorities, check_reports, mo):
    report_panels = {}
    summary = []
    for report in check_reports or []:
        name = report["filename"]
        authority = check_authorities[name]
        if report["returncode"] != 0:
            state = "Failed"
            next_action = "Inspect full report and fix failure."
        elif authority == "Diagnostic":
            state = "Completed—interpret"
            next_action = "Interpret output against method and source assumptions."
        elif authority == "Optional oracle" and "skip" in report["output"].lower():
            state = "Skipped"
            next_action = "Install optional dependency to run oracle."
        elif authority == "Optional oracle":
            state = "Completed—interpret"
            next_action = "Review independent comparison output."
        else:
            state = "Passed"
            next_action = "No action required."
        summary.append(
            {
                "check": name,
                "authority": authority,
                "state": state,
                "elapsed (s)": round(report["seconds"], 1),
                "next action": next_action,
            }
        )
        report_panels[f"{state} — {name} ({report['seconds']:.1f} s)"] = mo.md(
            f"```text\n{report['output']}\n```"
        )
    if check_reports is None:
        anchors_results = mo.callout("Select checks above, then run them when ready.", kind="info")
    elif not check_reports:
        anchors_results = mo.callout("Select at least one check.", kind="warn")
    else:
        anchors_results = mo.vstack(
            [
                mo.ui.table(summary, selection=None),
                mo.accordion(report_panels, multiple=True),
            ]
        )
    return (anchors_results,)


@app.cell(hide_code=True)
def _(mo):
    reproductions_intro = mo.md(r"""
    ## Zhai Fig. 1c reproduction

    Reproduces the HOPG spectra and bulk-versus-film comparison from Zhai et al.,
    *Nature Communications* **16**, 11218 (2025). If
    `checks/reference_data/zhai_fig1c.csv` is present, its digitized curves are
    overlaid automatically. Use 20,000 line electrons and 200 bremsstrahlung
    electrons for publication-quality output. Results are cached locally by
    sample counts, experimental inputs, and implementation version.
    """)
    return (reproductions_intro,)


@app.cell
def _(mo):
    ne_ui = mo.ui.number(
        start=10,
        stop=50_000,
        step=10,
        value=20_000,
        label="Line-spectrum electrons per energy",
    )
    ne_brem_ui = mo.ui.number(
        start=10,
        stop=1_000,
        step=10,
        value=200,
        label="Bremsstrahlung electrons per energy",
    )
    run_zhai_ui = mo.ui.run_button(label="Run Zhai reproduction")
    refresh_zhai_ui = mo.ui.checkbox(value=False, label="Recompute (ignore cache)")
    reproductions_controls = mo.vstack(
        [mo.hstack([ne_ui, ne_brem_ui]), mo.hstack([refresh_zhai_ui, run_zhai_ui])]
    )
    return ne_brem_ui, ne_ui, refresh_zhai_ui, reproductions_controls, run_zhai_ui


@app.cell
def _(af, mo):
    anchor = af.ZhaiAnchor()
    theory_energies = af.theory_line_energies(anchor)
    line_energy_anchors = mo.ui.table(
        [
            {"Beam energy (keV)": energy, "Eq. (10) line energy (eV)": line_energy}
            for energy, line_energy in theory_energies.items()
        ],
        selection=None,
        label="Analytic line-energy anchors",
    )
    return anchor, line_energy_anchors


@app.cell
def _(af, anchor, mo, ne_brem_ui, ne_ui, refresh_zhai_ui, run_zhai_ui):
    if not run_zhai_ui.value:
        zhai_cache_hit = None
        zhai_cache_path = None
        zhai_model = None
        zhai_reference = None
    else:
        with mo.status.spinner(
            title="Running Monte Carlo spectra",
            subtitle="This is the expensive step; all three figures reuse the result.",
        ):
            zhai_model, zhai_cache_hit, zhai_cache_path = af.cached_model_spectra(
                anchor,
                ne=int(ne_ui.value),
                ne_brem=int(ne_brem_ui.value),
                refresh=refresh_zhai_ui.value,
            )
            zhai_reference = af.reference_curve(anchor)
    return zhai_cache_hit, zhai_cache_path, zhai_model, zhai_reference


@app.cell
def _(
    af,
    anchor,
    mo,
    zhai_cache_hit,
    zhai_cache_path,
    zhai_model,
    zhai_reference,
):
    if zhai_model is None:
        reproductions_results = mo.callout(
            "Choose the sample counts, then run the reproduction.", kind="info"
        )
    else:
        rows = af.validation_table(anchor, zhai_model)
        headers = [
            "E0 (keV)",
            "MC peak (eV)",
            "Eq. 10 (eV)",
            "difference (eV)",
            "MC / closed (one segment)",
            "line flux (ph/e/0.066 sr)",
            "backscatter",
        ]
        validation = mo.ui.table(
            [dict(zip(headers, row, strict=True)) for row in rows],
            selection=None,
            label="Validation summary",
        )

        def spectra_tab():
            note = (
                "Digitized Zhai reference curves loaded."
                if zhai_reference
                else "No digitized reference CSV found; showing the theory-only overlay."
            )
            return mo.vstack([mo.md(note), af.figure_spectra(anchor, zhai_model, zhai_reference)])

        def flux_tab():
            return af.figure_flux_anchor(anchor, zhai_model)

        def enhancement_tab():
            return af.figure_enhancement(anchor, zhai_model)

        def background_tab():
            background = af.zhai_background_validation(anchor, zhai_model)
            fit = background["fit"]
            comparison = background["comparison"]
            summary = mo.ui.table(
                [
                    {
                        "External scale": fit.scale,
                        "Scale uncertainty": fit.scale_std,
                        "Reduced chi-square": fit.reduced_chi2,
                        "Sideband points": fit.n_points,
                        "cxr / external integrated brem": comparison.integrated_ratio,
                        "Normalized RMSE": comparison.normalized_rmse,
                        "Shape correlation": comparison.correlation,
                    }
                ],
                selection=None,
                label="Background validation metrics",
            )
            return mo.vstack(
                [
                    mo.md(
                        "Deposited DR-NTU V1 Figure 3b data; weighted scale fit uses "
                        "sidebands outside 900–1040 eV. DTSA-II identifies Zhai's "
                        "simulation software, not a canonical NIST spectrum."
                    ),
                    summary,
                    af.figure_background_validation(background, anchor, zhai_model),
                ]
            )

        reproductions_results = mo.vstack(
            [
                mo.callout(
                    f"{'Loaded cached' if zhai_cache_hit else 'Computed and cached'} result at "
                    f"`{zhai_cache_path.relative_to(zhai_cache_path.parents[1])}`.",
                    kind="success",
                ),
                validation,
                mo.ui.tabs(
                    {
                        "Fig. 1c spectra": spectra_tab,
                        "External brem + subtraction": background_tab,
                        "Absolute-flux anchor": flux_tab,
                        "Bulk vs 29 nm film": enhancement_tab,
                    },
                    lazy=True,
                ),
            ]
        )
    return (reproductions_results,)


@app.cell(hide_code=True)
def _(mo):
    supplementary_intro = mo.md(r"""
    ## Zhai supplementary coherent-emission studies

    These are separate from the Fig. 1c anchor: each run contains coherent
    PXR+CBS emission, convolved with the same Zhai EDS-plus-aperture detector
    response and normalized to **Phs/eV/s/nA**. WSe₂ and MoSe₂ each render four
    reported polar-tilt panels at 200 keV. Their TEM azimuth is not reported, so
    it must be selected explicitly below and is labeled exploratory. The 921 nm
    h-BN and HOPG use the thickness-specific polar/azimuth orientations reported
    in Supplementary Table 4 and overlay the 17.5, 20, 22.5, and 25 keV SEM
    spectra (the 219 nm h-BN sample has two reported orientations).
    """)
    return (supplementary_intro,)


@app.cell
def _(mo):
    supplementary_study_ui = mo.ui.dropdown(
        {
            "WSe₂ (800–1200 eV)": "wse2",
            "MoSe₂ (800–1200 eV)": "mose2",
            "h-BN (600–1200 eV)": "hbn",
            "HOPG (500–1250 eV)": "hopg",
        },
        value="WSe₂ (800–1200 eV)",
        label="Supplementary material",
    )
    return (supplementary_study_ui,)


@app.cell
def _(af, mo, supplementary_study_ui):
    supplementary_study = af.supplementary_study(supplementary_study_ui.value)
    supplementary_thickness_ui = mo.ui.dropdown(
        {f"{thickness:g} nm": thickness for thickness in supplementary_study.thicknesses_nm},
        value=f"{supplementary_study.thicknesses_nm[0]:g} nm",
        label="Crystal thickness",
    )
    return (supplementary_thickness_ui,)


@app.cell
def _(af, check_support, mo, supplementary_study_ui, supplementary_thickness_ui):
    _study = af.supplementary_study(supplementary_study_ui.value)
    _orientation_options = af.supplementary_orientations(
        _study, float(supplementary_thickness_ui.value)
    )
    supplementary_orientation_ui = (
        mo.ui.dropdown(
            {
                f"Polar {polar:g}°, azimuth {azimuth:g}°": (polar, azimuth)
                for polar, azimuth in _orientation_options
            },
            value=f"Polar {_orientation_options[0][0]:g}°, azimuth {_orientation_options[0][1]:g}°",
            label="Reported angle combination",
        )
        if _study.crystal in {"hbn", "hopg"}
        else None
    )
    supplementary_ne_ui = mo.ui.number(
        start=10,
        stop=5_000,
        step=10,
        value=200,
        label="Electrons per spectrum",
    )
    supplementary_azimuth_ui = mo.ui.number(
        start=0,
        stop=180,
        step=5,
        value=check_support.load_default_azimuth(),
        label="Exploratory TMD azimuth (deg; unreported)",
    )
    save_supplementary_azimuth_ui = mo.ui.run_button(label="Save repository default")
    run_supplementary_ui = mo.ui.run_button(label="Run supplementary study")
    refresh_supplementary_ui = mo.ui.checkbox(value=False, label="Recompute (ignore cache)")
    supplementary_controls = mo.vstack(
        [
            mo.hstack(
                [
                    supplementary_study_ui,
                    supplementary_thickness_ui,
                    *([supplementary_orientation_ui] if supplementary_orientation_ui else []),
                ]
            ),
            mo.hstack([supplementary_azimuth_ui, save_supplementary_azimuth_ui])
            if _study.has_unreported_azimuth
            else mo.md(
                "**Reported orientation(s):** encoded per thickness from Supplementary Table 4."
            ),
            mo.hstack([supplementary_ne_ui, refresh_supplementary_ui, run_supplementary_ui]),
            mo.md(
                "`Save repository default` mutates `tmd_exploratory_azimuth_deg` in "
                "`notebooks/validation_defaults.json`; the selected value is shown above."
            )
            if _study.has_unreported_azimuth
            else mo.md("Reported orientations are provenance inputs and are not persisted here."),
        ]
    )
    return (
        refresh_supplementary_ui,
        run_supplementary_ui,
        save_supplementary_azimuth_ui,
        supplementary_controls,
        supplementary_azimuth_ui,
        supplementary_ne_ui,
        supplementary_orientation_ui,
    )


@app.cell
def _(check_support, mo, save_supplementary_azimuth_ui, supplementary_azimuth_ui):
    if save_supplementary_azimuth_ui.value:
        check_support.save_default_azimuth(float(supplementary_azimuth_ui.value))
        supplementary_save_status = mo.callout(
            f"Saved {float(supplementary_azimuth_ui.value):g}° as the repository default. "
            "It will be selected on the next app startup.",
            kind="success",
        )
    else:
        supplementary_save_status = mo.md("")
    return (supplementary_save_status,)


@app.cell(hide_code=True)
def _(mo):
    remote_intro = mo.md(r"""
    ### Remote-first cache preparation

    This optional launcher prepares all expensive Zhai caches using the values
    currently selected above. It first probes the configured SSH GPU host. When
    available, it starts a detached job that continues after this app closes;
    status is polled below and completed caches are pulled automatically. If the
    SSH tools or host are unavailable before launch, the same batch runs locally.
    """)
    return (remote_intro,)


@app.cell
def _(mo, ne_brem_ui, ne_ui, supplementary_azimuth_ui, supplementary_ne_ui):
    prepare_zhai_caches_ui = mo.ui.run_button(label="Prepare expensive caches (remote first)")
    remote_status_refresh_ui = mo.ui.refresh(
        options=[5, 15, 30], default_interval=15, label="Remote status refresh"
    )
    remote_controls = mo.vstack(
        [
            mo.md(
                f"Selected batch: {int(ne_ui.value)} line electrons, "
                f"{int(ne_brem_ui.value)} bremsstrahlung electrons, "
                f"{int(supplementary_ne_ui.value)} supplementary electrons, "
                f"TMD azimuth {float(supplementary_azimuth_ui.value):g}°. "
                "Remote probe runs only after button press; unavailable SSH/GPU falls back locally."
            ),
            mo.hstack([prepare_zhai_caches_ui, remote_status_refresh_ui]),
        ]
    )
    return prepare_zhai_caches_ui, remote_controls, remote_status_refresh_ui


@app.cell
def _(mo):
    get_remote_zhai_job, set_remote_zhai_job = mo.state(None)
    get_remote_zhai_message, set_remote_zhai_message = mo.state(
        "No background cache-preparation job has been started."
    )
    return (
        get_remote_zhai_job,
        get_remote_zhai_message,
        set_remote_zhai_job,
        set_remote_zhai_message,
    )


@app.cell
def _(
    af,
    check_support,
    get_remote_zhai_job,
    get_remote_zhai_message,
    mo,
    ne_brem_ui,
    ne_ui,
    prepare_zhai_caches_ui,
    refresh_zhai_ui,
    refresh_supplementary_ui,
    remote_status_refresh_ui,
    set_remote_zhai_job,
    set_remote_zhai_message,
    supplementary_azimuth_ui,
    supplementary_ne_ui,
):
    _ = remote_status_refresh_ui.value
    _jobid = get_remote_zhai_job()
    if prepare_zhai_caches_ui.value and _jobid is None:
        _available, _reason = check_support.probe_remote_zhai()
        if _available:
            _jobid = check_support.start_remote_zhai(
                ne=int(ne_ui.value),
                ne_brem=int(ne_brem_ui.value),
                ne_supp=int(supplementary_ne_ui.value),
                tmd_azimuth=float(supplementary_azimuth_ui.value),
                refresh=refresh_zhai_ui.value or refresh_supplementary_ui.value,
            )
            set_remote_zhai_job(_jobid)
            set_remote_zhai_message(
                f"Remote job `{_jobid}` launched; it is running in the background."
            )
        else:
            set_remote_zhai_message(f"{_reason} Falling back to a local batch run.")
            with mo.status.spinner(title="Remote unavailable; preparing caches locally"):
                af.reproduce_all(
                    ne=int(ne_ui.value),
                    ne_brem=int(ne_brem_ui.value),
                    ne_supp=int(supplementary_ne_ui.value),
                    tmd_exploratory_azimuth_deg=float(supplementary_azimuth_ui.value),
                    refresh=refresh_zhai_ui.value or refresh_supplementary_ui.value,
                )
            set_remote_zhai_message("Remote unavailable; the local fallback completed.")
    elif _jobid is not None:
        _state, _report = check_support.remote_zhai_status(_jobid)
        if _state == "done":
            _pull_report = check_support.pull_remote_zhai()
            set_remote_zhai_job(None)
            set_remote_zhai_message(
                f"Remote job `{_jobid}` completed and its caches were pulled.\n\n{_pull_report}"
            )
        elif _state in {"failed", "error"}:
            set_remote_zhai_job(None)
            set_remote_zhai_message(f"Remote job `{_jobid}` {_state}.\n\n{_report}")
        else:
            set_remote_zhai_message(f"Remote job `{_jobid}`: {_state}.\n\n{_report}")
    remote_status = mo.callout(get_remote_zhai_message(), kind="info")
    return (remote_status,)


@app.cell
def _(
    af,
    mo,
    refresh_supplementary_ui,
    run_supplementary_ui,
    supplementary_azimuth_ui,
    supplementary_ne_ui,
    supplementary_orientation_ui,
    supplementary_study_ui,
    supplementary_thickness_ui,
):
    if not run_supplementary_ui.value:
        supplementary_results = mo.callout(
            "Choose a material and thickness, then run the study.", kind="info"
        )
    else:
        study = af.supplementary_study(supplementary_study_ui.value)
        thickness_nm = float(supplementary_thickness_ui.value)
        exploratory_azimuth_deg = (
            float(supplementary_azimuth_ui.value) if study.has_unreported_azimuth else None
        )
        with mo.status.spinner(
            title="Running detector-convolved supplementary spectra",
            subtitle="Reported-condition spectra are computed or loaded from cache.",
        ):
            spectra, cache_hit, cache_path = af.cached_coherent_spectra(
                study,
                thickness_nm,
                ne=int(supplementary_ne_ui.value),
                exploratory_azimuth_deg=exploratory_azimuth_deg,
                refresh=refresh_supplementary_ui.value,
            )
        figure = (
            af.figure_supplementary_sem(
                study,
                thickness_nm,
                spectra,
                orientation=tuple(supplementary_orientation_ui.value),
            )
            if study.crystal in {"hbn", "hopg"}
            else af.figure_supplementary_tmd(study, thickness_nm, spectra)
        )
        supplementary_results = mo.vstack(
            [
                mo.callout(
                    f"{'Loaded cached' if cache_hit else 'Computed and cached'} result at "
                    f"`{cache_path.relative_to(cache_path.parents[1])}`."
                    + (
                        f" Exploratory TMD azimuth: {exploratory_azimuth_deg:g}° "
                        "(not reported by Zhai et al.)."
                        if exploratory_azimuth_deg is not None
                        else " Reported Table 4 orientation(s) selected for this thickness."
                    ),
                    kind="success",
                ),
                mo.center(figure),
            ]
        )
    return (supplementary_results,)


@app.cell(hide_code=True)
def _(
    anchors_controls,
    anchors_intro,
    anchors_results,
    line_energy_anchors,
    mo,
    provenance_audit,
    provenance_findings,
    remote_controls,
    remote_intro,
    remote_status,
    reproductions_controls,
    reproductions_intro,
    reproductions_results,
    supplementary_controls,
    supplementary_intro,
    supplementary_results,
    supplementary_save_status,
):
    mo.ui.tabs(
        {
            "Anchors": mo.vstack(
                [
                    mo.md("**Authority: Anchor, Diagnostic, or Optional oracle**"),
                    anchors_intro,
                    anchors_controls,
                    anchors_results,
                ]
            ),
            "Reproductions": mo.vstack(
                [
                    mo.md("**Authority: Anchor** — published-condition reproduction."),
                    reproductions_intro,
                    reproductions_controls,
                    line_energy_anchors,
                    reproductions_results,
                ]
            ),
            "Supplementary": mo.vstack(
                [
                    mo.md("**Authority: Diagnostic** — complete output requires interpretation."),
                    supplementary_intro,
                    supplementary_controls,
                    supplementary_save_status,
                    supplementary_results,
                    mo.accordion(
                        {
                            "Remote-first cache preparation": mo.vstack(
                                [remote_intro, remote_controls, remote_status]
                            )
                        }
                    ),
                ]
            ),
            "Provenance": mo.vstack(
                [
                    mo.md(
                        "**Authority: Provenance** — derivation and reference artifacts; "
                        "not executable evidence."
                    ),
                    provenance_findings,
                    provenance_audit,
                ]
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
