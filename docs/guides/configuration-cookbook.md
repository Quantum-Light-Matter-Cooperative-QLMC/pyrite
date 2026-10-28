(configuration-cookbook)=
# Configure and inspect a campaign

Clone a bundled campaign, attach a narrower beam, and inspect a two-energy HOPG
scan before computing. This Linux shell exercise uses an installed `pyrite`;
it needs no reference tables, downloads, or simulation.

For an editable Python parameter study, start with
[Parameter sweeps](../notebooks/designing/08-parameter-sweeps.ipynb). Use this
recipe when the scan needs a reusable named CLI profile; the
[campaign reference](../reference/campaign-configuration.md) owns its settings.

(select-defaults-for-this-machine)=
(choose-an-output-home)=
(create-and-inspect-a-campaign)=
(reuse-a-named-beam)=
(change-one-material-safely)=
(derive-missing-photon-grids)=
(diagnose-precedence)=
## Create and inspect

Run the block in a subshell with `sh`, or download
{download}`configuration_cookbook.sh <../examples/configuration_cookbook.sh>`.
It isolates configuration, catalog edits and workspace in a temporary directory,
then removes the exercise on exit. Your existing defaults stay intact.

```{literalinclude} ../examples/configuration_cookbook.sh
:language: bash
```

Actual captured command output:

```{literalinclude} ../examples/data/configuration_cookbook.txt
:language: text
```

The scan keeps the demo's explicit `example`-derived beam, `reference` scalar
detector and adaptive sampling policy. It now selects 30 and 60 keV, with a
5 µm target (`50000` Å). `inherited` means a range comes from the profile;
`overridden` would identify a material-specific replacement. Inspect effective
ranges after edits: those resolved inputs determine checkpoint identity.

## Keep a laboratory setup

For persistent work, create your own profile with the same commands outside
the disposable exercise. Bundled names are read-only; user objects go into
the user catalog layer. Clone a single-material profile when one target needs
different ranges. Set `workspace.root` to your desired output directory
with `pyrite config set workspace.root PATH`. A complete, separately versioned
catalog is useful when the whole laboratory configuration must travel together;
follow [catalog setup](../reference/catalogs-and-workspaces.md#copy-a-complete-catalog).

Once reviewed, `pyrite run lab-scan` executes the saved campaign. Running
transport needs the [reference tables](getting-started.md#fetch-reference-data)
in the selected workspace. Changing the workspace also changes table discovery.

Use [profile fields](../repo-design/profile-settings.md) for units and setters,
[campaign contracts](../reference/campaign-configuration.md) for sampling,
beams and detectors, and [configuration precedence](../repo-design/configuration-resolution.md)
to diagnose which value wins. [Catalogs and workspaces](../reference/catalogs-and-workspaces.md)
owns output directories, input tables and catalog selection.
