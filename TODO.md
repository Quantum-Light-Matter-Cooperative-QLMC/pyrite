# Named sweep profiles and dataset identity

Define independently configurable `full` and provisional `survey` profiles.
Keep `full` production behavior compatible; reduce survey electron counts,
parameter sets, reflection count, and photon-grid extent/resolution.

Persist profile plus exact resolved settings/sweep provenance in checkpoint
identity. Isolate noncanonical variants in component checkpoint directories,
preserve manifests through archive/restore, and refuse merges across identities.
