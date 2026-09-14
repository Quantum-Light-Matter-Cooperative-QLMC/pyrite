"""Terminal presentation primitives shared by the CLI and the domain packages.

`cli/` drives; `checkpoints/`, `runs/` and `remote/` report progress and emit
automation payloads while they work. Both need the same colour, prompt, JSON
envelope, completion and dashboard code, so it lives here, below both, rather
than in `cli/` where reaching up for it put four packages in one import cycle
(issue #64, finding 2).

Nothing in this package may import a PyRITE package that sits above it.
"""
