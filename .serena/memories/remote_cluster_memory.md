# Remote cluster memory policy

- As reported by the lab cluster administrator on 2026-09-27: SLURM defaults to 1.5 GB RAM per requested CPU core for every submission. Explicit `#SBATCH --mem=<size>` or `#SBATCH --mem-per-cpu=<size>` may request more; maximum permitted per simulation is 42 GB. Verify the live policy before future submissions.
- PyRITE's shared `src/pyrite/remote/scripts.py::_slurm_batch_script` currently requests 8 CPUs per task and emits no memory directive. Its default allocation is therefore 12 GB under this policy.
- For memory-intensive remote measurements, request memory explicitly and record peak RSS/stage timings. A larger allocation does not establish or cure an inefficient algorithm.
