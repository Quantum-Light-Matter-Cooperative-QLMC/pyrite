# Source only inside a single-node allocation. Slurm binding alone was ineffective
# on the pilot WSL host; apply and log OS affinity to the assigned CPU IDs.
allocation_record=$(scontrol show job "$SLURM_JOB_ID" -d -o)
if [[ ! $allocation_record =~ CPU_IDs=([0-9,-]+) ]]; then
  echo 'Cannot determine allocated CPU IDs' >&2
  exit 1
fi
allocated_cpu_ids=${BASH_REMATCH[1]}
taskset -pc "$allocated_cpu_ids" "$$"
export AUDIT1432_CPUS=$SLURM_CPUS_PER_TASK
