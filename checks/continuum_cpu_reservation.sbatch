#!/bin/bash
#SBATCH --time=04:00:00
set -euo pipefail
while [[ ! -f "$1" ]]; do sleep 10; done
