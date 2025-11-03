#!/bin/bash
#SBATCH --job-name=find_modules
#SBATCH --output=modules.out
#SBATCH --error=modules.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

# This script just runs "module avail" on a compute node
# and saves the output to "modules.out"
module avail 2>&1