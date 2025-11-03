#!/bin/bash

#SBATCH --job-name=knn_cuda_maple
#SBATCH --output=mout.out
#SBATCH --error=merr.err
#SBATCH --partition=gpu          # Request the GPU partition
#SBATCH --gres=gpu:1             # Request one GPU (Generic way)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        
#SBATCH --time=00:30:00          
#SBATCH --mem=32G                

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"

# Load modules (adjust versions if necessary)
module purge
# *** IMPORTANT ***
# You must find the correct module names on the Maple/Apollo server
# by running "module avail 2>&1 | grep -i cuda"
module load gcc/10.5.0
module load cuda/12.3

make clean
make knn_cuda

# Check if compilation was successful
if [ ! -f "knn_cuda" ]; then
    echo "Compilation FAILED. Exiting."
    cat knn_cuda_maple_%j.err # Print the compile error
    exit 1
fi
echo "Compilation complete."

# --- Set Paths to Datasets ---
# !! IMPORTANT: Update this path to your dataset directory !!
DATA_DIR="./datasets"

# Using hyphenated names as requested
SMALL_TRAIN="${DATA_DIR}/small-train.arff"
SMALL_TEST="${DATA_DIR}/small-test.arff"
MEDIUM_TRAIN="${DATA_DIR}/medium-train.arff"
MEDIUM_TEST="${DATA_DIR}/medium-test.arff"
LARGE_TRAIN="${DATA_DIR}/large-train.arff"
LARGE_TEST="${DATA_DIR}/large-test.arff"

# --- Set Parameters ---
K_VALUE=3
BLOCK_SIZE=128 

# --- Run Experiments ---
echo "--- Running on Small Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${SMALL_TRAIN}" "${SMALL_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "--- Running on Medium Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${MEDIUM_TRAIN}" "${MEDIUM_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "--- Running on Large Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${LARGE_TRAIN}" "${LARGE_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "Job finished at $(date)"
