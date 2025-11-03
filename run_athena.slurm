#!/bin/bash

#SBATCH --job-name=knn_cuda
#SBATCH --output=knn_cuda_%j.out
#SBATCH --error=knn_cuda_%j.err
#SBATCH --partition=gpu          # Request the GPU partition
#SBATCH --gres=gpu:1             # Request one GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Request some CPUs for data loading
#SBATCH --time=00:30:00          # 30 minutes (adjust as needed)
#SBATCH --mem=32G                # Memory (adjust as needed)

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"

# Load modules (adjust versions if necessary)
module purge
module load gcc/11.2.0
module load cuda/11.7.0   # Athena has CUDA 11.7

# --- Compilation ---
# Clean previous builds and build everything
make clean
make knn_cuda

# Check if compilation was successful
if [ ! -f "knn_cuda" ]; then
    echo "Compilation FAILED. Exiting."
    exit 1
fi
echo "Compilation complete."

# --- Set Paths to Datasets ---
# !! IMPORTANT: Update these paths to your dataset locations !!
DATA_DIR="./datasets"
SMALL_TRAIN="${DATA_DIR}/small-train.arff"
SMALL_TEST="${DATA_DIR}/small-test.arff"
MEDIUM_TRAIN="${DATA_DIR}/medium-train.arff"
MEDIUM_TEST="${DATA_DIR}/medium-test.arff"
LARGE_TRAIN="${DATA_DIR}/large-train.arff"
LARGE_TEST="${DATA_DIT}/large-test.arff"

# --- Set Parameters ---
K_VALUE=5
# Block size to test. This is the parameter you should analyze.
# Good values to test: 128, 256, 512, 1024
BLOCK_SIZE=256 

# --- Run Experiments ---
echo "--- Running on Small Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${SMALL_TRAIN}" "${SMALL_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "--- Running on Medium Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${MEDIUM_TRAIN}" "${MEDIUM_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "--- Running on Large Dataset (k=${K_VALUE}, block_size=${BLOCK_SIZE}) ---"
./knn_cuda "${LARGE_TRAIN}" "${LARGE_TEST}" ${K_VALUE} ${BLOCK_SIZE}

echo "Job finished at $(date)"