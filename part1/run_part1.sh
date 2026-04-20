#!/bin/bash
# scripts/run_part1.sh
# Run TGU + all baselines from PROJECT_ROOT.

#SBATCH --job-name=speech_UL_job         # Job name
#SBATCH --partition=fat                 # Choose the appropriate partition
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gres=gpu:2                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=09:00:00                 # Time limit hrs:min:sec
#SBATCH --output=part1_log%j.log          # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie221@iitj.ac.in  # Your email address for notifications

set -e

# ── Create output/log dirs ──────────────────────────────────
mkdir -p logs outputs

python part1_transcription.py \
    --audio  ../data/lecture_segment.wav \
    --mode   full \
    --whisper_model large-v3

echo "Part 1 finished at $(date)"
