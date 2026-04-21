#!/bin/bash
# scripts/run_part1.sh
# Run TGU + all baselines from PROJECT_ROOT.

#SBATCH --job-name=speech_UL_job         # Job name
#SBATCH --partition=fat                 # Choose the appropriate partition
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=09:00:00                 # Time limit hrs:min:sec
#SBATCH --output=part2_log%j.log          # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie221@iitj.ac.in  # Your email address for notifications

set -e

mkdir -p logs outputs corpus

TRANSCRIPT="../part1/outputs/transcript.json"
MODE="full"

echo "Starting Part 2 at $(date)"
python part2_phonetic_translation.py --transcript "$TRANSCRIPT" --mode "$MODE"
echo "Part 2 finished at $(date)"
