#!/bin/bash
# scripts/run_part3.sh
# Run TGU + all baselines from PROJECT_ROOT.

#SBATCH --job-name=speech_UL_job         # Job name
#SBATCH --partition=fat                 # Choose the appropriate partition
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=09:00:00                 # Time limit hrs:min:sec
#SBATCH --output=part3_log%j.log          # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie221@iitj.ac.in  # Your email address for notifications

set -e


mkdir -p logs outputs outputs/chunks

REFERENCE="../data/student_voice_ref.wav"
LECTURE="../data/lecture_segment.wav"
TRANSCRIPT="../part2/outputs/bhojpuri_translation.json"


# Step 1: embed only (fast, ~1 min)
python part3_voice_cloning.py --reference "$REFERENCE" --lecture "$LECTURE" --transcript "$TRANSCRIPT" --mode embed

# Step 2: prosody extraction + DTW (CPU-friendly, ~5 min)
python part3_voice_cloning.py --reference "$REFERENCE" --lecture "$LECTURE" --transcript "$TRANSCRIPT" --mode prosody

# Step 3: synthesis (GPU-heavy, ~2-3 hrs for 10 min audio)
python part3_voice_cloning.py --reference "$REFERENCE" --lecture "$LECTURE" --transcript "$TRANSCRIPT" --mode synthesize

# Step 4: MCD evaluation
python part3_voice_cloning.py --reference "$REFERENCE" --lecture "$LECTURE" --transcript "$TRANSCRIPT" --mode mcd

echo "Part 3 finished at $(date)"
