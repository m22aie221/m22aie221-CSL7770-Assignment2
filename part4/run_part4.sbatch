#!/bin/bash
# scripts/run_part4.sh
# Run TGU + all baselines from PROJECT_ROOT.

#SBATCH --job-name=speech_UL_job         # Job name
#SBATCH --partition=fat                 # Choose the appropriate partition
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=09:00:00                 # Time limit hrs:min:sec
#SBATCH --output=part4_log%j.log          # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie221@iitj.ac.in  # Your email address for notifications


mkdir -p logs outputs

REFERENCE="../data/student_voice_ref.wav"
SYNTHESIZED="../part3/outputs/output_LRL_cloned.wav"
LID_MODEL="../part1/lid_model_best.pt"
LECTURE="../data/lecture_segment.wav"

echo "Part 4 started at $(date)"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# Task 4.1: Anti-spoofing CM + EER (~30-60 min)
python part4_adversarial.py --reference "$REFERENCE" --synthesized "$SYNTHESIZED" --lid_model "$LID_MODEL" --lecture "$LECTURE" --mode spoof --epochs 40

# Task 4.2: FGSM adversarial attack (~10-20 min)
python part4_adversarial.py --reference "$REFERENCE" --synthesized "$SYNTHESIZED" --lid_model "$LID_MODEL" --lecture "$LECTURE" --mode fgsm --segment_start 30.0

echo "Part 4 finished at $(date)"
