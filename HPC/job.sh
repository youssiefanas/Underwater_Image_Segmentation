#!/bin/bash

#SBATCH --job-name=Train_USIS
#SBATCH --partition=mundus            # Use the mundus partition
#SBATCH --nodelist=mundus-mir-3
#SBATCH --gres=gpu:a100-20:1          # Request 1 GPU of type a100-20
#SBATCH --ntasks=1                    # Single task
#SBATCH --cpus-per-task=4             # 4 CPU threads per task
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=7:00:00                # Time limit: 2 hours
#SBATCH --output=output/job_output_%j.log    # Output log
#SBATCH --error=output/job_error_%j.log      # Error log

# Load necessary modules
module purge
module load cuda/12.1
module load conda
conda activate seg


python /home/mundus/ymorsy172/USIS10K/vis_infer.py
# python tools/train.py project/our/configs/multiclass_usis_train.py


# python tools/train.py project/our/configs/multiclass_usis_train.py --work-dir work_dirs/multiclass_usis
