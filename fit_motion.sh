#!/bin/bash
#SBATCH --job-name="process"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --output=stdout_fit_motion.txt
#SBATCH --error=stderr_fit_motion.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

. ../motion_diffusion/bin/activate

python train.py

