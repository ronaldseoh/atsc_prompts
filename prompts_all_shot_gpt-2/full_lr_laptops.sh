#!/bin/bash
#SBATCH --job-name=f_lr_laptop
#SBATCH -o gypsum_logs/stdout/gpt-2_full_shot_prompt_lr_concatenate_atsc_laptops.txt
#SBATCH -e gypsum_logs/stderr/gpt-2_full_shot_prompt_lr_concatenate_atsc_laptops.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python gpt-2_full_shot_prompt_lr_concatenate_atsc_laptops.py