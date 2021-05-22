#!/bin/bash
#SBATCH --job-name=256_laptop
#SBATCH -o gypsum_logs/stdout/gpt-2_256_shot_laptops.txt
#SBATCH -e gypsum_logs/stderr/gpt-2_256_shot_laptops.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python gpt-2_256_shot_prompt_logit_softmax_sum_logits_atsc_laptops.py
python gpt-2_256_shot_prompt_lr_concatenate_atsc_laptops.py