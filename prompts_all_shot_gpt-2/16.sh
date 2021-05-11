#!/bin/bash
#SBATCH --job-name=16
#SBATCH -o gypsum_logs/stdout/gpt-2_16_shot.txt
#SBATCH -e gypsum_logs/stderr/gpt-2_16_shot.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python gpt-2_16_shot_prompt_logit_softmax_sum_logits_atsc_restaurants.py
python gpt-2_16_shot_prompt_logit_softmax_sum_logits_atsc_laptops.py
python gpt-2_16_shot_prompt_lr_concatenate_atsc_laptops.py
python gpt-2_16_shot_prompt_lr_concatenate_atsc_restaurants.py