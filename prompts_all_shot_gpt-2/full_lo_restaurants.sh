#!/bin/bash
#SBATCH --job-name=f_lo_rest
#SBATCH -o gypsum_logs/stdout/gpt-2_full_shot_prompt_logit_softmax_sum_logits_atsc_restaurants.txt
#SBATCH -e gypsum_logs/stderr/gpt-2_full_shot_prompt_logit_softmax_sum_logits_atsc_restaurants.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python gpt-2_full_shot_prompt_logit_softmax_sum_logits_atsc_restaurants.py