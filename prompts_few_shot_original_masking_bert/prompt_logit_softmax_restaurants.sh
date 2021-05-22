#!/bin/bash
#SBATCH --job-name=restaurants
#SBATCH -o gypsum_logs/stdout/restaurants.txt
#SBATCH -e gypsum_logs/stderr/restaurants.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python bert_16_shot_prompt_logit_softmax_sum_logits_atsc_restaurants_cross_domain.py
python bert_64_shot_prompt_logit_softmax_sum_logits_atsc_restaurants_cross_domain.py
python bert_256_shot_prompt_logit_softmax_sum_logits_atsc_restaurants_cross_domain.py
python bert_1024_shot_prompt_logit_softmax_sum_logits_atsc_restaurants_cross_domain.py
