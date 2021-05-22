#!/bin/bash
#SBATCH --job-name=pooled
#SBATCH -o gypsum_logs/stdout/pooled.txt
#SBATCH -e gypsum_logs/stderr/pooled.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

python bert_16_shot_no_prompt_pooled_lr_atsc_laptops.py
python bert_16_shot_no_prompt_pooled_lr_atsc_restaurants.py
python bert_64_shot_no_prompt_pooled_lr_atsc_laptops.py
python bert_64_shot_no_prompt_pooled_lr_atsc_restaurants.py
python bert_256_shot_no_prompt_pooled_lr_atsc_laptops.py
python bert_256_shot_no_prompt_pooled_lr_atsc_restaurants.py
python bert_1024_shot_no_prompt_pooled_lr_atsc_laptops.py
python bert_1024_shot_no_prompt_pooled_lr_atsc_restaurants.py
