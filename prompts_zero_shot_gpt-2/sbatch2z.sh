#!/bin/bash
#SBATCH --job-name=2zsbatch
#SBATCH -o gypsum_logs/stdout/gpt-2_zero_shot_prompt_logit_softmax_atsc_restaurants.txt
#SBATCH -e gypsum_logs/stderr/gpt-2_zero_shot_prompt_logit_softmax_atsc_restaurants.err
#SBATCH --ntasks=1
#SBATCH --partition=m40-short
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc
                       
python gpt-2_zero_shot_prompt_logit_softmax_atsc_restaurants.py