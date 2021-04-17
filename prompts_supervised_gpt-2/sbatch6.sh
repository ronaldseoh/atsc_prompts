#!/bin/bash
#SBATCH --job-name=6sbatch
#SBATCH -o gypsum_logs/stdout/gpt2_prompt_logit_softmax_atsc_laptops.txt
#SBATCH -e gypsum_logs/stderr/gpt2_prompt_logit_softmax_atsc_laptops.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

#python gpt2_prompt_logit_softmax_atsc_restaurants_multiple_prompts.py                           
#python gpt2_prompt_logit_softmax_atsc_restaurants_multiple_prompts_cross_domain.py
#python gpt2_prompt_logit_softmax_atsc_laptops_multiple_prompts.py
#python gpt2_prompt_logit_softmax_atsc_laptops_multiple_prompts_cross_domain.py

#python gpt2_prompt_logit_softmax_atsc_restaurants_cross_domain.py
python gpt2_prompt_logit_softmax_atsc_laptops.py
# python gpt2_prompt_logit_softmax_atsc_laptops_cross_domain.py  
# python gpt2_prompt_logit_softmax_atsc_restaurants.py 