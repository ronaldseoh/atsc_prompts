#!/bin/bash
#SBATCH --job-name=nli_few_in_domain_laptops
#SBATCH -o gypsum_logs/stdout/nli_few_in_domain_laptops.txt
#SBATCH -e gypsum_logs/stderr/nli_few_in_domain_restaurants.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1

cd /mnt/nfs/work1/696ds-s21/ibirle/zero_shot_atsc/nli_experiments/
eval "$(conda shell.bash hook)"
conda activate zero_shot_env
python3 nli_few_shot_in_domain_laptops_script.py