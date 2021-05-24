#!/bin/bash
#SBATCH --job-name=nli_cross_domain_laptops
#SBATCH -o gypsum_logs/stdout/nli_cross_domain_laptops.txt
#SBATCH -e gypsum_logs/stderr/nli_cross_domain_laptops.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1

cd /mnt/nfs/work1/696ds-s21/ibirle/zero_shot_atsc/nli_experiments/
eval "$(conda shell.bash hook)"
conda activate zero_shot_env
python3 nli_cross_domain_laptops_script.py
