#!/bin/bash
#SBATCH --job-name=nli_subtask4
#SBATCH -o gypsum_logs/stdout/nli_micro16.txt
#SBATCH -e gypsum_logs/stderr/nli_micro16.err
#SBATCH --ntasks=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1

cd /mnt/nfs/work1/696ds-s21/ibirle/zero_shot_atsc/nli_micro/
eval "$(conda shell.bash hook)"
conda activate zero_shot_env
python3 nli_micro16_script.py
