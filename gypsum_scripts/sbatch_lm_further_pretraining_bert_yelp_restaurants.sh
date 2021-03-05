#!/bin/bash
#
#SBATCH --job-name=lm_further_pretraining_bert_yelp_restaurants
#SBATCH --output=gypsum_logs/lm_further_pretraining_bert_yelp_restaurants_%j.txt
#SBATCH -e gypsum_logs/lm_further_pretraining_bert_yelp_restaurants_%j.err
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8

eval "$(conda shell.bash hook)"
conda activate zeroshotatsc

jupyter nbcovert --to notebook --execute /mnt/nfs/work1/696ds-s21/bseoh/lexalyticslm_further_pretraining_bert_yelp_restaurants.ipynb
