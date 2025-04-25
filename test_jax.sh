#!/bin/bash
#SBATCH --job-name=test_jax
#SBATCH --partition=kempner,kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 1-23:00                                                          # Runtime in D-HH:MM
#SBATCH -o /n/holylabs/LABS/sham_lab/Lab/lillian/DP-AdamW/results/output_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/holylabs/LABS/sham_lab/Lab/lillian/DP-AdamW/results/output_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account kempner_sham_lab


CODE_DIR=/n/holylabs/LABS/sham_lab/Lab/lillian/DP-AdamW
pushd $CODE_DIR

echo "Activating conda environment"
module load Mambaforge/23.11.0-fasrc01
conda activate dp-adamw
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cuda/12.4

python jax_implementation/main.py --seed 1024 --epochs 70 --progress_bar --batch_size 1024 \
    --learning_rate 0.000005 --dataset CIFAR10 --classifier_model CNN5 \
    --activation tanh --dp_l2_norm_clip 3.0 --trainer DPAdam --adam_corr \
    --eps_root 0.000000001 --target_eps 1.0 --exp_entity cs2080 --exp_proj dp-adamw --exp_group jax --exp_name cifar10_cnn5_dpadam_eps_1

echo "DONE"