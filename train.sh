#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8


export WANDB_API_KEY="your key"


wandb login $WANDB_API_KEY

source ~/.bashrc

conda activate PyTorch


export TORCH_HOME="your path"

cd "your path"
python train.py
