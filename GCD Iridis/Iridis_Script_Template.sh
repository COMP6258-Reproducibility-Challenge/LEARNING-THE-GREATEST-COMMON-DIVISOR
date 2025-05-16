#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -—mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lon1f17@soton.ac.uk
#SBATCH --time=24:0:00

module load conda/py3-latest 
conda activate GCD-local
python train.py
