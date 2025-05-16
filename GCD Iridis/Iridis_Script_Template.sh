#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -—mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user={Southampton ID}
#SBATCH --time=24:0:00

module load conda/py3-latest 
conda activate {Enviroment Name}
python train.py
