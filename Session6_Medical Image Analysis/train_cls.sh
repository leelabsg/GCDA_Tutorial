#!/bin/bash

#SBATCH --job-name=CXR_cls
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=leelabgpu
#SBATCH --time=0
#SBATCH --mem=125GB
#SBATCH --cpus-per-task=32
#SBATCH -o ./slurm-%A_%a.out

source /home/n1/leelabguest/.bashrc

echo "CNN classification with CheXpert data."
python3 classification.py train
#python3 classification.py test -m 8
