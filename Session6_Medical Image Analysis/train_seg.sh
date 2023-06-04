#!/bin/bash

#SBATCH --job-name=CXR_seg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=leelabgpu
#SBATCH --time=0
#SBATCH --mem=125GB
#SBATCH --cpus-per-task=32
#SBATCH -o ./slurm-%A_%a.out

source /home/n1/leelabguest/.bashrc

echo "Unet segmentation with CHN data."
python3 segmentation.py train
#python3 segmentation.py test -m 199
