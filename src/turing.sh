#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -J "Patch Train"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H100

python3 train.py --data_root ./data/ --train_list ./Src/list/small_train_list2.txt --lr 1e-2 --patch_size 256 --target_disp 70 -b 128 --name turing-batch-1 --num_epochs 1000
