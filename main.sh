#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=12G
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -J full_test
#SBATCH -o logfiles/full_test.out.%j
#SBATCH -e logfiles/full_test.err.%j
#SBATCH --account=project_2005099
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhixu.gu@helsinki.fi


module purge
module load pytorch
python main.py
