#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
name=$1
# split 
name_without_extension=${name%.*}

jupyter nbconvert --to python "$name"

python "$name_without_extension".py

# copy the script to the log directory, with appended slurm job id
mkdir -p logs/"$SLURM_JOB_ID"
cp "$name_without_extension".py logs/"$SLURM_JOB_ID"/"$name_without_extension"_"$SLURM_JOB_ID".py

#TODO: validate that this works