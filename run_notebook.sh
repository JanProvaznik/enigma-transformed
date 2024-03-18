#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
set -eux
date
name=$1
# split 
name_without_extension=${name%.*}
logdir=logs/slurm_"$SLURM_JOB_ID"

mkdir -p "$logdir"
cp -r ./src "$logdir"

jupyter nbconvert --to python "$name"

script_name=script_"$SLURM_JOB_ID".py
mv "$name_without_extension".py "$logdir"/"$script_name"

echo "$@"
python "$logdir"/"$script_name" "$@"


# copy the script to the log directory, with appended slurm job id
date