#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

python src/enigma_training_script.py --dataset_type=constcaesar --max_sentences=5000 --epochs=20 --batch_size=64 --model_type='google/byt5-small' --preserve_spaces --learning_rate=0.001
# python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=20 --batch_size=16 --model_type='t5-small' --learning_rate=0.001 --preserve_spaces