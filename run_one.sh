#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# python src/enigma_training_script.py --dataset_type=constcaesar --max_sentences=1000 --epochs=80 --batch_size=8 --model_type='google/byt5-small' --preserve_spaces --learning_rate=0.005 --wandb
python just_copy.py

# can we even learn to copy? idk what went wrong :(
# python src/enigma_training_script.py --dataset_type=nothing --max_sentences=1000 --epochs=80 --batch_size=16 --model_type='google/byt5-small' --preserve_spaces --learning_rate=0.005 --wandb
# python src/only_pytorch_copy.py --dataset_type=nothing --max_sentences=100 --epochs=80 --batch_size=8 --preserve_spaces --dataset_path="~/enigma-transformed/test_dataset/head100.txt"


# python src/enigma_training_script.py --evaluate --model_path='logs/enigma_training_script.py-2023-05-03_231306-bs=64,dp=news.2012.en.shuffled.deduped,dt=constcaesar,dd=,d=0.1,e=15,e=False,lr=0.001,ls=linear,ms=10000,mp=model.bin,mt=google/byt5-small,ps=True,s=42,w=True,wd=0.0model.bin'
# python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=20 --batch_size=16 --model_type='t5-small' --learning_rate=0.001 --preserve_spaces