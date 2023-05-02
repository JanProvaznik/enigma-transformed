#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# python3 enigma_training_script.py --dataset_type=constenigma --max_sentences=10 --epochs=1 
echo "trivial test complete"
echo "==================================="
echo "training constenigma"
# python enigma_training_script.py --dataset_type=constenigma --max_sentences=10000 --epochs=3 --batch_size=32
python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=64 --model_type='t5-small' --learning_rate=0.01 --wandb

python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=128 --model_type='t5-small' --learning_rate=0.01 --wandb

python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=64 --model_type='t5-small' --learning_rate=0.001 --wandb

python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=128 --model_type='t5-small' --learning_rate=0.001 --wandb

python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=64 --model_type='t5-small' --learning_rate=0.0001 --wandb

python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=128 --model_type='t5-small' --learning_rate=0.0001 --wandb

# python enigma_training_script.py --dataset_type=constenigma --max_sentences=50000 --epochs=5 --batch_size=64 --model_type='t5-base' --wandb
# # python enigma_training_script.py --dataset_type=constenigma --max_sentences=10000 --epochs=5 --batch_size=32 


# python enigma_training_script.py --dataset_type=constenigma --max_sentences=100000 --epochs=10 --batch_size=128
# echo "constenigma complete"
# echo "==================================="

# echo "training randomenigma"
# python enigma_training_script.py --dataset_type=randomenigma --max_sentences=10000 --epochs=3 --batch_size=16
# python enigma_training_script.py --dataset_type=randomenigma --max_sentences=50000 --epochs=3 --batch_size=16

# python enigma_training_script.py --dataset_type=randomenigma --max_sentences=10000 --epochs=5 --batch_size=32 
# python enigma_training_script.py --dataset_type=randomenigma --max_sentences=50000 --epochs=5 --batch_size=32 
# python enigma_training_script.py --dataset_type=randomenigma --max_sentences=100000 --epochs=5 --batch_size=32

# echo "randomenigma complete"

# # diff for each experiment

