# Enigma Transformed 

## Abstract
We explore the possibility of using a pre-trained Transformer language model to decrypt ciphers. The aim is also to discover what linguistic features of a text the model learns to use by measuring correlations of error rates.

1. create evaluation dataset with linguistic properties
2. train model on decipherment
3. evaluate correlations and predictability from linguistic properties

## Docs
### How to run 
- get dependencies and install local code
```
pip install -r requirements.txt
pip install -e .
```
#### Slurm cluster
- basic setting: `sbatch -p gpu -c1 --gpus=1 --mem=16G <bash_script_path>`
- use `./run_notebook.sh <notebook_path>` to run a Jupyter notebook on a slurm cluster

#### Colab
- clone this repo and use the desired `.ipynb` files
- add to each notebook: 
```
!git clone https://github.com/JanProvaznik/enigma-transformed
!pip install transformers[torch] Levenshtein py-enigma
```

### Source code
#### reproducible/
- scripts for fine-tuning ByT5 on ciphers
- Used in thesis: 
    - `21_vignere3_noisy_random_news_en.ipynb`, `22_vignere3_noisy_random_news_de.ipynb`, `23_vignere3_noisy_random_news_cs.ipynb` finetuning ByT5 to decrypt a random 3-letter key Vignere cipher on news sentences
    - `24_const_noisy_enigma_news_cs.ipynb`, `25_const_noisy_enigma_news_de.ipynb`, `26_const_noisy_enigma_news_en.ipynb` finetuning ByT5 to decrypt a simplified Enigma cipher on news sentences


- old experiments in `unused/` 
    - `01_copy_random_text.ipynb` - trains model to copy on random strings
    - `02_copy_news.ipynb` - trains model to copy on news sentences
    - `03_caesar_random_text.ipynb` - trains model to decrypt constant caesar cipher (only one setting) on random strings
    - `04_caesar_news.ipynb` - trains model to decrypt constant caesar cipher (only one setting) on news sentences
    - `05_triple_caesar_news.ipynb` - trains model to decrypt triple caesar cipher (3 settings) on news sentences
    - `06_all_caesar_hint_random_text.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on random strings given that the first word is known (this can be interpreted as a task prefix)
    - `07_all_caesar_hint_news.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on news sentences given that the first word is known (this can be interpreted as a task prefix)
    - `08_all_caesar_news.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on news sentences
        - first model that does not have a clear explanation how it works 
    - `09_vignere2_news` - trains model to decrypt constant 2 letter vignere cipher on news sentences
    - `10_vignere3_news` - trains model to decrypt constant 3 letter vignere cipher on news sentences
    - `11_vignere_long_news` - trains model to decrypt constant vignere cipher with key 'helloworld' on news sentences
    - `12_vignere_multiple_news` - trains model to decrypt 2 letter vignere cipher, with 3 settings on news sentences
    - ... and more

#### data/
- `weird_classify.ipynb` and `lang_classify.ipynb` - filter out sentences
- `measure_dataset(cs,de).ipynb` - annotate linguistic properties of a dataset
- `evaluation_batchedgpuevaluate_other_models.ipynb` - inference decipherments by different model checkpoints

#### analysis/
- `loss_curves.ipynb` - visualize loss curves of training with error density at checkpoints
- `corr_matrices.ipynb` - to create correlation matrices of error rates and linguistic properties
- `evo_correlation.ipynb` - to graph of evolution of correlations of error rates and linguistic properties
- `pred_shap.ipynb` - predict error rates with simple ML and analyze with shap


#### run_notebook.sh and run_notebook4gpu.sh
- script for running training or inference notebooks on a slurm cluster with GPUs


#### src/
- reusable code
##### `ciphers.py`
- cipher implementations
    - [Caesar](https://en.wikipedia.org/wiki/Caesar_cipher) 
       - with hint
    - [Vigenere](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher)
    - [Enigma (simplified)](https://en.wikipedia.org/wiki/Cryptanalysis_of_the_Enigma#The_Enigma_machine)
##### `preprocessing.py`
- text data preprocessing functions
##### `utils.py`: 
- random utility functions
     - downloading newscrawl dataset
     - printing error rate stats per example in model evaluation
##### `ByT5Dataset.py`
- classes for datasets for the experiments that work in a way that one puts a list of strings in them and it applies the cipher and tokenizes them to be used in the model
    - `ByT5Dataset` - superclass with functionality
    - other inherit and specify a preprocess function

#### divergent/
- for exploratory code/experiments that don't fit into the project anymore
##### `lens_train.py`
- script to replicate reproducible/03 with [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library and minimal amount of resources (only 1 layer transformer)

### What happens when training
0. get data from the internet or generate it
1. filter the data for the given experiment (e.g. only sentences 100-200 characters long)
2. preprocess the data: only a-z + spaces, trim/pad to desired length
3. make *training pairs*: take each line of preprocessed data and encrypt it using an encryption algorithm, (unencrypted, encrypted), also possible to add a prefix/suffix
4. create a model, set hyperparameters
5. train the model on the training pairs
6. save the model
7. evaluate the performance of the model (during training and after training)

### Meta info
- uses lowercased letters in all experiments
- vigenere preserves spaces, enigma replaces them with X
- using the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) to measure the error rate of the model on an evaluation dataset
- using the [statmt newscrawl](https://statmt.org/) dataset to obtain real world text for training and evaluation
- using the [Huggingface Transformers library](https://huggingface.co/transformers/) running on [PyTorch](https://pytorch.org/)
- using pretrained [ByT5](https://arxiv.org/abs/2105.13626) character level models and fine-tuning them on ciphers

### Training hyperparameters:
#### number of training examples
- the more the better (if model sees all cipher configuations it won't have to generalize the cipher procedure, but only detect which configuation is used and apply it)

#### trainable parameters in model
- the more the better, but we're limited by the GPU memory (and time), bigger models will use have harder time  using big batch sizes
#### epochs
- the more the better, but we're limited by the time we have

#### batch size
- if too low, models won't be able to learn any patterns
- generally the higher the better, but we're limited by the GPU memory 
    - trick: use gradient accumulation 
        - e.g. if we have batch size 16 and gradient accumulation 16 -> the effective batch size is 256
            - we pay for this by having to train 16 times longer to get the same amount of updates
- in literature we can find batch sizes of tokens and not examples, to translate that it's just `effective_batch_size * num_tokens_in_example` in each experiment, where `num_tokens_in_example` is 2x `dataset_max_len` in each reproducible notebook (because we have both input and output)

#### learning rate
- has to be quite high because we're not fine-tuning for a language task but for a quite strange translaton
- usually use the huggingface default LR schedule for `Seq2SeqTrainer` (linear decay); and set relative warmup (e.g. 0.2 of total steps) 


