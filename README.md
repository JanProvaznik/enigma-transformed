# enigma-transformed

## How to run 
### Slurm cluster
- basic setting: `sbatch -p gpu -c1 --gpus=1 --mem=16G <bash_script_path>`
- use `run_notebook.sh <notebook_path>` to run a Jupyter notebook on a slurm cluster

### Colab
- clone this repo and use the desired `.ipynb` files
- add to each notebook: 
```
!git clone https://github.com/JanProvaznik/enigma-transformed
!pip install transformers[torch] Levenshtein py-enigma
```

## Docs
### Source code
#### reproducible/
- for each experiment contians a notebook that can be used to reproduce it
    - `01_copy_random_text.ipynb` - trains model to copy on random strings
    - `02_copy_news.ipynb` - trains model to copy on news sentences
    - `03_caesar_random_text.ipynb` - trains model to decrypt constant caesar cipher (only one setting) on random strings
    - `04_caesar_news.ipynb` - trains model to decrypt constant caesar cipher (only one setting) on news sentences
    - `05_triple_caesar_news.ipynb` - trains model to decrypt triple caesar cipher (3 settings) on news sentences
    - `06_all_caesar_hint_random_text.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on random strings given that the first word is known (this can be interpreted as a task prefix)
    - `07_all_caesar_hint_news.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on news sentences given that the first word is known (this can be interpreted as a task prefix)
    - `08_all_caesar_news.ipynb` - trains model to decrypt all caesar ciphers (26 settings) on news sentences
        - first model that does not have a clear explanation how it works 

#### src/
- reusable code
##### `ciphers.py`
- cipher implementations
    - [Caesar](https://en.wikipedia.org/wiki/Caesar_cipher) 
       - with hint
    - [Vigenere](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher)
    - [Enigma (simplified)](https://en.wikipedia.org/wiki/Cryptanalysis_of_the_Enigma#The_Enigma_machine)

##### `preprocessing.py`:
- text data preprocessing functions
        
##### `utils.py`: 
- random utility functions
     - downloading newscrawl dataset

`run_notebook.sh` - script for running a notebook on a slurm cluster 

#### divergent
- for exploratory code/experiments that don't fit into the project anymore
##### lens_train.py: 
- script to replicate reproducible/03 with [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library and minimal amount of resources (only 1 layer transformer)


## Usual experiment pipeline
0. get data from the internet or generate it
1. filter the data for the given experiment (e.g. only sentences 100-200 characters long)
2. preprocess the data: only a-z + spaces, trim/pad to desired length
3. make *training pairs*: take each line of preprocessed data and encrypt it using an encryption algorithm, (unencrypted, encrypted), also possible to add a prefix/suffix
4. create a model, set hyperparameters
5. train the model on the training pairs
6. save the model
7. evaluate the performance of the model (during training and after training)
    - e.g. edit distances

## Training hyperparameters:
### number of training examples
- the more the better (if model sees all cipher configuations it won't have to generalize the cipher procedure, but only detect which configuation is used and apply it)

### trainable parameters in model
- the more the better, but we're limited by the GPU memory (and time), bigger models will use have harder time to use big batch sizes
### epochs
- the more the better, but we're limited by the time we have

### batch size
- if too low, model won't be able to learn any patterns
- generally the higher the better, but we're limited by the GPU memory 
    - trick: use gradient accumulation 
        - e.g. if we have batch size 16 and gradient accumulation 16 -> the effective batch size is 256
            - we pay for this by having to train 16 times longer to obtain same amount of updates
- in literature we can find batch sizes of tokens and not examples, to translate that it's just `effective_batch_size * num_tokens_in_example` in each experiment, where `num_tokens_in_example` is 2x `dataset_max_len` in each reproducible notebook (because we have both input and output)

### learning rate
- has to be quite high because we're not fine-tuning for a language task but for a quite strange translaton
- usually use the huggingface default LR schedule for `Seq2SeqTrainer` (linear decay); and set relative warmup (e.g. 0.2 of total steps) 