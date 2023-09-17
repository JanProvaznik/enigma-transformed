# enigma-transformed

## How to run 
### Slurm cluster
- recommended to use a bash script that contains conversion and runs the script
- `sbatch -p gpu -c1 --gpus=1 --mem=16G <bash_script_path>`
- conversion: `jupyter nbconvert --to python YourNotebookFile.ipynb`

### Colab
- clone this repo and use the desired `.ipynb` files
- add to each notebook: 
```
!git clone https://github.com/JanProvaznik/enigma-transformed
!pip install transformers[torch] Levensthein py-enigma
```

## Docs
1. The **reproducible** directory for each experiment contians it's script and the results and link to the trained model
2. **src** contains reusable code
3. **data** contains evaluation data

## Usual experiment pipeline
0. get data from the internet
2. preprocess the data: only A-Z + spaces, trim/pad to desired length
3. make *training pairs*: take each line of preprocessed data and encrypt it using an encryption algorithm, (unencrypted, encrypted), also possible to add a prefix/suffix
4. create a model, set hyperparameters
5. train the model on the training pairs
6. save the model
7. evaluate the performance of the model (during training and after training)
    - e.g. edit distances
