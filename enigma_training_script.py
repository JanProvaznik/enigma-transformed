import argparse
from torch.utils.data import random_split
from transformers import T5ForConditionalGeneration, T5Config, Seq2SeqTrainingArguments, Seq2SeqTrainer
from enigma.machine import EnigmaMachine
import os
import datetime 
import re
import torch
import wandb
from transformers import ByT5Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer
MAX_SEQ_LEN = 200


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--model_type", default="google/byt5-small", type=str, help="")
parser.add_argument("--model_path", default="model.bin",type=str, help="")
parser.add_argument("--dataset_path", default="news.2012.en.shuffled.deduped", type=str, help="Training data file.")
parser.add_argument("--download_dataset", default="", type=str, help="Training data file.")
# parser.add_argument("--dataset_preprocessed", default=..., type=str, help="Preprocessed training data file.")
# dataset type [constenigma, randomenigma]
parser.add_argument("--dataset_type", default="constenigma", type=str, help="Training data file.")

parser.add_argument("--evaluate", default=False, action="store_true", help="")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to train on.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# weight decay
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
# learning rate
parser.add_argument("--learning_rate", default=0.00005, type=float, help="Learning rate.")

# def download_dataset(args):
#     # download from link and unzip to dataset_path
#     import requests
#     import zipfile

#     url = args.download_dataset
#     r = requests.get(url, allow_redirects=True)
#     open('data.zip', 'wb').write(r.content)

class EnigmaDataset(Dataset):
    # only letters and capitalize
    def only_letters(self, text):
        return re.sub(r'[^A-Za-z]+', '', text).upper()

    def __init__(self, data_file: str, encrypt_function, tokenizer: PreTrainedTokenizer, max_length: int = 512, max_sentences:int = 10000) -> None:
        self.data = []
        self.encrypt_function = encrypt_function
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        with open(data_file, "r") as file:
            i = 0
            for line in file:
                text = line.strip()
                text = self.only_letters(text)
                encrypted_text = self.encrypt_function(text)
                tokenized_encrypted_text = self.tokenizer.encode(encrypted_text, max_length=self.max_length, padding='max_length', truncation=True)
                tokenized_text = self.tokenizer.encode(text, max_length=self.max_length, padding='max_length', truncation=True)
                self.data.append((tokenized_encrypted_text, tokenized_text))
                i+=1
                if i>=max_sentences:
                    break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.data[index]

class CustomDataCollator(object):
    def __call__(self, batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = zip(*batch)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

# TODO don't create a new machine for sentence
def encrypt_all_the_same(text):
    machine = EnigmaMachine.from_key_sheet(
       rotors='I II III',
       reflector='B',
       ring_settings=[0, 0, 0],
       plugboard_settings=None)
    start_display = 'ABC'
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"
import random
rand = random.Random(42)

# TODO don't create a new machine for sentence and use randomness correctly
def encrypt_random(text, seed=42):
    machine = EnigmaMachine.from_key_sheet(
        rotors='I II III',
        reflector='B',
        ring_settings=[0,0,0],
        plugboard_settings=None)
    start_display = ''.join(rand.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(3))
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"


def evaluate(args, model = None,tokenizer = None, data=None):
    # load
    if model is None:
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    if tokenizer is None:
        tokenizer = ByT5Tokenizer.from_pretrained(args.model_type)
    if data is None:
        data = EnigmaDataset(args.dataset_path, encrypt_all_the_same, tokenizer, max_length=MAX_SEQ_LEN, max_sentences=args.max_sentences)

    # for i in range(10):
    #     tokenized_encrypted_text, tokenized_gold_label = test[i]
    #     input_ids = torch.tensor(tokenized_encrypted_text, dtype=torch.long).unsqueeze(0)

    #     with torch.no_grad():
    #         outputs = trainer.generate(input_ids)

    #     predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     gold_label = tokenizer.decode(tokenized_gold_label, skip_special_tokens=True)

    #     print(f"Example {i + 1}:")
    #     print(f"Predicted: {predicted_text}")
    #     print(f"Gold Label: {gold_label}")
    #     print("=" * 80)
    


# load model and train it more
def train_pretrained(args):
    pass

def main(args):

    if args.evaluate:
        evaluate(args)
        return
    
    # nice logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    # create logdir
    os.makedirs(args.logdir, exist_ok=True)
    
    wandb.init(project="enigma-transformed", name=args.logdir)


    tokenizer = ByT5Tokenizer.from_pretrained(args.model_type)

    if args.dataset_type == "constenigma":
        fn = encrypt_all_the_same
    elif args.dataset_type == "randomenigma":
        fn = encrypt_random


    enigma_dataset = EnigmaDataset(args.dataset_path, fn, tokenizer, max_length=MAX_SEQ_LEN, max_sentences=args.max_sentences)

    # Access an example pair from the dataset
    encrypted_text, original_text = enigma_dataset[0]


    train_dataset, val_dataset, test = random_split(enigma_dataset, [int(len(enigma_dataset)*0.8), int(len(enigma_dataset)*0.1), int(len(enigma_dataset)*0.1)])

    config = T5Config.from_pretrained(args.model_type)
    config.tie_word_embeddings = False

    model = T5ForConditionalGeneration(config)

    # Create training arguments and data collator
    # weight decay? lr? warmup?
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.logdir+"byt5_output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # save_steps=10_000, # ???
        # save_total_limit=2, 
        evaluation_strategy="epoch", # ???
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,

    )
    ## TODO: understand what is a collator and why it is needed (I just put this here so that it wouldn't crash)
    data_collator = CustomDataCollator() # ???

    # Create trainer and train the model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save model
    trainer.save_model(args.logdir + args.model_path)

    evaluate(args, model, tokenizer, test)
    # trainer.evaluate(test_dataset=test)

    # model = T5ForConditionalGeneration.from_pretrained("byt5_output/checkpoint-10000")





if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
