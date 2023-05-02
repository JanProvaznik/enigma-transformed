import argparse
from datasets import load_dataset
from transformers import TrainerCallback
import Levenshtein
from torch.utils.data import random_split
from transformers import T5ForConditionalGeneration, T5Config, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from transformers import DataCollatorWithPadding, pipeline
import datetime 
import re
import torch
import torch.cuda
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, AdamW, Adafactor
from typing import List, Tuple, Dict
from ciphers import encrypt_all_the_same, encrypt_random, caesar, caesar_random, caesar_random_hint
from transformers import pipeline
from EditDistanceCallback import EditDistanceCallback
MAX_SEQ_LEN = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--model_type", default="google/byt5-small", type=str, help="")
parser.add_argument("--model_path", default="model.bin",type=str, help="")
parser.add_argument("--dataset_path", default="news.2012.en.shuffled.deduped", type=str, help="Training data file.")
parser.add_argument("--download_dataset", default="", type=str, help="Training data file.")
# parser.add_argument("--dataset_preprocessed", default=..., type=str, help="Preprocessed training data file.")
# dataset type [constenigma, randomenigma]
parser.add_argument("--dataset_type", default="constenigma", type=str, help="Training data file.")
# preserve spaces
parser.add_argument("--preserve_spaces", default=False, action="store_true", help="")


parser.add_argument("--evaluate", default=False, action="store_true", help="")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to train on.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# weight decay
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
# dropout
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
# learning rate
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
# lr schedule options: [linear, constant, cosine]
parser.add_argument("--lr_scheduler", default="linear", type=str, help="Learning rate schedule.")
# wandb
parser.add_argument("--wandb", default=False, action="store_true", help="Use wandb.")

    
def only_letters(text, preserve_spaces=False):
        if preserve_spaces:
            return re.sub(r'[^A-Za-z ]+', '', text).upper()
        return re.sub(r'[^A-Za-z]+', '', text).upper()


def evaluate(args, model = None, tokenizer = None, data=None):
    # load
    if model is None:
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if data is None:
        pass
        # data = EnigmaDataset(args.dataset_path, encrypt_all_the_same, tokenizer, max_length=MAX_SEQ_LEN, max_sentences=args.max_sentences)


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
    if args.wandb:
        import wandb
        # get key from os
        key = os.environ.get("WANDB_API_KEY")
        wandb.login(key=key)

        wandb.init(project="enigma-transformed", name=args.logdir)
    else:
        os.environ["WANDB_DISABLED"] = "true"

    from transformers import ByT5Tokenizer
    tokenizer = ByT5Tokenizer()
    # check how does it handle End of sequence symbol
    # dropout 0.1 is applied automatically
    # warmup from the paper
    # first big learning rate 0.01, then smaller .001
    # idk why other lr schedules would be better and how to think about it 

    if args.dataset_type == "constenigma":
        fn = encrypt_all_the_same
    elif args.dataset_type == "randomenigma":
        fn = encrypt_random
    elif args.dataset_type == "constcaesar":
        fn = caesar
    elif args.dataset_type == "hintcaesar":
        fn = caesar_random_hint
    elif args.dataset_type == "randomcaesar":
        fn = caesar_random
    #TODO: vignere?


    # load dataset: a file that contains one sentence per line
    line_dataset = load_dataset('text', data_files=args.dataset_path)
    # take only args.max_sentences sentences
    line_dataset = line_dataset['train'].select(range(args.max_sentences))

    def preprocess_dataset(examples,fn):
        original = [only_letters(text, args.preserve_spaces)[:MAX_SEQ_LEN] for text in examples["text"]]
        encrypted = [fn(text)[:MAX_SEQ_LEN] for text in examples["text"]]
        return tokenizer.prepare_seq2seq_batch(encrypted, original, truncation=True,return_tensors="pt",padding='max_length', max_length=MAX_SEQ_LEN)


    enigma_dataset = line_dataset.map(preprocess_dataset,fn_kwargs={'fn':fn}, batched=True)

    train_val_ratio = 0.8
    split = enigma_dataset.train_test_split(train_size=train_val_ratio , shuffle=True, seed=args.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    config = T5Config.from_pretrained(args.model_type)
    config.tie_word_embeddings = False
    # byT5-smaller -> 300m parameters to 100m
    if args.model_type == 'google/byT5-small':
        config.d_ff = config.d_ff // 2
        config.d_model = config.d_model // 2
        config.num_heads = 10
    
    # note: this is probably the same as AutoModelForSeq2SeqLM
    model = T5ForConditionalGeneration(config).to(device)

    real_batch_size = 4
    gradient_accumulation_steps = args.batch_size // real_batch_size
    # weight decay? lr? warmup?
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.logdir+"trained_model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=real_batch_size,
        per_device_eval_batch_size=real_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_type="adafactor",
        warmup_ratio = 0.05,
        
        save_steps=10_000, 
        save_total_limit=2, 
        evaluation_strategy="epoch",

        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
    )

    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = Adafactor(model.parameters(), 
    # lr=args.learning_rate,
    #  weight_decay=args.weight_decay
    #  )
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding='max_length',  max_length=MAX_SEQ_LEN, return_tensors="pt", model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        
        # optimizers=(optimizer, None), # ??? how to use this?
        callbacks=[EditDistanceCallback(tokenizer, val_dataset, train_dataset,seq_len=MAX_SEQ_LEN,device=device)] 
        
    )
    # s = args.lr_scheduler
    # scheduler = get_scheduler(
    #     name=s,
    #     num_warmup_steps=0,
    #     num_training_steps=args.epochs * len(train_dataset) // args.batch_size,
    #     optimizer=trainer.optimizer,
    # )
    # trainer.lr_scheduler = scheduler
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()

    trainer.train()

    # use generate: it can do beam search, (even sampling )
    # save model
    trainer.save_model(args.logdir + args.model_path)

    # evaluate(args, model, tokenizer, test)
    # trainer.evaluate(test_dataset=test) ???

    # from transformers import pipeline
    # pipelines can be used for easier access
    # generator = pipeline("translation", model=model, tokenizer=tokenizer)
    # # Use the model to generate S2 for a given S1
    # S1 = raw_enigma_dataset[0]["original_text"]
    # generated_text = generator(S1)[0]["generated_text"] 
    # S2 = generated_text[len(S1):].strip()
    # print(S2)



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
