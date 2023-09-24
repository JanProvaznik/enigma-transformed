#!/usr/bin/env python
# coding: utf-8

# demonstrates how to train a transformer model using TransformerLens library 
# to decrypt caesar cipher with hint (start of sequence constant) with a miniature transformer (1 layer)
# - shows that is's a simple task and 
# -we can both have lots of parameters and low training time or few parameters and high training time (my other experiments) to get this outcome

# modified from: https://github.com/MatthewBaggins/one-attention-head-is-all-you-need/

# Fixed length of list to be deciphered
LIST_LENGTH = 50

# Size of vocabulary
# 26 letters, space, BOS token, EOS token, MID token, PAD token 
D_VOCAB = 32

# Attention only? (False -> model includes MLPs)
ATTN_ONLY = False

# Model dimenions
N_LAYERS = 1
N_HEADS = 1
D_MODEL = 128 # d_model in hf
D_HEAD = 32 # d_kv in hf
D_MLP = 256 # d_ff in hf 


if ATTN_ONLY:
    D_MLP = None

# Default batch size
TRAIN_BATCH_SIZE = 256

print(f"{LIST_LENGTH = }", f"{ATTN_ONLY = }", f"{N_LAYERS = }", f"{N_HEADS = }", f"{D_MODEL = }", f"{D_HEAD = }", f"{D_MLP = }", f"{TRAIN_BATCH_SIZE= }")

from dataclasses import dataclass
from datetime import datetime as dt
import os
import pickle
import random
from typing import cast, Generator, Literal

import torch
from torch import tensor, Tensor
from transformer_lens import HookedTransformerConfig, HookedTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE = }")

# Seeds to generate training, validation, and test data
TRAIN_SEED = 42
VAL_SEED = 66
TEST_SEED = 1729

# Context length: [start, *(unsorted_)list_length, mid, *(sorted_)list_length]
N_CTX = 2 * LIST_LENGTH + 2

# 26 letters and space
VOCAB_MIN_ID = 0
VOCAB_MAX_ID = 26

# START token is D_VOCAB - 2 and MID token is D_VOCAB - 1
START_TOKEN_ID = VOCAB_MAX_ID + 1
MID_TOKEN_ID = VOCAB_MAX_ID + 2
END_TOKEN_ID = VOCAB_MAX_ID + 3
PAD_TOKEN_ID = VOCAB_MAX_ID + 4



# ### Data generator and datasets

# add 'hello' to each example so the model has something constant to get a clue of the key
def generate_list(batch_size: int) -> Tensor:
    randoms = torch.randint(VOCAB_MIN_ID, VOCAB_MAX_ID, (batch_size, LIST_LENGTH))
    # replace first 5 elements in each row with: [7 4 11 11 14]
    randoms[:, :5] = torch.tensor([7, 4, 11, 11, 14])
    return randoms.to(DEVICE)

caesar_min = 0
caesar_max = 25
print (f"{caesar_min = }", f"{caesar_max = }")
def modify(x: Tensor, ceasar=5) -> Tensor:
    """Add caesar to each element of x from 0 to 26 and then mod 26"""
    y = x.clone().detach()
    mask = (y >=0) & (y <26 )
    # r = random.randint(caesar_min,caesar_max)
    r=11
    # y[mask] = (y[mask] + caesar) % 26
    y[mask] = (y[mask] + r) % 26 
    return y


# General generator
def make_data_gen(
    *,
    batch_size: int = TRAIN_BATCH_SIZE,
    dataset: Literal["train", "val", "test"], # probably this arg needs a better name,
) -> Generator[Tensor, None, None]:
    assert dataset in ("train", "val", "test")
    if dataset == "train":
        seed = TRAIN_SEED
    elif dataset == "val":
        seed = VAL_SEED
    else: # test
        seed = TEST_SEED
    torch.manual_seed(seed)
    while True:
        # Generate random numbers
        x = generate_list(batch_size)
        # modify 
        x_modified = modify(x) # just copy now, then replace with substitution cipher fn
        # START tokens
        x_start = START_TOKEN_ID * torch.ones(batch_size, dtype=torch.int32).reshape(batch_size, -1).to(DEVICE)
        # MID tokens
        x_mid = MID_TOKEN_ID * torch.ones(batch_size, dtype=torch.int32).reshape(batch_size, -1).to(DEVICE)
        # NOTE: swap x and x_modified to simulate decipherment
        x, x_modified = x_modified, x
        yield torch.cat((x_start, x, x_mid, x_modified), dim=1)


# Training data generator (kinda wrapper)
def make_train_gen() -> Generator[Tensor, None, None]:
    """Make generator of training data"""
    return make_data_gen(batch_size=TRAIN_BATCH_SIZE, dataset="train")

# Validation and test data

val_data = next(make_data_gen(batch_size=1000, dataset="val"))
test_data = next(make_data_gen(batch_size=1000, dataset="test"))


# ### Loss function
def loss_fn(
    logits: Tensor, # [batch, pos, d_vocab] 
    tokens: Tensor, # [batch, pos] 
    return_per_token: bool = False
) -> Tensor: # scalar
    sorted_start_pos = LIST_LENGTH + 2
    logits = logits[:, (sorted_start_pos-1):-1]
    tokens = tokens[:, sorted_start_pos : None]
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    if return_per_token:
        return -correct_log_probs
    return -correct_log_probs.mean()



def get_diff_row_inds(
    a: Tensor, # [dim1, dim2]
    b: Tensor  # [dim1, dim2]
) -> Tensor:   # [dim1]
    """Find indices of rows where a and b differ"""
    assert a.shape == b.shape
    return ((a == b).prod(dim=1) == 0).nonzero(as_tuple=True)[0]

def acc_fn(
    logits: Tensor, # [batch, pos, d_vocab]
    tokens: Tensor, # [batch, pos]
    per: Literal["token", "sequence"] = "sequence"
) -> float:
    """Compute accuracy as percentage of correct predictions"""
    sorted_start_pos = LIST_LENGTH + 2
    # Get logits of predictions for position
    logits = logits[:, (sorted_start_pos-1):-1]
    preds = logits.argmax(-1)
    tokens = tokens[:, sorted_start_pos:]
    if per == "sequence":
        return (preds == tokens).prod(dim=1).float().mean().item()
    return (preds == tokens).float().mean().item()

def validate(
    model: HookedTransformer, 
    data: Tensor # [batch, pos]
) -> float:
    """Test this model on `data`"""
    logits = model(data)
    acc = acc_fn(logits, tokens=data)
    return acc

def show_mispreds(
    model: HookedTransformer, 
    data: Tensor # [batch, pos]
) -> None:
    """Test this model on `data` and print mispredictions"""
    logits = model(data)
    sorted_start_pos = LIST_LENGTH + 2
    logits = logits[:, (sorted_start_pos-1):-1]
    tokens = data[:, sorted_start_pos:].cpu()
    preds = logits.argmax(-1).cpu()
    mispred_inds = get_diff_row_inds(tokens, preds)
    for i in mispred_inds:
        expected = tokens[i].numpy().tolist()
        predicted = preds[i].numpy().tolist()
        positions_wrong = 0
        for j in range(len(predicted)):
            if predicted[j] != expected[j]:
                positions_wrong += 1
        print(f"[{i}] {expected} | {predicted} | {positions_wrong=}")
        # print(f"[{i}] {tokens[i].numpy().tolist()} | {preds[i].numpy().tolist()}")
    print(f"{len(mispred_inds)}/{len(preds)} ({len(mispred_inds) / len(preds) :.2%})")

# ## Training

# ### Model

cfg = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=N_CTX,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=42,
    device=DEVICE,
    attn_only=ATTN_ONLY
)
model = HookedTransformer(cfg, move_to_device=True)


# ### Training setup

@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]

def converged(val_accs: list[float], n_last: int = 10) -> bool:
    return cast(bool, (tensor(val_accs[-n_last:]) == 1).all().item())

# Number of epochs
n_epochs = 40000

# Optimization
lr = 1e-3
betas = (.9, .999)
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

# Training data generator
train_gen = make_train_gen()

def train_model(model: HookedTransformer) -> TrainingHistory:
    losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        tokens = next(train_gen).to(device=DEVICE)
        logits = model(tokens)
        loss = loss_fn(logits, tokens)
        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            losses.append(loss.item())
            train_batch_acc = acc_fn(logits, tokens)
            val_acc = validate(model, val_data)
            val_loss = loss_fn(model(val_data), val_data)

            train_accuracies.append(train_batch_acc)
            val_accuracies.append(val_acc)
            print(
                f"Epoch {epoch}/{n_epochs} ({epoch / n_epochs:.0%}) : "
                f"loss = {loss.item():.4f}; {train_batch_acc=:.3%}; "
                f"{val_acc=:.3%}; lr={scheduler._last_lr[0]}" #type:ignore
            )
            # If last 10 recorded val_accuracies are 100%
            if converged(val_accuracies):
                print(f"\nAchieved consistent perfect validation accuracy after {epoch} epochs")
                break
    return TrainingHistory(losses, train_accuracies, val_accuracies)

def load_model_state(model: HookedTransformer, filename: str) -> None:
    assert os.path.isdir("models"), "Make a directory `models` with model state dicts"
    if not filename.startswith("models/"):
        filename = f"models/{filename}"
    with open(filename, "rb") as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)


# ### Train or load model


history = train_model(model)


# ### Testing post-training

print("Validating on validation data:")
val_acc = validate(model, val_data)
print(f"\t{val_acc=:.3%}\n")
if val_acc < 1:
    show_mispreds(model, val_data)

print("\nValidating on test data:")
test_acc = validate(model, test_data)
print(f"\t{test_acc=:.3%}\n")
if test_acc < 1:
    show_mispreds(model, test_data)


# ### Saving trained model

def save_model_state_dict(
    model: HookedTransformer, 
    filename: str | None = None
) -> None:
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not filename:
        timestamp = dt.now().isoformat("T", "minutes").replace(":", "-")
        filename = f"model_state_dict_{timestamp}.pkl"
    with open(f"models/{filename}", "wb") as f:
        pickle.dump(model.state_dict(), f)

save_model_state_dict(model)