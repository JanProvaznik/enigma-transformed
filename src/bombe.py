import torch
import transformers
# class BombeTokenizer:
# tokenize only to english capital letters, space and EOS


class Bombe100M(torch.nn.Module):

    """
    Transformer that has only letter embeddings in order to be efficient on the Engima decryption task.

    The starting point is the google/byT5 model from HuggingFace.
    We
    """
    