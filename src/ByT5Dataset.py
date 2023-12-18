import random
import functools
from transformers import ByT5Tokenizer
from torch.utils.data import Dataset
import src.ciphers as ciphers
from src.preprocessing import prepend_hello


class ByT5Dataset(Dataset):
    """
    Base dataset class for ByT5.

    Takes in ciphertext and plaintext preprocessing functions,
    a dataset, and a max length. Applies the preprocessing functions
    to the dataset and tokenizes the results.
    """

    def __init__(
        self, ciphertext_preprocess_fn, plaintext_preprocess_fn, dataset, max_length
    ) -> None:
        self.tokenizer: ByT5Tokenizer = ByT5Tokenizer.from_pretrained(
            "google/byt5-small"
        )
        # apply preprocessing functions to dataset
        self.input = [ciphertext_preprocess_fn(x) for x in dataset]
        self.output = [plaintext_preprocess_fn(x) for x in dataset]

        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx) -> dict:
        input_text = self.input[idx]
        output_text = self.output[idx]
        encoding = self.tokenizer(
            input_text,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoding_labels = self.tokenizer(
            output_text,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "labels": encoding_labels["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "input_text": input_text,
            "output_text": output_text,
        }


class ByT5DatasetOnlyPreprocessCiphertext(ByT5Dataset):
    """
    Simplified version of ByT5Dataset that only preprocesses the ciphertext.
    """

    def __init__(self, ciphertext_preprocess_fn, dataset, max_length) -> None:
        super().__init__(ciphertext_preprocess_fn, ciphers.nothing, dataset, max_length)


class ByT5CopyDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Identity cipher - copies plaintext to ciphertext.
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.nothing, data, max_length)


class ByT5CaesarDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using Caesar cipher preprocessing.
    """

    def __init__(self, data, max_length, shift=3) -> None:
        super().__init__(
            functools.partial(ciphers.caesar, shift=shift), data, max_length
        )


class ByT5MultiCaesarDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using Caesar cipher with multiple possible shifts preprocessing.
    """

    def __init__(self, data, max_length) -> None:
        multi_caesar = ciphers.make_multi_caesar()
        super().__init__(multi_caesar, data, max_length)


class ByT5RiggedCaesarDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using Caesar cipher that uses all but one setting.
    """

    def __init__(self, data, max_length, excluded_shift) -> None:
        shifts = list(range(26))
        # random shuffle
        random.shuffle(shifts)

        shifts.remove(excluded_shift)

        multi_caesar = ciphers.make_multi_caesar(shifts=shifts)
        super().__init__(multi_caesar, data, max_length)


class ByT5CaesarRandomDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using random Caesar cipher preprocessing.
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.caesar_random, data, max_length)


class ByT5CaesarRandomWithHelloHintDataset(ByT5Dataset):
    """
    Dataset prepending 'hello' to each example and using random Caesar cipher preprocessing
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(
            ciphers.caesar_random_hello_hint, prepend_hello, data, max_length
        )


class ByT5Vignere2Dataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using 2-letter Vignere cipher.
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.vignere2, data, max_length)


class ByT5Vignere3Dataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using 3-letter Vignere cipher.
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.vignere3, data, max_length)


class ByT5Vignere2RandomDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using random 2-letter Vignere cipher.
    """

    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.random_vignere2, data, max_length)


class ByT5MultiVignereDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using multiple Vignere ciphers.
    """

    def __init__(self, data, max_length) -> None:
        multivignere = ciphers.make_multi_vignere()
        super().__init__(multivignere, data, max_length)


class ByT5LongVignereDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using Vignere cipher with a long/specified key.
    """

    def __init__(self, data, max_length, key="helloworld") -> None:
        # make a closure that returns a vignere cipher when key="helloworld"
        long_vignere = functools.partial(ciphers.vignere, key=key)
        super().__init__(long_vignere, data, max_length)


class ByT5ConstEnigmaDataset(ByT5DatasetOnlyPreprocessCiphertext):
    """
    Dataset using Enigma cipher with a constant key.
    """

    def __init__(self, data, max_length) -> None:
        const_enigma = ciphers.make_const_enigma()
        super().__init__(const_enigma, data, max_length)