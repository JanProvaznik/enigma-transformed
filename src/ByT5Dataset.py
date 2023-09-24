from transformers import ByT5Tokenizer
import ciphers
from preprocessing import prepend_hello



class ByT5Dataset:
    def __init__(self, ciphertext_preprocess_fn, plaintext_preprocess_fn, dataset, max_length) -> None:
        self.tokenizer: ByT5Tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        self.input = [ciphertext_preprocess_fn(x) for x in dataset]
        self.output = [plaintext_preprocess_fn(x) for x in dataset]
        self.max_length = max_length
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_text = self.input[idx]
        output_text = self.output[idx]
        encoding = self.tokenizer(
            input_text, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        encoding_labels = self.tokenizer(
            output_text, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "labels": encoding_labels["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "input_text": input_text,
            "output_text": output_text,
        }

class ByT5DatasetOnlyPreprocessCiphertext(ByT5Dataset):
    def __init__(self, ciphertext_preprocess_fn, dataset, max_length) -> None:
        super().__init__(ciphertext_preprocess_fn, ciphers.nothing, dataset, max_length)

class ByT5CopyDataset(ByT5DatasetOnlyPreprocessCiphertext):
    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.nothing, data, max_length)


class ByT5CaesarDataset(ByT5DatasetOnlyPreprocessCiphertext):
    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.caesar, data, max_length)

class ByT5MultiCaesarDataset(ByT5DatasetOnlyPreprocessCiphertext):
    def __init__(self, data, max_length) -> None:
        multi_caesar = ciphers.make_multi_caesar()
        super().__init__(multi_caesar, data, max_length)

class ByT5CaesarRandomDataset(ByT5DatasetOnlyPreprocessCiphertext):
    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.caesar_random, data, max_length)

class ByT5CaesarRandomWithHelloHintDataset(ByT5Dataset):
    def __init__(self, data, max_length) -> None:
        super().__init__(ciphers.caesar_random_hello_hint, prepend_hello, data, max_length)