import random
import logging
import re

logging.basicConfig(level=logging.INFO)


def generate_random_dataset(
    rows: int, min_length: int, max_length: int, space_frequency: float, seed: int = 42
) -> list[str]:
    """Samples a random dataset from the alphabet [a-z] (uniformly) and space which can be given custom probability.

    Input:
        rows: number of rows in the dataset
        min_length: minimum character length of a row
        max_length: maximum character length of a row
        space_frequency: how big proportion of the text should be spaces?
        seed: random seed

    Output:
        dataset: dataset as a python list of strings
    """
    random.seed(seed)
    letter_weight_without_space = (1 - space_frequency) / 26
    letters = " abcdefghijklmnopqrstuvwxyz"
    letter_weights = [space_frequency] + [
        letter_weight_without_space for _ in range(26)
    ]
    dataset = []
    for _ in range(rows):
        length = random.randint(min_length, max_length)
        dataset.append(
            "".join(random.choices(letters, weights=letter_weights, k=length))
        )
    logging.info(
        "Generated dataset with %d rows, min_length=%d, max_length=%d, space_frequency=%f, seed=%d",
        rows,
        min_length,
        max_length,
        space_frequency,
        seed,
    )
    return dataset


def load_dataset(
    rows: int, min_length: int, max_length: int, file_path: str, seed: int = 42, exclude_length: int = None
) -> list[str]:
    """Samples a dataset from a file and truncates each rows to a random length generated for it from a range.

    Input:
        rows: number of desired rows in the dataset
        min_length: minimum character length of a row truncation
        max_length: maximum character length of a row truncation
        file_path: path to the file
        seed: random seed for sampling and generating truncations
        exclude_length: exclude rows with less than this length

    Output:
        dataset: dataset as a python list of strings
    """
    random.seed(seed)
    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    if exclude_length is not None:
        all_lines = [line for line in all_lines if len(line) >= exclude_length]
    selected_rows = random.sample(all_lines, rows)
    dataset = [line.strip() for line in selected_rows]
    dataset = [line[: random.randint(min_length, max_length)] for line in dataset]
    logging.info(
        "Loaded dataset from %s, containing %d rows, min_length=%d, max_length=%d, seed=%d",
        file_path,
        len(dataset),
        min_length,
        max_length,
        seed,
    )
    return dataset


def only_letters(text, preserve_spaces=False):
    """Removes all non-letter characters from the text, optionally preserving spaces."""
    pattern = r"[^A-Za-z \n]+" if preserve_spaces else r"[^A-Za-z\n]+"
    return re.sub(pattern, "", text).lower()


def replace_digits(text: str) -> str:
    """Replaces digits with their textual representation."""
    digit_map = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "0": "zero",
    }
    return "".join(digit_map.get(char, char) for char in text)


def preprocess_text(
    text: str, preserve_spaces: bool = True, convert_digits: bool = False
) -> str:
    """Preprocesses the text by removing special characters, optionally preserving spaces and converting digits to their textual representation."""
    if convert_digits:
        text = replace_digits(text)
    text = only_letters(text, preserve_spaces)
    return text


def preprocess_file(
    path: str, preserve_spaces: bool = True, convert_digits: bool = False
) -> None:
    """Preprocesses the file by removing special characters, optionally preserving spaces and converting digits to their textual representation."""
    with open(path, "r+", encoding="utf-8") as f:
        text = f.readlines()
        text = [preprocess_text(line, preserve_spaces, convert_digits) for line in text]
        f.seek(0)
        f.writelines(text)
        f.truncate()
    logging.info(
        "Preprocessed file %s, removed special characters, convert_digits=%s, preserve_spaces=%s",
        path,
        convert_digits,
        preserve_spaces,
    )


def prepend_hello(text: str) -> str:
    """Prepends 'hello' to the text."""
    return "hello " + text
