import random
import logging
import re


def generate_dataset(
    rows: int, min_length: int, max_length: int, space_frequency: float, seed: int = 42
) -> list[str]:
    """
    params:
        rows: number of rows in the dataset
        min_length: minimum character length of a row
        max_length: maximum character length of a row
        space_frequency: how big proportion of the text should be spaces?
        seed: random seed

    returns:
        dataset: dataset as a python list of strings
    """
    random.seed(seed)
    letter_weight_without_space = (1 - space_frequency) / 26
    letters = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
        f"Generated dataset with {rows} rows, min_length={min_length}, max_length={max_length}, space_frequency={space_frequency}, seed={seed}"
    )
    return dataset


def only_letters(text, preserve_spaces=False):
    pattern = r"[^A-Za-z \n]+" if preserve_spaces else r"[^A-Za-z\n]+"
    return re.sub(pattern, "", text).upper()


def replace_digits(text: str) -> str:
    digit_map = {
        "1": "ONE",
        "2": "TWO",
        "3": "THREE",
        "4": "FOUR",
        "5": "FIVE",
        "6": "SIX",
        "7": "SEVEN",
        "8": "EIGHT",
        "9": "NINE",
        "0": "ZERO",
    }
    return "".join(digit_map.get(char, char) for char in text)


def preprocess_text(text: str, preserve_spaces: bool, convert_digits: bool) -> str:
    if convert_digits:
        text = replace_digits(text)
    text = only_letters(text, preserve_spaces)
    return text


def preprocess_file(path: str, preserve_spaces: bool, convert_digits: bool) -> None:
    with open(path, "r+") as f:
        text = f.read()
        text = preprocess_text(text, preserve_spaces, convert_digits)
        f.seek(0)
        f.write(text)
        f.truncate()
    logging.info(
        f"Preprocessed file {path}, removed special characters, {convert_digits=}, {preserve_spaces=}"
    )

def test_preprocess_text():
    assert only_letters("abc! 123", True) == "ABC "
    assert only_letters("abc! 123", False) == "ABC"
    # Test replace_digits
    assert replace_digits("123") == "ONETWOTHREE"
    # Test preprocess
    assert preprocess_text("abc! 123", True, True) == "ABC ONETWOTHREE"
    assert preprocess_text("abc! 123", False, False) == "ABC"

    logging.info("preprocess tests passed.")