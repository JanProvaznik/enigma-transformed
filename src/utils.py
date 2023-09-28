import os
import Levenshtein
import logging
import torch.cuda
from math import floor


def download_newscrawl(year=2012, language="en") -> None:
    """Download a newscrawl dataset from https://data.statmt.org/news-crawl/

    Input:
        year: year of the dataset
        language: language of the dataset
    """
    newscrawl_url = f"https://data.statmt.org/news-crawl/{language}/"
    filename = f"news.{year}.{language}.shuffled.deduped.gz"
    url = newscrawl_url + filename
    logging.info(f"Downloading {filename} from {url}")
    os.system(f"wget {url}")
    os.system(f"gunzip {filename}")
    logging.info(
        f"Downloaded and extracted {filename}",
        f"full path: {os.path.abspath(filename)}",
    )


def calculate_batch_size(
    target_batch_size: int, tokens_per_example: int
) -> tuple[int, int]:
    """
    Input:
        target_batch_size: effective batch size we want to achieve
        tokens_per_example: number of tokens in each example

    output: (batch_size, gradient_accumulation_steps) that will fit in memory

    Reference: for 10G memory and 100 tokens per example we would return (16, 16)
    NOTE: this holds only for the ByT5-small model and the default Seq2Seq traning scheme

    """
    memory_GB = torch.cuda.mem_get_info()[0] // (1024**3)

    reference_chars_per_gb = 100 * 16 // 10
    batch_size = min(
        target_batch_size,
        floor(memory_GB * reference_chars_per_gb) // tokens_per_example,
    )
    gradient_accumulation_steps = target_batch_size // batch_size
    return batch_size, gradient_accumulation_steps


def levensthein_distance(orig, gen):
    """Levensthein distance between two strings, no weights"""
    return Levenshtein.distance(orig, gen)


def print_avg_median_mode_error(error_counts: list[int]) -> tuple[float, float, int]:
    """Print average, median and mode of error count for each example

    Input: list of error counts
    Output: tuple of average, median and mode
    """
    avg_error = sum(error_counts) / len(error_counts)
    median_error = sorted(error_counts)[len(error_counts) // 2]
    mode_error = max(set(error_counts), key=error_counts.count)
    print(f"Average errors: {avg_error}")
    print(f"Median errors: {median_error}")
    print(f"Mode errors: {mode_error}")
    return avg_error, median_error, mode_error
