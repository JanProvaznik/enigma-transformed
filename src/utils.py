import os
import difflib
import Levenshtein
import logging

def download_newscrawl(year=2012, language="en"):
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

def batch_size(batch_size, memory, tokens_per_example):
    """
    input: desired batch_size

    output: (batch_size, gradient_accumulation_steps) that will fit in memory

    """
    #TODO implement


def print_diffs(orig, gen):
    d = difflib.Differ()
    diff = d.compare(orig.split(), gen.split())
    # levensthein distance / number of characters
    dist_ratio = Levenshtein.distance(orig, gen) / len(orig)

    logging.info(orig, gen, dist_ratio , '\n'.join(diff))


def levensthein_distance(orig, gen):
    return Levenshtein.distance(orig, gen)


def print_avg_median_mode_error(error_counts: list[int]) -> tuple[float, float, int]:
    """
    Print average, median and mode of error count for each example
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