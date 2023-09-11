import os
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