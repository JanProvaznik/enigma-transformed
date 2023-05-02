
import requests
import zipfile
def download_dataset(args):
    # download from link and unzip to dataset_path

    url = args.download_dataset
    r = requests.get(url, allow_redirects=True)
    open('data.zip', 'wb').write(r.content)