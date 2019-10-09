


import numpy as np
import torch




from pathlib import Path
import requests


def download_data(directory="mnist"):
    DATA_PATH=Path("data")
    PATH= DATA_PATH / directory
    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    return content


