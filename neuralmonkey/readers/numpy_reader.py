from typing import List

import numpy as np

# tests: lint


def numpy_reader(files: List[str]):
    if len(files) == 1:
        return np.load(files[0])
    else:
        np.concatenate([np.load(f) for f in files], axis=0)
