from multiprocessing.connection import wait
from random import random
from math import floor
import numpy as np

def sample_patches(x: np.ndarray, n: int, w: int):
    start = lambda: (floor(random()*(x.shape[0] - w)), floor(random()*(x.shape[1] - w)))
    return np.asarray([x[i:i+w, j:j+w] for (i, j) in [start() for _ in range(n)]])
