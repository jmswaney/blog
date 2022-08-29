from typing import Optional
import os
import numpy as np
from skimage.io import imread
from tensorflow.keras.datasets import mnist, cifar10

MNIST_NAME = "mnist"
MNIST_CACHE = "mnist.npz"
CIFAR10_NAME = "cifar10"
NUMPY_EXT = ".npy"

def load(name: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if name == MNIST_NAME:
        return mnist.load_data(MNIST_CACHE)[0]
    if name == CIFAR10_NAME:
        return cifar10.load_data()[0]
    ext = os.path.splitext(name)[-1]
    if ext == NUMPY_EXT: return np.load(name), None
    return np.asarray([imread(os.path.join(name, x)) for x in os.listdir(name)]), None
    
def save(filename: str, x: np.ndarray, compress: bool) -> None:
    np.savez_compressed(filename, x) if compress else np.save(filename, x)
