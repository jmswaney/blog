from typing import Optional
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="white", context="poster", rc={ "figure.figsize": (14, 10), "lines.markersize": 1 })

def make_scatter(x: np.ndarray, y: Optional[np.ndarray]):
    if y is None:
        plt.scatter(x[idx, 0], x[idx, 1])
    else:
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            plt.scatter(x[idx, 0], x[idx, 1], label=label)
            plt.legend(bbox_to_anchor=(0, 1), loc="upper left", markerscale=6)

def plot_umap(x: np.ndarray, y: Optional[np.ndarray], filename: str):
    make_scatter(x, y)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(filename)
    plt.savefig(filename)

def plot_pca(x: np.ndarray, y: Optional[np.ndarray], filename: str):
    make_scatter(x, y)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(filename)
    plt.savefig(filename)
