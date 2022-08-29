import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.decomposition import PCA
from umap import UMAP

def pca(x: np.ndarray, n: int) -> np.ndarray:
    return PCA(n_components=n).fit_transform(x)

def gpuGrowthPatch():
    # https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def batch(iterable, n=1):
    for i in range(0, len(iterable), n):
        yield iterable[i:min(i + n, len(iterable))]

def resnet(x: np.ndarray, batch_size = None, layers = None) -> np.ndarray:
    gpuGrowthPatch()
    full = tf.keras.applications.ResNet50V2(include_top=False)
    model = full if layers is None else tf.keras.models.Model(full.input, full.layers[layers].output)
    print(model.summary())
    if batch_size is None:
        maps = model.predict(preprocess_input(x))
    else:
        maps = np.concatenate([model.predict(preprocess_input(xb)) for xb in batch(x, batch_size)])
    if np.prod(maps.shape) == maps.shape[-1] * len(x):
        return np.squeeze(maps)
    else:
        return maps.mean(axis=1).mean(axis=1)

def embed(x: np.ndarray, model: str, n: int, batch: int | None, layers: int | None):
    match model:
        case "pca":
            return pca(x.reshape(len(x), -1), n)
        case "resnet":
            if x.ndim == 3:
                return resnet(np.repeat(x[..., np.newaxis], 3, axis=-1), batch, layers)
            return resnet(x, batch, layers)
        case _:
            raise Exception("Invalid model provided")

def umap(x, seed = None):
    if seed: np.random.seed(seed)
    return UMAP().fit_transform(x)
