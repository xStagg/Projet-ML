import os
import numpy as np

FILES = [
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
]

def load_images(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28*28).astype(np.float32) / 255.0

def load_labels(path):
    with open(path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

def load_mnist_ubyte():
    paths = [os.path.join(root, f) for f in FILES]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier manquant: {p}")
    x_train = load_images(paths[0])
    y_train = load_labels(paths[1])
    x_test  = load_images(paths[2])
    y_test  = load_labels(paths[3])
    return (x_train, y_train), (x_test, y_test)

root = os.getcwd() + "\data"  # <-- remplace par ton vrai dossier
(x_train, y_train), (x_test, y_test) = load_mnist_ubyte()

print(x_train.shape, y_train.shape)
print(x_test.shape,  y_test.shape)