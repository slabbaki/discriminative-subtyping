import numpy as np
import tensorflow as tf

from . import fashion_mnist
from . import imdb
from . import mnist
from . import physionet
from . import satellite
from . import support2
from . import tcga
from . import tcga_v2


def load(
    name,
    epochs,
    context_kwargs,
    feature_kwargs=None,
    max_train_size=None,
    permute=True,
    seed=42
):
    if name == "mnist":
        load_data = mnist.load_data
        load_interp_features = mnist.load_interp_features
    elif name == "fashion_mnist":
        load_data = fashion_mnist.load_data
        load_interp_features = fashion_mnist.load_interp_features
    elif name == "imdb":
        load_data = imdb.load_data
        load_interp_features = imdb.load_interp_features
    elif name == "satellite":
        load_data = satellite.load_data
        load_interp_features = satellite.load_interp_features
    elif name == "support2":
        load_data = support2.load_data
        load_interp_features = support2.load_interp_features
    elif name == "physionet":
        load_data = physionet.load_data
        load_interp_features = physionet.load_interp_features
    elif name == "tcga":
        load_data = tcga.load_data
    elif name == "tcga_v2":
        load_data = tcga_v2.load_data
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if not name.startswith("tcga"):
        train, valid, test = load_data(seed=seed, permute=permute, **context_kwargs)
        data = {
            "train": [(train[0],), train[1]],
            "valid": [(valid[0],), valid[1]],
            "test": [(test[0],), test[1]],
        }
    elif name == "tcga":
        img_train, interp_train, lbl_train, img_valid, interp_valid, lbl_valid, img_test, interp_test, lbl_test = load_data(**context_kwargs)
        data = {
            "train": tf.data.Dataset.zip((tf.data.Dataset.zip((img_train, interp_train)), lbl_train)),
            "valid": tf.data.Dataset.zip((tf.data.Dataset.zip((img_valid, interp_valid)), lbl_valid)),
            "test": tf.data.Dataset.zip((tf.data.Dataset.zip((img_test, interp_test)), lbl_test)),
        }
    elif name == "tcga_v2":
        data = load_data(seed=seed, epochs=epochs, **context_kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {name}.")

    if max_train_size is not None:
        data["train"][0] = tuple(x[:max_train_size] for x in data["train"][0])
        data["train"][1] = data["train"][1][:max_train_size]

    return data

def load_flipped(
    name,
    epochs,
    context_kwargs,
    feature_kwargs=None,
    max_train_size=None,
    permute=True,
    seed=42
):
    if name == "mnist":
        load_data = mnist.load_data
        load_interp_features = mnist.load_interp_features
    elif name == "fashion_mnist":
        load_data = fashion_mnist.load_data
        load_interp_features = fashion_mnist.load_interp_features
    elif name == "imdb":
        load_data = imdb.load_data
        load_interp_features = imdb.load_interp_features
    elif name == "satellite":
        load_data = satellite.load_data
        load_interp_features = satellite.load_interp_features
    elif name == "support2":
        load_data = support2.load_data
        load_interp_features = support2.load_interp_features
    elif name == "physionet":
        load_data = physionet.load_data
        load_interp_features = physionet.load_interp_features
    elif name == "tcga":
        load_data = tcga.load_data
    elif name == "tcga_v2":
        load_data = tcga_v2.load_data_flipped
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if not name.startswith("tcga"):
        train, valid, test = load_data(seed=seed, permute=permute, **context_kwargs)
        data = {
            "train": [(train[0],), train[1]],
            "valid": [(valid[0],), valid[1]],
            "test": [(test[0],), test[1]],
        }
    elif name == "tcga":
        img_train, interp_train, lbl_train, img_valid, interp_valid, lbl_valid, img_test, interp_test, lbl_test = load_data(**context_kwargs)
        data = {
            "train": tf.data.Dataset.zip((tf.data.Dataset.zip((img_train, interp_train)), lbl_train)),
            "valid": tf.data.Dataset.zip((tf.data.Dataset.zip((img_valid, interp_valid)), lbl_valid)),
            "test": tf.data.Dataset.zip((tf.data.Dataset.zip((img_test, interp_test)), lbl_test)),
        }
    elif name == "tcga_v2":
        data = load_data(seed=seed, epochs=epochs,  **context_kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {name}.")

    if max_train_size is not None:
        data["train"][0] = tuple(x[:max_train_size] for x in data["train"][0])
        data["train"][1] = data["train"][1][:max_train_size]

    return data


def merge(data):
    """Merges training, validation, and test data."""
    set_names = ["train", "valid", "test"]
    inputs = tuple(
        np.concatenate([
            data[set_name][0][i] for set_name in set_names
        ], axis=0)
        for i in range(len(data["train"][0]))
    )
    labels = np.concatenate([
        data[set_name][1] for set_name in set_names
    ], axis=0)
    return inputs, labels


def split(data, train_ids, test_ids, valid_ids=None):
    """Split data into train, test (and validation) subsets."""
    datasets = {
        "train": (
            tuple(map(lambda x: x[train_ids], data[0])), data[1][train_ids]
        ),
        "test": (
            tuple(map(lambda x: x[test_ids], data[0])), data[1][test_ids]
        ),
    }
    if valid_ids is not None:
        datasets["valid"] = (
            tuple(map(lambda x: x[valid_ids], data[0])), data[1][valid_ids]
        )
    else:
        datasets["valid"] = None
    return datasets
