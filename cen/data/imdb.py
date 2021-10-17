"""Loader and preprocessors for IMDB data."""

import logging
import os

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import sequence

from . import utils

logger = logging.getLogger(__name__)

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 20000, 5000, 25000
NB_CLASSES = 2


def load_imdb(path, load_text=False, vocab_size=None, skip_top=0,
              start_char=1, oov_char=2, index_from=3):
    """Loads the IMDB dataset.
    Taken from `keras.datasets.imdb` and edited as necessary.
    """
    f = np.load(path, allow_pickle=True)
    x_train = f["x_train"]
    labels_train = f["y_train"]
    x_test = f["x_test"]
    labels_test = f["y_test"]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if load_text:
        vocab_path = os.path.join(os.path.dirname(path), "vocab.txt")
        with open(vocab_path) as fp:
            vocab = [line.strip() for line in fp]
        xs = [" ".join([vocab[i - 1] for i in x]) for x in xs]
    else:
        if start_char is not None:
            xs = [[start_char] + [w + index_from for w in x] for x in xs]
        elif index_from:
            xs = [[w + index_from for w in x] for x in xs]

        if not vocab_size:
            vocab_size = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve `index_from` (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [
                np.asarray([
                    oov_char if (w >= vocab_size or w < skip_top)
                    else w for w in x
                ], dtype=np.int32)
                for x in xs
            ]
        else:
            new_xs = []
            for x in xs:
                nx = []
                for w in x:
                    if w >= vocab_size or w < skip_top:
                        nx.append(w)
                new_xs.append(np.asarray(nx, dtype=np.int32))
            xs = new_xs

    # Convert labels to one-hot.
    labels = np_utils.to_categorical(labels, NB_CLASSES)

    # Split back into train and test.
    x_train = np.array(xs[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])
    x_test = np.array(xs[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (x_train, y_train), (x_test, y_test)


def load_data(datapath=None,
              set_name=None,
              tokenizer=None,
              vocab_size=20000,
              maxlen=512,
              skip_top=0,
              permute=True,
              max_workers=16,
              seed=42):
    """Load the sequential representation of the data.

    Args:
        datapath: str or None (default: None)
        set_name: str or None (default: None)
        tokenizer: str (default: None)
        nb_topics: uint (default: 50)
        vocab_size: uint (default: 20000)
        maxlen: uint (default: 80)
        skip_top: uint (default: 0)
        permute: bool (default: True)
        max_workers: int (default: 16)
        seed: uint (default: 42)

    Returns:
        data: tuples (X, y) of ndarrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/IMDB/imdb.npz"
    datapath = os.path.expandvars(datapath)

    if tokenizer is not None:
        (X_train, y_train), (X_test, y_test) = load_imdb(
            datapath, load_text=True
        )
        tokenize = utils.get_tokenizer(tokenizer, max_workers=max_workers)
        X_train = tokenize(X_train, max_length=maxlen)
        X_test = tokenize(X_test, max_length=maxlen)
    else:
        (X_train, y_train), (X_test, y_test) = load_imdb(
            datapath,
            load_text=False,
            vocab_size=vocab_size,
            skip_top=skip_top
        )
        # Zero-padding.
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    # Permute the train data if necessary.
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X_train))
        X_train = X_train[order]
        y_train = y_train[order]

    # Split test into valid and test if no specific set is requested.
    if set_name is None:
        X_valid = X_train[-VALID_SIZE:]
        y_valid = y_train[-VALID_SIZE:]
        X_train = X_train[:-VALID_SIZE]
        y_train = y_train[:-VALID_SIZE]

        # Sanity check
        assert len(X_train) == TRAIN_SIZE
        assert len(X_valid) == VALID_SIZE
        assert len(X_test) == TEST_SIZE

        data = (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

        logger.debug(f"X shape: {X_train.shape[1:]}")
        logger.debug(f"Y shape: {y_train.shape[1:]}")
        logger.debug(f"{len(X_train)} train sequences")
        logger.debug(f"{len(X_valid)} validation sequences")
        logger.debug(f"{len(X_test)} test sequences")
    else:
        assert set_name in {"train", "test"}
        data = (X_train, y_train) if set_name == "train" else (X_test, y_test)
        logger.debug(f"{len(data[0])} {set_name} sequences")

    return data


def load_interp_features(datapath=None,
                         set_name=None,
                         feature_type="topics",
                         nb_topics=50,
                         bow_vocab_size=2000,
                         topic_vocab_size=20000,
                         extended=False,
                         remove_const_features=True,
                         standardize=True,
                         permute=True,
                         signal_to_noise=None,
                         seed=42):
    """Load the interpretable representation of the data.

    Args:
        datapath: str or None (default: None)
        set_name: str or None (default: None)
        feature_type: str (default: "topics")
            Interpretable representation (one of {"topics", "BoW", "both"}).
        nb_topics: int (default: 50)
        bow_vocab_size: int (default: 20000)
        topic_vocab_size: int (default: 20000)
        extended: bool (default: False)
            Whether to load the representation produced by a topic model
            trained on the extended dataset.
        permute: bool (default: True)
        signal_to_noise: float or None (default: None)
            If not None, adds white noise to each feature with a specified SNR.
        seed: uint (default: 42)

    Returns:
        data: tuple (Z_train, Z_valid, Z_test) of ndarrays.
    """
    if datapath is None:
        datapath = "$DATA_PATH/IMDB/"
    datapath = os.path.expandvars(datapath)

    if feature_type in {"BoW", "both"}:
        with open(os.path.join(datapath, "bow", "imdb.vocab")) as fp:
            vocab = [word.strip() for word in fp.readlines()]
        bow_train, _ = load_svmlight_file(
            os.path.join(datapath, "bow", "bow_train.feat"),
            n_features=len(vocab))
        bow_test, _ = load_svmlight_file(
            os.path.join(datapath, "bow", "bow_test.feat"),
            n_features=len(vocab))

        # Pre-process the data (reduce the vocabulary size).
        word_ids = np.nonzero(
            [w not in ENGLISH_STOP_WORDS for w in vocab]
        )[0][:bow_vocab_size]
        bow_train = bow_train[:, word_ids].toarray()
        bow_test = bow_test[:, word_ids].toarray()

    if feature_type in {"topics", "both"}:
        prefix = "utrain" if extended else "train"
        train_path = os.path.join(datapath, "topics",
            "%s_%d_%d.npy" % (prefix, nb_topics, topic_vocab_size))
        topics_train = np.load(train_path)

        prefix = "utest" if extended else "test"
        test_path = os.path.join(datapath, "topics",
            "%s_%d_%d.npy" % (prefix, nb_topics, topic_vocab_size))
        topics_test = np.load(test_path)

    if feature_type == "both":
        Z_train = np.hstack([topics_train, bow_train])
        Z_test = np.hstack([topics_test, bow_test])
    elif feature_type == "BoW":
        Z_train, Z_test = bow_train, bow_test
    elif feature_type == "topics":
        Z_train, Z_test = topics_train, topics_test

    if remove_const_features:
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train = Z_train[:, nonconst]
        Z_test = Z_test[:, nonconst]

    if standardize:
        Z_mean = Z_train.mean(axis=0)
        Z_std = Z_train.std(axis=0)
        Z_train -= Z_mean
        Z_train /= Z_std
        Z_test -= Z_mean
        Z_test /= Z_std

    # Cast dtype to float32.
    Z_train = Z_train.astype(np.float32)
    Z_test = Z_test.astype(np.float32)

    # Remove unsupervised data, if was used for learning topics.
    if extended:
        Z_train = Z_train[:TRAIN_SIZE+VALID_SIZE]

    # Permute the train data, if necessary.
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z_train))
        Z_train = Z_train[order]

    if signal_to_noise is not None and signal_to_noise > 0.:
        rng = np.random.RandomState(seed)
        N_train = rng.normal(scale=1./signal_to_noise, size=Z_train.shape)
        N_test = rng.normal(scale=1./signal_to_noise, size=Z_test.shape)
        Z_train += N_train
        Z_test += N_test

    if set_name is None:
        Z_valid = Z_train[-VALID_SIZE:]
        Z_train = Z_train[:-VALID_SIZE]

        # Sanity check
        assert len(Z_train) == TRAIN_SIZE
        assert len(Z_valid) == VALID_SIZE
        assert len(Z_test) == TEST_SIZE

        data = Z_train, Z_valid, Z_test
    else:
        data = Z_train if set_name == "train" else Z_test

    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return data
