"""Data utils."""

from concurrent import futures
from functools import partial

import numpy as np
import scipy as sp


def get_tokenizer(name="bert-base-uncased", max_workers=16):

    import transformers

    def tokenize(inputs, max_length=None):
        # Build tokenizer.
        if name.startswith("bert"):
            tokenizer = transformers.BertTokenizer.from_pretrained(name)
        else:
            raise ValueError(f"Unknown tokenizer name: {name}.")
        encode = partial(
            tokenizer.encode,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
        )
        # Tokenize inputs.
        chunksize = len(inputs) // max_workers
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            tokenized = list(executor.map(encode, inputs, chunksize=chunksize))
        return np.asarray(tokenized, dtype=np.int32)

    return tokenize


def get_zca_whitening_mat(X, eps=1e-6):
    flat_X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    Sigma = np.dot(flat_X.T, flat_X) / flat_X.shape[0]
    U, S, _ = sp.linalg.svd(Sigma)
    M = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    return M


def zca_whiten(X, W):
    shape = X.shape
    flat_X = np.reshape(X, (shape[0], np.prod(shape[1:])))
    white_X = np.dot(flat_X, W)
    return np.reshape(white_X, shape)
