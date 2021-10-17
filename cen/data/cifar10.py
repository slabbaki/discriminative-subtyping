"""Loader and preprocessors for CIFAR10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.keras.datasets import cifar
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import backend as K

from ..utils.data import get_zca_whitening_mat, zca_whiten

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 40000, 10000, 10000
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
NB_CLASSES = 10


def load_data(datapath=None,
              whitened=False,
              standardize=False,
              permute=True,
              seed=42,
              verbose=1):
    """Load CIFAR10 data.

    Args:
        datapath : str or None (default: None)
        permute : bool (default: False)
        seed : uint (default: 42)
        verbose : uint (default: 1)

    Returns:
        data: tuples (X, y) of ndarrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/CIFAR10/"
    datapath = os.path.expandvars(datapath)

    if not whitened:
        datapath = os.path.join(datapath, 'cifar-10-batches-py')
        X_train = np.zeros(
            (TRAIN_SIZE + VALID_SIZE, IMG_CHANNELS, IMG_ROWS, IMG_COLS),
            dtype="uint8")
        y_train = np.zeros((TRAIN_SIZE + VALID_SIZE,), dtype="uint8")

        for i in range(1, 6):
            fpath = os.path.join(datapath, 'data_batch_' + str(i))
            data, labels = cifar.load_batch(fpath)
            X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = os.path.join(datapath, 'test_batch')
        X_test, y_test = cifar.load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
    else:
        data = np.load(os.path.join(datapath, 'cifar10.whitened.npz'))
        X_train, y_train = data['X_train'], data['Y_train'] - 1
        X_test, y_test = data['X_test'], data['Y_test'] - 1

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    if not whitened:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train -= np.asarray([[[[103.939, 116.779, 123.68]]]])
        X_test -= np.asarray([[[[103.939, 116.779, 123.68]]]])
    else:
        pass
        # X_train -= np.asarray([[[[125.3, 123.0, 113.9]]]])
        # X_train /= np.asarray([[[[63.0,  62.1,  66.7]]]])
        # X_test -= np.asarray([[[[125.3, 123.0, 113.9]]]])
        # X_test /= np.asarray([[[[63.0,  62.1,  66.7]]]])

    if standardize:
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train -= X_mean
        X_train /= X_std
        X_test -= X_mean
        X_test /= X_std

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X_train))
        X_train = X_train[order]
        y_train = y_train[order]

    # Split train into train and validation
    X_valid = X_train[-VALID_SIZE:]
    X_train = X_train[:-VALID_SIZE]
    y_valid = y_train[-VALID_SIZE:]
    y_train = y_train[:-VALID_SIZE]

    # Sanity checks
    assert X_train.shape[0] == TRAIN_SIZE
    assert X_valid.shape[0] == VALID_SIZE
    assert X_test.shape[0] == TEST_SIZE

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    if verbose:
        print('X shape:', X_train.shape[1:])
        print('Y shape:', Y_train.shape[1:])
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'validation samples')
        print(X_test.shape[0], 'test samples')

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def load_interp_features(datapath=None,
                         feature_type='pixels16x16',
                         remove_const_features=True,
                         standardize=True,
                         whiten=False,
                         permute=True,
                         seed=42,
                         verbose=1):
    """Load an interpretable representation for MNIST.

    Args:
        datapath : str or None (default: None)
        remove_const_features : bool (default: True)
        feature_type : str (default: 'pixels16x16')
            Possible values are:
            {'pixels16x16', 'pixels24x24', 'pixels32x32', 'hog3x3', 'hog4x4'}.
        standardize : bool (default: True)
        whiten : bool (default: False)
        verbose : uint (default: 1)

    Returns:
        data: tuple (Z_train, Z_valid, Z_test) of ndarrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/CIFAR10/cifar10.interp.%s.npz" % feature_type
    datapath = os.path.expandvars(datapath)

    if verbose:
        print("Loading interpretable features...")

    data = np.load(datapath)
    Z_train, Z_test = data['Z_train'], data['Z_test']

    if feature_type.startswith('pixels'):
        Z_train = Z_train.astype('float32')
        Z_test = Z_test.astype('float32')

    Z_train = Z_train.reshape((TRAIN_SIZE + VALID_SIZE, -1))
    Z_test = Z_test.reshape((TEST_SIZE, -1))

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

    if whiten:
        WM = get_zca_whitening_mat(Z_train)
        Z_train = zca_whiten(Z_train, WM)
        Z_test = zca_whiten(Z_test, WM)

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z_train))
        Z_train = Z_train[order]

    # Split train into train and validation
    Z_valid = Z_train[-VALID_SIZE:]
    Z_train = Z_train[:-VALID_SIZE]

    # Sanity checks
    assert Z_train.shape[0] == TRAIN_SIZE
    assert Z_valid.shape[0] == VALID_SIZE
    assert Z_test.shape[0] == TEST_SIZE

    if verbose:
        print('Z shape:', Z_train.shape[1:])
        print(Z_train.shape[0], 'train samples')
        print(Z_valid.shape[0], 'validation samples')
        print(Z_test.shape[0], 'test samples')

    return Z_train, Z_valid, Z_test
