"""Loader and preprocessors for tcga data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
from os import listdir
from os.path import isfile, join

from .tcga_dataset import Dataset_withAug_withGen

logger = logging.getLogger(__name__)

def load_data(train_dir=None,
              valid_dir=None,
              test_dir=None,
              epochs=None,
              batch_size = 128,
              num_preprocess_threads = 4,
              ):

    path_train = [train_dir + '/' + f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    path_valid = [valid_dir + '/' + f for f in listdir(valid_dir) if isfile(join(valid_dir, f))]
    path_test = [test_dir + '/' + f for f in listdir(test_dir) if isfile(join(test_dir, f))]

    img_train, interp_train, lbl_train = Dataset_withAug_withGen.tfrecord_train_input_fn(train=True, batch_size=batch_size, path=path_train,
                                                          num_preprocess_threads=num_preprocess_threads,epochs=epochs)
    img_valid, interp_valid, lbl_valid = Dataset_withAug_withGen.tfrecord_train_input_fn(train=True, batch_size=batch_size, path=path_valid,
                                                          num_preprocess_threads=num_preprocess_threads, epochs=epochs)
    img_test, interp_test, lbl_test = Dataset_withAug_withGen.tfrecord_train_input_fn(train=True, batch_size=batch_size, path=path_test,
                                                          num_preprocess_threads=num_preprocess_threads, epochs=epochs)
    return img_train, interp_train, lbl_train, img_valid, interp_valid, lbl_valid, img_test, interp_test, lbl_test  