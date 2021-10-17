"""Dataset constructor for TCGA."""

import os
import glob
import logging

import numpy as np
import tensorflow as tf

from functools import partial

logger = logging.getLogger(__name__)


def deserialize(example_serialized, genes=None):
    schema = {
        "image/encoded": tf.io.FixedLenSequenceFeature(
            shape=[], dtype=tf.string, allow_missing=True,
        ),
        "class/label": tf.io.FixedLenFeature(
            shape=[], dtype=tf.int64,
        ),
    }
    if genes is not None:
        for g in genes:
            schema[g] = tf.io.FixedLenFeature(
                shape=[], dtype=tf.float32, default_value=-1.
            )
    return tf.io.parse_single_example(example_serialized, schema)


def remove_no_genes(features, genes=None):
    """Filters out samples for which genomic data is missing."""
    if genes is None:
        # Don't filter anything if we are not using genomic data.
        return True
    else:
        # Samples must have non-negative gene expression values.
        return tf.reduce_all([features[g] >= 0. for g in genes])


def distort_color(image, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.compat.v1.keras.backend.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = np.random.randint(2)

    if color_ordering == 0:
      image = tf.compat.v1.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.compat.v1.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.compat.v1.image.random_hue(image, max_delta=0.2)
      image = tf.compat.v1.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.compat.v1.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.compat.v1.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.compat.v1.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.compat.v1.image.random_hue(image, max_delta=0.2)

    image = tf.compat.v1.clip_by_value(image, 0.0, 1.0)
  return image


def preprocess_image_train(image, img_height=299, img_width=299):
    """Preprocesses images for training."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [img_height, img_width])
    image.set_shape([img_height, img_width, 3])

    image = tf.image.random_flip_left_right(image)
    image = distort_color(image)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image_eval(image, img_height=299, img_width=299):
    """Preprocesses images for evaluation."""
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [img_height, img_width])
    image.set_shape([img_height, img_width, 3])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def images_decode_and_preprocess(inputs, label, train=False):
    """Decodes images from JPEG and applies preprocessing."""
    inputs["C"] = tf.image.decode_jpeg(inputs["C"], channels=3)
    if train:
        inputs["C"] = preprocess_image_train(inputs["C"])
    else:
        inputs["C"] = preprocess_image_eval(inputs["C"])
    return inputs, label


def genes_normalize(
    inputs, label, log_expr_means=None, log_expr_stds=None
):
    if "X" in inputs:
        inputs["X"] = tf.math.log(inputs["X"] + 1e-3) - log_expr_means
        inputs["X"] = inputs["X"] / log_expr_stds
    return inputs, label


def parse_features(features, num_classes, genes=None):
    """Decode features into (image, genes, label) tuples."""
    # Parse image patches.
    patches_jpeg = features["image/encoded"]
    inputs = {"C": patches_jpeg}

    # Parse labels.
    labels = tf.repeat(
        [tf.cast(features["class/label"] - 1, dtype=tf.int32)],
        repeats=[tf.shape(patches_jpeg)[0]],
        axis=0
    )
    labels = tf.one_hot(indices=labels, depth=num_classes,)

    # Parse genes.
    if genes is not None:
        genes = [features[g] for g in genes]
        genes = tf.convert_to_tensor(genes, dtype=tf.float32)
        genes = tf.repeat(
            [genes], repeats=[tf.shape(patches_jpeg)[0]], axis=0
        )
        inputs["X"] = genes

    return inputs, labels


def load_data(
    train_dir, valid_dir, test_dir,
    genes_filepath=None,
    log_expr_path=None,
    epochs=None,
    min_log_expr_std=1e-2,
    num_classes=3,
    batch_size=32,
    num_parallel_calls=16,
    shuffle_buffer_size=4000,
    seed=42,
):
    train_tfrecords = glob.glob(os.path.join(train_dir, "*"))
    valid_tfrecords = glob.glob(os.path.join(valid_dir, "*.TFRecord"))
    test_tfrecords = glob.glob(os.path.join(test_dir, "*.TFRecord"))

    # Construct TFRecord datasets.
    data = {
        "train": tf.data.TFRecordDataset(train_tfrecords),
        "valid": tf.data.TFRecordDataset(valid_tfrecords),
        "test": tf.data.TFRecordDataset(test_tfrecords),
    }

    # Load genetic information.
    genes = None
    if genes_filepath is not None:
        with open(genes_filepath) as fp:
            genes = ["gene/FPKM-UQ/" + line.strip() for line in fp]
    if log_expr_path is not None:
        log_expr_means = np.load(log_expr_path + "_means.npy").tolist()
        log_expr_stds = np.load(log_expr_path + "_stds.npy").tolist()
        genes = [
            g for g, s in zip(genes, log_expr_stds)
            if s > min_log_expr_std
        ]
        log_expr_means = [
            m for m, s in zip(log_expr_means, log_expr_stds)
            if s > min_log_expr_std
        ]
        log_expr_stds = [s for s in log_expr_stds if s > min_log_expr_std]
    else:
        log_expr_means = None
        log_expr_stds = None

    for set_name in ["train", "valid", "test"]:
        # Deserialize and filter.
        data[set_name] = data[set_name].map(
            partial(deserialize, genes=genes),
            num_parallel_calls=num_parallel_calls
        )
        data[set_name] = data[set_name].filter(
            partial(remove_no_genes, genes=genes),
        )
        # Parse and unbatch.
        data[set_name] = data[set_name].map(
            partial(
                parse_features,
                num_classes=num_classes,
                genes=genes,
            ),
            num_parallel_calls=num_parallel_calls
        )
        data[set_name] = data[set_name].unbatch()
        # Decode images.
        data[set_name] = data[set_name].map(
            partial(
                images_decode_and_preprocess,
                train=(set_name == "train")
            ),
            num_parallel_calls=num_parallel_calls
        )
        # Normalize genes.
        data[set_name] = data[set_name].map(
            partial(
                genes_normalize,
                log_expr_means=log_expr_means,
                log_expr_stds=log_expr_stds,
            ),
            num_parallel_calls=num_parallel_calls
        )
        if set_name == "train":
            data[set_name] = data[set_name].shuffle(
                shuffle_buffer_size, seed=seed
            )
        data[set_name] = data[set_name].batch(batch_size).repeat(epochs)

    return data

def parse_features_flipped(features, num_classes, genes=None):
    """Decode features into (image, genes, label) tuples."""
    # Parse image patches.
    patches_jpeg = features["image/encoded"]
    inputs = {"C": patches_jpeg}

    # Parse labels.
    labels = tf.repeat(
        [tf.cast(3 - features["class/label"], dtype=tf.int32)],
        repeats=[tf.shape(patches_jpeg)[0]],
        axis=0
    )
    labels = tf.one_hot(indices=labels, depth=num_classes,)

    # Parse genes.
    if genes is not None:
        genes = [features[g] for g in genes]
        genes = tf.convert_to_tensor(genes, dtype=tf.float32)
        genes = tf.repeat(
            [genes], repeats=[tf.shape(patches_jpeg)[0]], axis=0
        )
        inputs["X"] = genes

    return inputs, labels

def load_data_flipped(
    train_dir, valid_dir, test_dir,
    genes_filepath=None,
    log_expr_path=None,
    epochs=None,
    min_log_expr_std=1e-2,
    num_classes=3,
    batch_size=32,
    num_parallel_calls=16,
    shuffle_buffer_size=4000,
    seed=42,
):
    train_tfrecords = glob.glob(os.path.join(train_dir, "*"))
    valid_tfrecords = glob.glob(os.path.join(valid_dir, "*.TFRecord"))
    test_tfrecords = glob.glob(os.path.join(test_dir, "*.TFRecord"))

    # Construct TFRecord datasets.
    data = {
        "train": tf.data.TFRecordDataset(train_tfrecords),
        "valid": tf.data.TFRecordDataset(valid_tfrecords),
        "test": tf.data.TFRecordDataset(test_tfrecords),
    }

    # Load genetic information.
    genes = None
    if genes_filepath is not None:
        with open(genes_filepath) as fp:
            genes = ["gene/FPKM-UQ/" + line.strip() for line in fp]
    if log_expr_path is not None:
        log_expr_means = np.load(log_expr_path + "_means.npy").tolist()
        log_expr_stds = np.load(log_expr_path + "_stds.npy").tolist()
        genes = [
            g for g, s in zip(genes, log_expr_stds)
            if s > min_log_expr_std
        ]
        log_expr_means = [
            m for m, s in zip(log_expr_means, log_expr_stds)
            if s > min_log_expr_std
        ]
        log_expr_stds = [s for s in log_expr_stds if s > min_log_expr_std]
    else:
        log_expr_means = None
        log_expr_stds = None

    for set_name in ["train", "valid", "test"]:
        # Deserialize and filter.
        data[set_name] = data[set_name].map(
            partial(deserialize, genes=genes),
            num_parallel_calls=num_parallel_calls
        )
        data[set_name] = data[set_name].filter(
            partial(remove_no_genes, genes=genes),
        )
        # Parse and unbatch.
        data[set_name] = data[set_name].map(
            partial(
                parse_features_flipped,
                num_classes=num_classes,
                genes=genes,
            ),
            num_parallel_calls=num_parallel_calls
        )
        data[set_name] = data[set_name].unbatch()
        # Decode images.
        data[set_name] = data[set_name].map(
            partial(
                images_decode_and_preprocess,
                train=(set_name == "train")
            ),
            num_parallel_calls=num_parallel_calls
        )
        # Normalize genes.
        data[set_name] = data[set_name].map(
            partial(
                genes_normalize,
                log_expr_means=log_expr_means,
                log_expr_stds=log_expr_stds,
            ),
            num_parallel_calls=num_parallel_calls
        )
        if set_name == "train":
            data[set_name] = data[set_name].shuffle(
                shuffle_buffer_size, seed=seed
            )
        data[set_name] = data[set_name].batch(batch_size).repeat(epochs)

    return data
