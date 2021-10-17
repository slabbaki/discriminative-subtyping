from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os

from os import listdir
from os.path import isfile, join

def decode_jpeg(image_buffer, scope=None):
  with tf.compat.v1.keras.backend.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    image = tf.compat.v1.image.decode_jpeg(image_buffer, channels=3)
    image = tf.compat.v1.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def distort_color(image, thread_id=0, scope=None):
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
  # with tf.compat.v1.keras.backend.name_scope(values=[image], name=scope, default_name='distort_color'):
  #   color_ordering = thread_id % 2

  #   if color_ordering == 0:
  #     image = tf.compat.v1.image.random_brightness(image, max_delta=32. / 255.)
  #     image = tf.compat.v1.image.random_saturation(image, lower=0.5, upper=1.5)
  #     image = tf.compat.v1.image.random_hue(image, max_delta=0.2)
  #     image = tf.compat.v1.image.random_contrast(image, lower=0.5, upper=1.5)
  #   elif color_ordering == 1:
  #     image = tf.compat.v1.image.random_brightness(image, max_delta=32. / 255.)
  #     image = tf.compat.v1.image.random_contrast(image, lower=0.5, upper=1.5)
  #     image = tf.compat.v1.image.random_saturation(image, lower=0.5, upper=1.5)
  #     image = tf.compat.v1.image.random_hue(image, max_delta=0.2)

  #   # The random_* ops do not necessarily clamp.
  #   image = tf.compat.v1.clip_by_value(image, 0.0, 1.0)
  return image

def distort_image(image, height, width, thread_id=0, scope=None):
    
    resize_method = thread_id % 4
    distorted_image = tf.compat.v1.image.resize_images(image, [height, width],
                                             method=resize_method)

    distorted_image.set_shape([height, width, 3])
    # distorted_image = tf.compat.v1.image.random_flip_left_right(distorted_image)
    # distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image

def eval_image(image, height, width, scope=None):

  with tf.compat.v1.keras.backend.name_scope(values=[image, height, width], name=scope,
                     default_name='eval_image'):
    image = tf.compat.v1.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image
    
def image_preprocessing(image_buffer, train, thread_id=0):
  """Decode and preprocess one image for evaluation or training.

  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """

  image = decode_jpeg(image_buffer)
  height = 299
  width = 299

  if train:
    image = distort_image(image, height, width, thread_id)
  else:
    image = eval_image(image, height, width)

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image



def parse_img_proto(example_serialized, train):

    feature_map = {
        'image/encoded': tf.io.FixedLenSequenceFeature([], dtype=tf.string,
                                            default_value='', allow_missing=True),
    }

    features = tf.io.parse_single_example(example_serialized, feature_map)
    
    return features['image/encoded']

def parse_gen_proto(example_serialized, feature_map, genes, means, stds, train):

    features = tf.io.parse_single_example(example_serialized, feature_map)

    genetics = [tf.math.log(features[g] + 1e-3) for g in genes]
    
    return (tf.convert_to_tensor(genetics, dtype=np.float32) - tf.convert_to_tensor(means, dtype=np.float32)) / tf.convert_to_tensor(stds, dtype=np.float32)

def parse_lbl_proto(example_serialized, train):

    feature_map = {'class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
    }

    features = tf.io.parse_single_example(example_serialized, feature_map)
    
    label = tf.keras.backend.cast(features['class/label'] - 1, dtype=tf.int32)
    label = tf.keras.backend.one_hot(label, num_classes=3)
    label = label[0,:]
    
    return label

def image_fn(img, train, num_preprocess_threads):

    thread_id = np.random.randint(num_preprocess_threads)
    img = image_preprocessing(img, train, thread_id)
    height = 299
    width = 299
    depth = 3
    img = tf.keras.backend.cast(img, tf.float32)
    img = tf.reshape(img, shape=[height, width, depth])

    return img

def tfrecord_train_input_fn(train, path, batch_size=128, num_preprocess_threads=None, epochs=None):

    feature_map = {}
    genes = []
    gene_filename = '/content/cosmic_gene_names.txt'
    means_path = '/content/genome_means.npy'
    stds_path = '/content/genome_stds.npy'
    cosmic_selector_path = '/content/cosmic_selector.npy'
    
    genome_means = np.load(means_path)
    genome_stds = np.load(stds_path)
    cosmic_selector = np.load(cosmic_selector_path)

    means = genome_means.dot(cosmic_selector)
    stds = genome_stds.dot(cosmic_selector)

    float_feature = tf.io.FixedLenFeature([], dtype=tf.float32,
                                                default_value=-1.0)
    if gene_filename is not None:
      with open(gene_filename) as fp:
        gene = fp.readline()
        while gene:
          name = 'gene/FPKM-UQ/' + gene[:-1]
          feature_map.update(
              {name: float_feature}
              )
          genes.append(name)
          gene = fp.readline()

    dataset = tf.data.TFRecordDataset(path[0])
    with tf.device('/cpu:0'):
        if train:
            dataset_img = dataset.map(lambda   x:parse_img_proto(x, train)).unbatch()
            dataset_img = dataset_img.map(lambda   x:image_fn(x, train, num_preprocess_threads))
            
            img_times = 0
            for x in dataset_img:
              img_times += 1
            
            dataset_gen = dataset.map(lambda   x:parse_gen_proto(x, feature_map, genes, means, stds, train)).repeat(img_times)
            dataset_lbl = dataset.map(lambda   x:parse_lbl_proto(x, train)).repeat(img_times)

            for files in path[1:]:
                dataset = tf.data.TFRecordDataset(files)
                dataset_img_tmp = dataset.map(lambda   x:parse_img_proto(x, train)).unbatch()
                dataset_img_tmp = dataset_img_tmp.map(lambda   x:image_fn(x, train, num_preprocess_threads))

                img_times = 0
                for x in dataset_img_tmp:
                    img_times += 1

                dataset_gen_tmp = dataset.map(lambda   x:parse_gen_proto(x, feature_map, genes, means, stds, train)).repeat(img_times)
                dataset_lbl_tmp = dataset.map(lambda   x:parse_lbl_proto(x, train)).repeat(img_times)
                
                dataset_img = dataset_img.concatenate(dataset_img_tmp)
                dataset_gen = dataset_gen.concatenate(dataset_gen_tmp)
                dataset_lbl = dataset_lbl.concatenate(dataset_lbl_tmp)

            return dataset_img.batch(batch_size), dataset_gen.batch(batch_size), dataset_lbl.batch(batch_size)