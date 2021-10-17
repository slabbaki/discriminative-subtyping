from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


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
  with tf.compat.v1.keras.backend.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2

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

def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    
    sample_distorted_bounding_box = tf.compat.v1.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    resize_method = thread_id % 4
    distorted_image = tf.compat.v1.image.resize_images(image, [height, width],
                                             method=resize_method)

    distorted_image.set_shape([height, width, 3])
    distorted_image = tf.compat.v1.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, thread_id)
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
    
def image_preprocessing(image_buffer, bbox, train, thread_id=0):
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
  if bbox is None:
    raise ValueError('Please supply a bounding box.')

  image = decode_jpeg(image_buffer)
  height = 299
  width = 299

  if train:
    image = distort_image(image, height, width, bbox, thread_id)
  else:
    image = eval_image(image, height, width)

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image



def parse_example_proto(example_serialized, train, num_preprocess_threads):

    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.io.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }

    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.keras.backend.cast(features['image/class/label'] - 1, dtype=tf.int32)
    label = tf.keras.backend.one_hot(label, num_classes=3)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    
    thread_id = np.random.randint(num_preprocess_threads)
    img = image_preprocessing(features['image/encoded'],bbox, train, thread_id)

    height = 299
    width = 299
    depth = 3

    img = tf.keras.backend.cast(img, tf.float32)
    img = tf.reshape(img, shape=[height, width, depth])
    
    return img, label[0,:]

def tfrecord_train_input_fn(train, path, batch_size=128, num_preprocess_threads=None, epochs=None):
    dataset = tf.data.TFRecordDataset(path)
    with tf.device('/cpu:0'):
        if train:    
            dataset = dataset.map(lambda   x:parse_example_proto(x, train, num_preprocess_threads)).shuffle(True).batch(batch_size).repeat(epochs)
        else:
            dataset = dataset.map(lambda   x:parse_example_proto(x, train, num_preprocess_threads)).batch(batch_size).repeat(epochs)
    return dataset