# Initial version without Dropout or any customization we can find in DeepPATH


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""Inception V3 model for Keras.
Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export


WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


@keras_export('keras.applications.inception_v3.InceptionV3',
              'keras.applications.InceptionV3')
def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=(299,299,3),
                pooling=None,
                classes=1000,
                isTraining = True):
  """Instantiates the Inception v3 architecture.
  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.
  Arguments:
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(299, 299, 3)` (with `channels_last` data format)
      or `(3, 299, 299)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 75.
      E.g. `(150, 150, 3)` would be one valid value.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
  Returns:
    A Keras model instance.
  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
  """
  end_points = {}
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=299,
      min_size=75,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  if backend.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = 3

  end_points['conv0'] = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
  end_points['conv1'] = conv2d_bn(end_points['conv0'], 32, 3, 3, padding='valid')
  end_points['conv2'] = conv2d_bn(end_points['conv1'], 64, 3, 3)
  end_points['pool1'] = layers.MaxPooling2D((3, 3), strides=(2, 2))(end_points['conv2'])

  end_points['conv3'] = conv2d_bn(end_points['pool1'], 80, 1, 1, padding='valid')
  end_points['conv4'] = conv2d_bn(end_points['conv3'], 192, 3, 3, padding='valid')
  end_points['pool2'] = layers.MaxPooling2D((3, 3), strides=(2, 2))(end_points['conv4'])
  net = end_points['pool2']
  # mixed 0: 35 x 35 x 256
  branch1x1 = conv2d_bn(net, 64, 1, 1)

  branch5x5 = conv2d_bn(net, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(net, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(net)
  branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
  net = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed0')
  end_points['mixed_35x35x256a'] = net
  # mixed 1: 35 x 35 x 288
  branch1x1 = conv2d_bn(net, 64, 1, 1)

  branch5x5 = conv2d_bn(net, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(net, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(net)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  net = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed1')
  end_points['mixed_35x35x288a'] = net
  # mixed 2: 35 x 35 x 288
  branch1x1 = conv2d_bn(net, 64, 1, 1)

  branch5x5 = conv2d_bn(net, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(net, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(net)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  net = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed2')
  end_points['mixed_35x35x288b'] = net
  # mixed 3: 17 x 17 x 768
  branch3x3 = conv2d_bn(net, 384, 3, 3, strides=(2, 2), padding='valid')

  branch3x3dbl = conv2d_bn(net, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(
      branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(net)
  net = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed3')
  end_points['mixed_17x17x768a'] = net
  # mixed 4: 17 x 17 x 768
  branch1x1 = conv2d_bn(net, 192, 1, 1)

  branch7x7 = conv2d_bn(net, 128, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(net, 128, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed4')
  end_points['mixed_17x17x768b'] = net
  # mixed 5: 17 x 17 x 768
  
  branch1x1 = conv2d_bn(net, 192, 1, 1)
    
  branch7x7 = conv2d_bn(net, 160, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    
  branch7x7dbl = conv2d_bn(net, 160, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    
  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(
                                            net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed' + str(5))
  end_points['mixed_17x17x768c'] = net
  # mixed 6: 17 x 17 x 768
  
  branch1x1 = conv2d_bn(net, 192, 1, 1)
    
  branch7x7 = conv2d_bn(net, 160, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    
  branch7x7dbl = conv2d_bn(net, 160, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    
  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(
                                            net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed' + str(6))
  end_points['mixed_17x17x768d'] = net
  # mixed 7: 17 x 17 x 768
  branch1x1 = conv2d_bn(net, 192, 1, 1)

  branch7x7 = conv2d_bn(net, 192, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(net, 192, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = layers.AveragePooling2D(
      (3, 3), strides=(1, 1), padding='same')(net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed7')
  end_points['mixed_17x17x768e'] = net
  '''
  # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.compat.v1.variable_scope('aux_logits'):
          aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                  padding='VALID')
          aux_logits = ops.flatten(aux_logits)
          aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                              stddev=0.001, restore=restore_logits)
          end_points['aux_logits'] = aux_logits
  '''
  # mixed 8: 8 x 8 x 1280
  branch3x3 = conv2d_bn(net, 192, 1, 1)
  branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

  branch7x7x3 = conv2d_bn(net, 192, 1, 1)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
  branch7x7x3 = conv2d_bn(
      branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(net)
  net = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                         axis=channel_axis,
                         name='mixed8')
  end_points['mixed_17x17x1280a'] = net
  # mixed 9: 8 x 8 x 2048
  branch1x1 = conv2d_bn(net, 320, 1, 1)

  branch3x3 = conv2d_bn(net, 384, 1, 1)
  branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
  branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
  branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                 axis=channel_axis,
                                 name='mixed9_' + str(0))

  branch3x3dbl = conv2d_bn(net, 448, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
  branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
  branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
  branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                    axis=channel_axis)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(
                                            net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed' + str(9))
  end_points['mixed_8x8x2048a'] = net
  # mixed 10: 8 x 8 x 2048
  branch1x1 = conv2d_bn(net, 320, 1, 1)

  branch3x3 = conv2d_bn(net, 384, 1, 1)
  branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
  branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
  branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                 axis=channel_axis,
                                 name='mixed9_' + str(1))

  branch3x3dbl = conv2d_bn(net, 448, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
  branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
  branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
  branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                    axis=channel_axis)

  branch_pool = layers.AveragePooling2D((3, 3),
                                        strides=(1, 1),
                                        padding='same')(
                                            net)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  net = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                         axis=channel_axis,
                         name='mixed' + str(10))
  end_points['mixed_8x8x2048b'] = net
  if include_top:
    # Classification block
    net = layers.GlobalAveragePooling2D(name='avg_pool')(net)
    net = layers.Dropout(0.2)(net, training=isTraining)
    predictions = layers.Dense(classes, activation='softmax', name='predictions')(net)
  else:
    if pooling == 'avg':
      net = layers.GlobalAveragePooling2D()(net)
      predictions = layers.Dropout(0.2)(net, training=isTraining)
    elif pooling == 'max':
      net = layers.GlobalMaxPooling2D()(net, training=isTraining)
      predictions = layers.Dropout(0.2)(net)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, predictions, name='inception_v3')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      weights_path = data_utils.get_file(
          'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    else:
      weights_path = data_utils.get_file(
          'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
  """Utility function to apply conv + BN.
  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.
  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
  if backend.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = 3
  x = layers.Conv2D(
      filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      use_bias=False,
      name=conv_name)(
          x)
  x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = layers.Activation('relu', name=name)(x)
  return x


@keras_export('keras.applications.inception_v3.preprocess_input')
def preprocess_input(x, data_format=None):
  return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


@keras_export('keras.applications.inception_v3.decode_predictions')
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)