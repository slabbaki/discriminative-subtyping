"""A collection of DenseNet models."""

import tensorflow as tf

from .Keras_custom_models import Inceptionv3_model

__all__ = ["Inception_v3"]


def Inception_v3(include_top=False, weights=None, input_tensor=None, 
input_shape=(299,299,3), pooling='avg', classes=3, isTraining=True):
    
    def network(inputs):
        previous = Inceptionv3_model.InceptionV3(
            include_top=include_top,
                weights=weights,
                input_tensor=inputs,
                input_shape=input_shape,
                pooling=pooling,
                classes=classes,
                isTraining = isTraining
        ).output
        return previous

    return network