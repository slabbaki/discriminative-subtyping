import tensorflow as tf

from .survival import NegativeLogLikelihoodSurvival


def get(name, **kwargs):
    if name == "nll_survival":
        return NegativeLogLikelihoodSurvival(**kwargs)
    else:
        if kwargs:
            return tf.keras.losses.get({
                "class_name": name,
                "config": kwargs,
            })
        else:
            return tf.keras.losses.get(name)

