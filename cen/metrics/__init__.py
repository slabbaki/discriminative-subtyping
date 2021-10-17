import tensorflow as tf

from . import survival


def get(name, **kwargs):
    if name == "survival_accuracy_score":
        return survival.SurvivalAccuracyScore(**kwargs)
    elif name == "survival_brier_score":
        return survival.SurvivalBrierScore(**kwargs)
    else:
        if kwargs:
            return tf.keras.metrics.get({
                "class_name": name,
                "config": kwargs,
            })
        else:
            return tf.keras.metrics.get(name)

