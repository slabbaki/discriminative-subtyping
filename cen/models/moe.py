"""Mixtures of contextual linear experts."""

import numpy as np
import tensorflow as tf

from .. import layers


def build_model(
    encoder,
    input_dtypes,
    input_shapes,
    output_shape,
    num_experts,
    expert_kwargs,
    mixture_kwargs,
    top_dense_layers=1,
    top_dense_units=128,
    top_dense_activation="relu",
    top_dense_dropout=0.5,
):
    context_dtype, features_dtype = input_dtypes
    context_shape, features_shape = input_shapes

    # Input nodes.
    context = tf.keras.layers.Input(context_shape, dtype=context_dtype)
    features = tf.keras.layers.Input(features_shape, dtype=features_dtype)

    # Build encoded context.
    encodings = encoder(context)
    for _ in range(top_dense_layers):
        encodings = tf.keras.layers.Dense(
            top_dense_units, activation=top_dense_activation
        )(encodings)
        encodings = tf.keras.layers.Dropout(top_dense_dropout)(encodings)

    # Experts.
    experts = [
        tf.keras.layers.Dense(np.prod(output_shape), **expert_kwargs)
        for _ in range(num_experts)
    ]

    # Build contextual mixture of experts layer.
    features_flat = tf.keras.layers.Flatten()(features)
    mixture = layers.get_contextual(
        "mixture", experts=experts, **mixture_kwargs
    )
    outputs = mixture((encodings, features_flat))
    outputs = tf.keras.layers.Reshape(output_shape)(outputs)

    # Create a Keras model.
    model = tf.keras.models.Model(inputs=(context, features), outputs=outputs)

    return model
