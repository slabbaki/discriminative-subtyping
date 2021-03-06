"""CEN models."""

import numpy as np
import tensorflow as tf


def build_model(
    encoder,
    input_dtypes,
    input_shapes,
    output_shape,
    output_activation="softmax",
    top_dense_layers=0,
    top_dense_units=128,
    top_dense_activation="relu",
    top_dense_dropout=0.5,
):
    # Input nodes.
    inputs = tf.keras.layers.Input((299,299,3), dtype='float32')

    # Build top dense layers.
    previous = encoder(inputs)
    for _ in range(top_dense_layers):
        previous = tf.keras.layers.Dense(
            top_dense_units, activation=top_dense_activation
        )(previous)
        previous = tf.keras.layers.Dropout(top_dense_dropout)(previous)

    # Build outputs.
    outputs = tf.keras.layers.Dense(
        np.prod(output_shape), activation=output_activation
    )(previous)

    # Create a Keras model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
