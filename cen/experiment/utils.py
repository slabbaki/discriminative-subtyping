"""Experiment utils."""

import os

import tensorflow as tf

from .. import losses
from .. import metrics
from .. import models
from .. import networks


class ModeKeys(object):
    TRAIN = "train"
    EVAL = "eval"


def get_input_dtypes(data):
    """Returns input get_input_dtypes."""
    return ('float32','float32')


def get_input_shapes(data):
    """Returns input shapes."""
    return ((299,299,3),(696))


def get_output_shape(data):
    """Returns output shapes."""
    return (3,)


def build(cfg, input_dtypes, input_shapes, output_shape, mode=ModeKeys.TRAIN):
    """Builds model and callbacks for training or evaluation."""

    # Build model.
    net = networks.get(**cfg.network)
    model = models.get(
        cfg.model.name,
        encoder=net,
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        **cfg.model.kwargs,
    )

    # Build loss and optimizer.
    loss = losses.get(**cfg.train.loss)
    opt = tf.keras.optimizers.get(dict(**cfg.optimizer))

    # Build metrics.
    metrics_list = None
    if cfg.eval.metrics:
        metrics_list = [metrics.get(**v) for _, v in cfg.eval.metrics.items()]

    if mode == ModeKeys.TRAIN:
        model.compile(optimizer=opt, loss=loss, metrics=metrics_list)
        callbacks = []
        if cfg.train.checkpoint_kwargs:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(os.getcwd(), "checkpoint"),
                    **cfg.train.checkpoint_kwargs,
                )
            )
        if cfg.train.tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(os.getcwd(), "tensorboard")
                )
            )
        return model, callbacks

    if mode == ModeKeys.EVAL:
        checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        model.compile(loss=loss, optimizer=opt, metrics=metrics_list)
        return model
