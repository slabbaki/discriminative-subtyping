"""Training."""

import logging
import os

from . import utils

logger = logging.getLogger(__name__)


def train(cfg, train_data, validation_data=None):
    logger.info("Building...")

    input_dtypes = {
        k: train_data.element_spec[0][k].dtype for k in ["C", "X"]
    }
    input_shapes = {
        k: train_data.element_spec[0][k].shape[1:] for k in ["C", "X"]
    }
    output_shape = train_data.element_spec[1].shape[1:]
    model, callbacks = utils.build(
        cfg,
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        mode=utils.ModeKeys.TRAIN
    )
    model.summary()
    logger.info("Training...")

    history = model.fit(
        train_data,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        validation_data=validation_data,
        verbose=cfg.train.verbose,
        steps_per_epoch=int((1440424-1e-5)/cfg.dataset.context_kwargs.batch_size)+1,
        validation_steps=int((299277-1e-5)/cfg.dataset.context_kwargs.batch_size)+1
    )

    # Save model if checkpointing was off.
    if cfg.train.checkpoin_kwargs is None:
        checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
        model.save(checkpoint_path)

    return history
