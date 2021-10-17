"""Evaluation."""

import logging

from . import utils

logger = logging.getLogger(__name__)


def evaluate(cfg, data):
    if cfg.eval.verbose:
        logger.info("Building...")

    model = utils.build(
        cfg,
        input_dtypes='float32',
        input_shapes=(299,299,3),
        output_shape=(1,3),
        mode=utils.ModeKeys.EVAL
    )

    if cfg.eval.verbose:
        logger.info("Evaluating...")

    metrics = {}
    for set_name, dataset in data.items():
        if dataset is None:
            continue
        metric_names = ["loss"] + list(cfg.eval.metrics.keys())
        metric_values = model.evaluate(
            dataset,
            verbose=0,
        )
        metrics[set_name] = dict(zip(metric_names, metric_values))
        if cfg.eval.verbose:
            logger.info(f"{set_name} metrics: {metrics[set_name]}")

    return metrics
