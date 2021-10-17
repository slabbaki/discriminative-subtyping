"""Entry point for running experiments."""

import hydra
import logging
import os
import pickle

from . import data
from .experiment import train, evaluate, cross_validate

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yaml", strict=False)
def main(cfg):
    logger.info("Experiment config:\n" + cfg.pretty())

    logger.info("Loading data...")
    datasets = data.load(
        name=cfg.dataset.name,
        epochs=cfg.train.epochs,
        context_kwargs=cfg.dataset.context_kwargs,
        feature_kwargs=cfg.dataset.feature_kwargs,
        max_train_size=cfg.dataset.max_train_size,
        permute=cfg.dataset.permute,
        seed=cfg.dataset.seed,
    )
    datasets_flipped = data.load_flipped(
        name=cfg.dataset.name,
        epochs=cfg.train.epochs,
        context_kwargs=cfg.dataset.context_kwargs,
        feature_kwargs=cfg.dataset.feature_kwargs,
        max_train_size=cfg.dataset.max_train_size,
        permute=cfg.dataset.permute,
        seed=cfg.dataset.seed,
    )

    # Cross-validation.
    if cfg.crossval:
        metrics = cross_validate(cfg, datasets)
        save_path = os.path.join(os.getcwd(), "cv.metrics.pkl")
        with open(save_path, "wb") as fp:
            pickle.dump(metrics, fp)
    else:
        # Train.
        if cfg.train:
            history = train(cfg, datasets["train"], datasets_flipped["valid"])
            save_path = os.path.join(os.getcwd(), "train.history.pkl")
            with open(save_path, "wb") as fp:
                pickle.dump(history.history, fp)

        # Evaluate.
        if cfg.eval:
            metrics = evaluate(cfg, datasets_flipped)
            save_path = os.path.join(os.getcwd(), "eval.metrics.pkl")
            with open(save_path, "wb") as fp:
                pickle.dump(metrics, fp)


if __name__ == "__main__":
    main()
