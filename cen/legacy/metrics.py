"""A collection of custom metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics


def accuracy_score(y_true, logits, t, censored_indicator):
    """Accuracy score of the survival probability predictions.

    Args:
        y_true : <float32> [batch_size, time_steps, 2]
            y_true[i, t, 0] should indicate whether the instance i was censored
            at time t; y_true[i, t, 1] indicates occurrence of the event for
            instance i at time t_event <= t.
        logits : <float32> [batch_size, time_steps, 1]
        t : float
        censored_indicator : float

    Returns:
        accuracy : <float32> []
    """
    # Resolve inputs.
    y_true_c = y_true[:, :, 0]
    y_true_e = y_true[:, :, 1]
    logits = logits[:, :, 0]

    # Find the non censored instances.
    not_censored_at_t = tf.not_equal(y_true_c[:, t], censored_indicator)
    y_true_uc = tf.boolean_mask(y_true_e, not_censored_at_t)
    logits_uc = tf.boolean_mask(logits, not_censored_at_t)

    # Compute survival probabilities.
    logits_uc_cumsum = tf.cumsum(logits_uc, axis=-1, reverse=True)
    logits_uc_cumsum_padded = tf.pad(
        logits_uc_cumsum,
        paddings=[[0, 0], [0, 1]])
    lognum = tf.reduce_logsumexp(logits_uc_cumsum_padded[:, t:], axis=-1)
    logdenom = tf.reduce_logsumexp(logits_uc_cumsum_padded, axis=-1)
    survival_prob = tf.exp(lognum - logdenom)

    # Compute accuracy on the non-censored instances.
    survived = 1 - y_true_uc[:, t]
    acc = metrics.binary_accuracy(survived, survival_prob)

    return acc


def get_accuracy_scores(at_time_intervals, censored_indicator=1.0):
    """Construct accuracy scores for survival probability predictions."""
    metrics = []
    for t in at_time_intervals:
        metric = partial(
            accuracy_score, t=t, censored_indicator=censored_indicator)
        metric.__name__ = 'acc_at_%d' % t
        metrics.append(metric)
    return metrics


def brier_score(y_true, logits, t, censored_indicator):
    """Brier score of the survival probability predictions.

    Args:
        y_true : <float32> [batch_size, time_steps, 2]
            y_true[i, t, 0] should indicate whether the instance i was censored
            at time t; y_true[i, t, 1] indicates occurrence of the event for
            instance i at time t_event <= t.
        logits : <float32> [batch_size, time_steps, 1]
        t : float
        censored_indicator : float

    Returns:
        score : <float32> []
    """
    # Resolve inputs.
    y_true_c = y_true[:, :, 0]
    y_true_e = y_true[:, :, 1]
    logits = logits[:, :, 0]

    # Find the non censored instances.
    not_censored_at_t = tf.not_equal(y_true_c[:, t], censored_indicator)
    y_true_uc = tf.boolean_mask(y_true_e, not_censored_at_t)
    logits_uc = tf.boolean_mask(logits, not_censored_at_t)

    # Compute survival probabilities.
    logits_uc_cumsum = tf.cumsum(logits_uc, axis=-1, reverse=True)
    logits_uc_cumsum_padded = tf.pad(
        logits_uc_cumsum,
        paddings=[[0, 0], [0, 1]])
    lognum = tf.reduce_logsumexp(logits_uc_cumsum_padded[:, t:], axis=-1)
    logdenom = tf.reduce_logsumexp(logits_uc_cumsum_padded, axis=-1)
    survival_prob = tf.exp(lognum - logdenom)

    # Compute brier score.
    survived = 1 - y_true_uc[:, t]
    score = metrics.mean_squared_error(survived, survival_prob)

    return score


def get_brier_scores(at_time_intervals, censored_indicator=1.0):
    """Construct Brier score metrics for survival probability predictions."""
    metrics = []
    for t in at_time_intervals:
        metric = partial(
            brier_score, t=t, censored_indicator=censored_indicator)
        metric.__name__ = 'bs_at_%d' % t
        metrics.append(metric)
    return metrics


def predictive_loss(y_true, y_pred, loss, loss_matrix, censored_indicator):
    """Predictive loss of the survival probability predictions."""
    n = K.cast(K.shape(y_true)[0], dtype='float32')
    m = loss_matrix.shape[0]

    censored = K.not_equal(y_true[:, -1, 1], censored_indicator)
    not_censored = K.equal(y_true[:, -1, 1], censored_indicator)

    lognum = K.cumsum(y_pred[:, :, 0], axis=1, reverse=True)
    logdenom = K.logsumexp(K.cumsum(y_pred, axis=1, reverse=True), axis=1)
    probs = K.exp(lognum - logdenom)

    pred_time = 1. + K.cast(K.argmin(
        K.dot(probs, K.variable(loss_matrix)), axis=1), dtype='float32')
    pred_time_c = K.boolean_mask(pred_time, censored)
    pred_time_nc = K.boolean_mask(pred_time, not_censored)

    y_true_c = K.boolean_mask(y_true[:, :, 0], censored)
    true_time_c = K.cast(K.argmax(y_true_c, axis=1), dtype='float32')
    true_time_c = K.switch(
        K.equal(y_true_c[:, -1], censored_indicator),
        1. + true_time_c, K.ones_like(true_time_c) * m)

    y_true_nc = K.boolean_mask(y_true[:, :, 1], not_censored)
    true_time_nc = K.cast(K.argmax(y_true_nc, axis=1), dtype='float32')

    loss_c = K.sum(loss(pred_time_c, true_time_c, censored=True))
    loss_nc = K.sum(loss(pred_time_nc, true_time_nc, censored=False))

    return (loss_c + loss_nc) / n


def get_predictive_loss(loss, loss_matrix, censored_indicator=1.0):
    """Construct predictive loss for survival probability predictions."""
    metric = partial(
        predictive_loss,
        loss=loss,
        loss_matrix=loss_matrix,
        censored_indicator=censored_indicator)
    metric.__name__ =  loss.__name__
    return [metric]


def get_ae_loss_matrix(m):
    p = np.arange(1, m + 1).astype('float32')
    t = np.arange(1, m + 1).astype('float32')[:, None]
    return np.abs(p - t)


def get_rae_loss_matrix(m):
    p = np.arange(1, m + 1).astype('float32')
    t = np.arange(1, m + 1).astype('float32')[:, None]
    return np.abs(p - t) / p


def ae_loss(p, t, censored=False):
    ae = K.abs(p - t)
    if censored:
        ae = K.switch(K.greater(t, p), ae, K.zeros_like(ae))
    return ae


def rae_loss(p, t, censored=False):
    rae = K.abs(p - t) / p
    if censored:
        rae = K.switch(K.greater(t, p), rae, K.zeros_like(rae))
    return rae
