# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utils for autoencoders
from __future__ import print_function, absolute_import, division
import tensorflow as tf

__all__ = [
    'cross_entropy',
    'kullback_leibler'
]


def cross_entropy(actual, predicted, eps=1e-10):
    """Binary cross entropy

    Parameters
    ----------
    actual : TensorFlow ``Tensor``
        Actual

    predicted : TensorFlow ``Tensor``
        Predicted

    eps : float, optional (default=1e-10)
        The amount to offset difference in ``predicted`` and ``actual``
        to avoid any log(0) operations.
    """
    # clip to avoid nan
    p_ = tf.clip_by_value(predicted, eps, 1 - eps)
    return -tf.reduce_sum(actual * tf.log(p_) + (1 - actual) * tf.log(1 - p_), 1)


def kullback_leibler(mu, log_sigma):
    """Gaussian Kullback-Leibler divergence:

        KL(q | p)

    Parameters
    ----------
    mu : TensorFlow ``Tensor``
        The z_mean tensor.

    log_sigma : TensorFlow ``Tensor``
        The z_log_sigma tensor.
    """
    # -0.5 * (1 + log(sigma ** 2) - mu ** 2 - sigma ** 2)
    return -0.5 * tf.reduce_sum(1 + (2 * log_sigma) - (mu ** 2) - tf.exp(2 * log_sigma), 1)
