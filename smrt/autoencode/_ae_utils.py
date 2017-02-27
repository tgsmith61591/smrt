# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utils for autoencoders
from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
from . import base

__all__ = [
    'relu',
    'sigmoid',
    'xavier_initialization'
]


# local modules - since the autoencoder is designed to be able
# to score offline (i.e., no tensorflow session), we need local
# copies of activation functions that doesn't need a session to be
# live to run. This limits what activation functions we can use...
def relu(x):
    return np.maximum(x, 0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xavier_initialization(shape, seed, c=1):
    # Xavier initialization of network weights
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    # see: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    fan_in, fan_out = shape  # will raise ValueError if unpacks incorrectly
    low = -c * np.sqrt(6.0 / (fan_in + fan_out))
    high = c * np.sqrt(6.0 / (fan_in + fan_out))

    return tf.random_uniform(shape=[fan_in, fan_out], minval=low, maxval=high,
                             dtype=base.DTYPE, seed=seed)
