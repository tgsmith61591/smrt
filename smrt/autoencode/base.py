# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Base utils for the autoencoder(s)

from __future__ import division, absolute_import, division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from ..utils import validate_float

__all__ = [
    'BaseAutoEncoder',
    'ReconstructiveMixin'
]

DTYPE = tf.float32


def _validate_positive_integer(instance, name):
    val = getattr(instance, name)
    try:
        assert not isinstance(val, (bool, np.bool))
        res = int(val)
        assert res >= 0
    except:
        raise ValueError('%s must be an int >= 0' % name)
    return res


def _validate_float(instance, name, upper_bound=1.):
    try:
        res = float(getattr(instance, name))
        validate_float(res, name, upper_bound, gtet=False)
    except:
        raise
    return res


class BaseAutoEncoder(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."
    """

    def __init__(self, activation_function, learning_rate, n_epochs, batch_size, n_hidden, compression_ratio,
                 min_change, verbose, display_step, learning_function, early_stopping, initial_weight_stddev, seed,
                 xavier_init):

        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.compression_ratio = compression_ratio
        self.min_change = min_change
        self.verbose = verbose
        self.display_step = display_step
        self.learning_function = learning_function
        self.early_stopping = early_stopping
        self.initial_weight_stddev = initial_weight_stddev
        self.seed = seed
        self.xavier_init = xavier_init

    def _validate_for_fit(self):
        # validate floats and other params...
        self.compression_ratio = _validate_float(self, 'compression_ratio', upper_bound=np.inf)
        self.min_change = _validate_float(self, 'min_change')
        self.learning_rate = _validate_float(self, 'learning_rate')
        self.seed = _validate_positive_integer(self, 'seed')
        self.verbose = _validate_positive_integer(self, 'verbose')
        self.display_step = _validate_positive_integer(self, 'display_step')
        self.n_epochs = _validate_positive_integer(self, 'n_epochs')
        self.initial_weight_stddev = _validate_float(self, 'initial_weight_stddev', upper_bound=np.inf)

    @abstractmethod
    def _encoding_function(self, x, weights, biases, activation):
        """The encoding function should define the function that encodes the ``x``
        into a lower dimensional space. In the case of image recognition, it could be considered
        the 'recognition' task.
        """

    @abstractmethod
    def _decoding_function(self, x, weights, biases, activation):
        """The decoding function should define the function that projects data in the 'encoded space'
        back into the original feature space. In computer vision, this would may be a deconvolution layer.
        """

    def clean_session(self):
        if hasattr(self, 'sess'):
            self.sess.close()
            delattr(self, 'sess')


class ReconstructiveMixin:
    @abstractmethod
    def reconstruct(self, X):
        """Pass a matrix, ``X``, through both the encoding and decoding functions
        to obtain a reconstructed observation.
        """


class GenerativeMixin:
    @abstractmethod
    def generate(self):
        """Do something currently undefined...
        """
        #todo
