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
from ..utils import validate_float, get_random_state

__all__ = [
    'BaseAutoEncoder',
    'GenerativeMixin',
    'ReconstructiveMixin'
]

# common dtype used throughout
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

    def __init__(self, activation_function, learning_rate, n_epochs, batch_size, n_hidden, min_change,
                 verbose, display_step, learning_function, early_stopping, bias_strategy, random_state,
                 layer_type, dropout, scope):

        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.min_change = min_change
        self.verbose = verbose
        self.display_step = display_step
        self.learning_function = learning_function
        self.early_stopping = early_stopping
        self.bias_strategy = bias_strategy
        self.random_state = get_random_state(random_state)
        self.layer_type = layer_type
        self.dropout = dropout
        self.scope = scope

        # at exit, make sure we close the session
        import atexit
        atexit.register(self.clean_session)

    def _validate_for_fit(self):
        # validate floats and other params...
        self.min_change = _validate_float(self, 'min_change')
        self.learning_rate = _validate_float(self, 'learning_rate')
        self.verbose = _validate_positive_integer(self, 'verbose')
        self.display_step = _validate_positive_integer(self, 'display_step')
        self.n_epochs = _validate_positive_integer(self, 'n_epochs')

    @abstractmethod
    def transform(self, X):
        """Inherited (nominally) from the ``TransformerMixin``, the ``transform`` method encodes the
        input, projecting it into the compressed feature space. The transform method should return a
        numpy array.
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
    def generate(self, *args, **kwargs):
        """Generate a new example or set of examples, given a fit generative
        model. Usually, the args passed in will relate to some fit parameters.
        """
