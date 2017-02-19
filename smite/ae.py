# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The autoencoder

from __future__ import division, absolute_import, division
from .utils import get_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.externals import six
from numpy.random import RandomState
import numpy as np
import tensorflow as tf
import uuid
import timeit
import sys

__all__ = [
    'AutoEncoder'
]

if sys.version_info[0] >= 3:
    long = int
    NoneType = type(None)
else:
    from types import NoneType

PERMITTED_ACTIVATIONS = {
    'sigmoid': tf.nn.sigmoid,
    'relu': tf.nn.relu
}


def _validate_float(val, name):
    try:
        res = float(val)
        assert 0 < val < 1  # the assertion error will be caught and valueerror raised
    except:
        raise ValueError('%s must be a float between 0 and 1' % name)
    return res


def _weights_biases_from_hidden(n_hidden, n_features, compression):
    """Internal method. Given the hidden layer structure, number of features,
    and compression factor, generate the initial weights and bias matrix.
    """
    if n_hidden is None:
        n_hidden = max(1, int(compression * n_features))

    if not isinstance(n_hidden, (list, tuple)):
        if not isinstance(n_hidden, (long, int, np.int)):
            raise ValueError('n_hidden must be an int, tuple or list')

        # it's an int
        weights = {
            'encode': {'h0': tf.Variable(tf.random_normal([n_features, n_hidden]))},
            'decode': {'h0': tf.Variable(tf.random_normal([n_hidden, n_features]))}
        }

        biases = {
            'encode': {'b0': tf.Variable(tf.random_normal([n_hidden]))},
            'decode': {'b0': tf.Variable(tf.random_normal([n_features]))}
        }
    else:
        # it's iterable. There will be two times as many layers as the length of n_hidden:
        # n_hidden * encode layer, and n_hidden * decode layer. Since the dimensions are
        # piped into one another, stagger them (zipped with a lag), and then reverse for
        # the decode layer. First, though, append n_features to n_hidden
        n_hidden.insert(0, n_features)

        encode_dimensions = list(zip(n_hidden[:-1], n_hidden[1:]))
        decode_dimensions = [(v, k) for k, v in reversed(encode_dimensions)]  # pyramid back to n_features

        weights, biases = {'encode': {}, 'decode': {}}, {'encode': {}, 'decode': {}}
        for i, t in enumerate(encode_dimensions):
            enc_a, enc_b = t
            dec_a, dec_b = decode_dimensions[i]

            # initialize weights for encode/decode layer. While the encode layeres progress through the
            # zipped dimensions, the decode layer steps back up to eventually mapping back to the input space
            weights['encode']['h%i' % i] = tf.Variable(tf.random_normal([enc_a, enc_b]))
            weights['decode']['h%i' % i] = tf.Variable(tf.random_normal([dec_a, dec_b]))

            # the dimensions of the bias vectors are equivalent to the [1] index of the tuple
            biases['encode']['b%i' % i] = tf.Variable(tf.random_normal([enc_b]))
            biases['decode']['b%i' % i] = tf.Variable(tf.random_normal([dec_b]))

    return weights, biases



class AutoEncoder(BaseEstimator, TransformerMixin):
    """


    Parameters
    ----------
    activation_function : str or callable, optional (default='relu')
        The activation function. If str, should be one of PERMITTED_ACTIVATIONS. If
        callable, should be contained in the ``tensorflow.nn`` module.

    """

    def __init__(self, activation_function='relu', learning_rate=0.01, n_epochs=20, batch_size=256,
                 n_hidden=None, compression_ratio=0.6):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.compression_ratio = compression_ratio

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        # validate X, then make it into TF structure
        X = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        n_samples, n_features = X.shape

        # assign X to tf
        X = tf.placeholder('float', [None, n_features])

        # validate floats
        _validate_float(self.compression_ratio, 'compression_ratio')
        _validate_float(self.learning_rate, 'learning_rate')

        # validate activation, set it:
        if isinstance(self.activation_function, six.string_types):
            if self.activation_function not in PERMITTED_ACTIVATIONS:
                raise ValueError('Permitted activation functions: %r' % PERMITTED_ACTIVATIONS)
            self.activation_function = PERMITTED_ACTIVATIONS[self.activation_function]
        # if it's a callable just let it pass
        elif not hasattr(self.activation_function, '__call__'):
            raise ValueError('Activation function must be a string or a callable')

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        self.w_, self.b_ = _weights_biases_from_hidden(self.n_hidden, n_features, self.compression_ratio)

        # actually fit the model, now
        encode_op = self._encode(X)
        decode_op = self._decode(encode_op)
        y_true, y_pred = X, decode_op

        # get loss and optimizer, minimize MSE
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)

        # initialize global vars for tf, then run it
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int(n_samples / self.batch_size)

            # training cycle:
            for epoch in range(self.n_epochs):
                # loop batches
                for i in range(total_batch):
                    # generate batches...

    def _encode(self, X):
        return self._encode_decode_from_stages(X, 'encode')

    def _decode(self, X):
        return self._encode_decode_from_stages(X, 'decode')

    def _encode_decode_from_stages(self, X, key):
        w, b = self.w_[key], self.b_[key]
        plan = sorted(w.keys())  # key is either 'encode' or 'decode'

        next_layer = None
        for stage in plan:
            weights, biases = w[stage], b[stage]

            # if it's the first hidden layer, the input tensor is X, otherwise the last layer
            tensor = X if next_layer is None else next_layer
            next_layer = self.activation_function(tf.add(tf.matmul(tensor, weights), biases))

        return next_layer

    def transform(self, X):
        check_is_fitted(self, 'w_')
        X = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        return self._encode(X)

    def inverse_transform(self, X):
        check_is_fitted(self, 'w_')
        X = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        return self._decode(X)
