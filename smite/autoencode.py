# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The autoencoder

from __future__ import division, absolute_import, division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_batches, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
import numpy as np
import tensorflow as tf
import sys
import time

__all__ = [
    'AutoEncoder'
]

if sys.version_info[0] >= 3:
    long = int

PERMITTED_ACTIVATIONS = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}


def _validate_positive_integer(instance, name):
    val = getattr(instance, name)
    try:
        assert not isinstance(val, (bool, np.bool))
        res = int(val)
        assert res >= 0
    except:
        raise ValueError('%s must be an int >= 0' % name)
    return res


def _validate_float(instance, name):
    val = getattr(instance, name)
    try:
        res = float(val)
        assert 0 < val < 1  # the assertion error will be caught and valueerror raised
    except:
        raise ValueError('%s must be a float between 0 and 1' % name)
    return res


def _weights_biases_from_hidden(n_hidden, n_features, compression, seed):
    """Internal method. Given the hidden layer structure, number of features,
    and compression factor, generate the initial weights and bias matrix.
    """
    if n_hidden is None:
        n_hidden = max(1, int(compression * n_features))

    if not isinstance(n_hidden, list):
        if not isinstance(n_hidden, (long, int, np.int)):
            raise ValueError('n_hidden must be an int or list')
        n_hidden = [n_hidden]

    # otherwise it's iterable. There will be two times as many layers as the length of n_hidden:
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
        weights['encode']['h%i' % i] = tf.Variable(tf.random_normal(shape=[enc_a, enc_b], seed=seed))
        weights['decode']['h%i' % i] = tf.Variable(tf.random_normal(shape=[dec_a, dec_b], seed=seed))

        # the dimensions of the bias vectors are equivalent to the [1] index of the tuple
        biases['encode']['b%i' % i] = tf.Variable(tf.random_normal(shape=[enc_b], seed=seed))
        biases['decode']['b%i' % i] = tf.Variable(tf.random_normal(shape=[dec_b], seed=seed))

    return weights, biases


class AutoEncoder(BaseEstimator, TransformerMixin):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."

    The ``AutoEncoder`` class, as it is intended in ``smrt``, is used to ultimately identify
    the more minority-class-phenotypical training examples to "jitter" and reconstruct as
    synthetic training set observations.


    Parameters
    ----------
    activation_function : str or callable, optional (default='relu')
        The activation function. If a str, it should be one of PERMITTED_ACTIVATIONS. If a
        callable, it should be an activation function contained in the ``tensorflow.nn`` module.

    learning_rate : float, optional (default=0.01)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``eps`` between epochs.

    batch_size : int, optional (default=256)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    n_hidden : int, list or tuple, optional (default=None)
        The hidden layer structure. If an int is provided, a single hidden layer is constructed,
        with ``n_hidden`` neurons. If ``n_hidden`` is an iterable, ``len(n_hidden)`` hidden layers
        are constructed, with as many neurons as correspond to each index, respectively. If no
        value is passed for ``n_hidden`` (default), the ``AutoEncoder`` defaults to a single hidden
        layer of ``compression_ratio * n_features`` in order to force the network to learn a compressed
        feature space.

    compression_ratio : float, optional (default=0.6)
        If no value is passed for ``n_hidden`` (default), the ``AutoEncoder`` defaults to a single hidden
        layer of ``compression_ratio * n_features`` in order to force the network to learn a compressed
        feature space. Default ``compression_ratio`` is 0.6.

    eps : float, optional (default=0.001)
        An early stopping criterion. If the delta between the last cost and the new cost
        is less than ``eps``, the network will stop fitting early.

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=1)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the weight and bias initialization.


    Notes
    -----
    This class is based loosely on an example located at:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py


    Attributes
    ----------
    w_ : dict
        The weights. There are two keys:
          * 'encode': maps to a sub-dictionary of encoding hidden layers, and their respective weights.
          * 'decode': maps to a sub-dictionary of decoding hidden layers, and their respective weights.

    b_ : dict
        The biases. There are two key:
          * 'encode': maps to a sub-dictionary of encoding bias vectors, and their respective values.
          * 'decode': maps to a sub-dictionary of decoding bias vectors, and their respective values.


    References
    ----------
    [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document
        recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
    """

    def __init__(self, activation_function='relu', learning_rate=0.01, n_epochs=20, batch_size=256,
                 n_hidden=None, compression_ratio=0.6, eps=0.001, verbose=0, display_step=1, seed=42):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.compression_ratio = compression_ratio
        self.eps = eps
        self.verbose = verbose
        self.display_step = display_step
        self.seed = seed

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        # validate X, then make it into TF structure
        X_original = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        n_samples, n_features = X_original.shape

        # assign X to tf as a placeholder for now
        X = tf.placeholder('float', [None, n_features])

        # validate floats and other params...
        self.compression_ratio = _validate_float(self, 'compression_ratio')
        self.eps = _validate_float(self, 'eps')
        self.learning_rate = _validate_float(self, 'learning_rate')
        self.seed = _validate_positive_integer(self, 'seed')
        self.verbose = _validate_positive_integer(self, 'verbose')
        self.display_step = _validate_positive_integer(self, 'display_step')

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
        self.w_, self.b_ = _weights_biases_from_hidden(self.n_hidden, n_features, self.compression_ratio, self.seed)

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
            batches = gen_batches(n_samples, self.batch_size)
            epoch_times = []
            last_cost = None

            # training cycle:
            for epoch in range(self.n_epochs):
                start_time = time.time()

                # loop batches
                for batch in batches:
                    _, c = sess.run([optimizer, cost], feed_dict={X: X_original[batch, :]})

                # add the time to the times array to compute average later
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)

                # Display logs if display_step and verbose
                if epoch % self.display_step == 0 and self.verbose:
                    print('Epoch: %i, cost=%.4f, time=%.3f (sec)' % (epoch + 1, c, epoch_time))

                # update cost
                if last_cost is None:
                    last_cost = c
                else:
                    delta = abs(last_cost - c)
                    if delta < self.eps:
                        if self.verbose:
                            print('Convergence reached at epoch %i, stopping early' % epoch)
                        break

        if self.verbose > 1:
            print('Optimization complete. Average epoch time: %.4f seconds' % (np.average(epoch_times)))
        return self

    def _encode(self, X):
        return self._encode_decode_from_stages(X, 'encode')

    def _decode(self, X):
        return self._encode_decode_from_stages(X, 'decode')

    def _encode_decode_from_stages(self, X, key):
        w, b = self.w_[key], self.b_[key]
        plan = zip(sorted(w.keys()), sorted(b.keys()))  # key is either 'encode' or 'decode'

        next_layer = None
        for stage in plan:
            w_stage, b_stage = stage
            weights, biases = w[w_stage], b[b_stage]

            # if it's the first hidden layer, the input tensor is X, otherwise the last layer
            tensor = X if next_layer is None else next_layer
            next_layer = self.activation_function(tf.add(tf.matmul(tensor, weights), biases))

        return next_layer

    def _apply_transformation_steps(self, X, *steps):
        check_is_fitted(self, 'w_')

        X_orig = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        X = tf.placeholder('float', [None, X_orig.shape[1]])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            # do the steps all in one batch, because scoring is cheap:
            result = None
            for step in steps:
                if result is None:
                    result = step(X)
                else:
                    result = step(result)

            return sess.run(result, feed_dict={X: X_orig})

    def transform(self, X):
        return self._apply_transformation_steps(X, *(self._encode,))

    def inverse_transform(self, X):
        return self._apply_transformation_steps(X, *(self._decode,))

    def encode_and_reconstruct(self, X):
        return self._apply_transformation_steps(X, *(self._encode, self._decode))
