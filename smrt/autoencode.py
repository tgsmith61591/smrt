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


def _relu(x):
    return np.maximum(x, 0, x)


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


LOCAL_ACTIVATIONS = {
    'relu': _relu,
    'sigmoid': _sigmoid,
    'tanh': np.tanh
}

PERMITTED_ACTIVATIONS = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}

PERMITTED_OPTIMIZERS = {
    'rms_prop': tf.train.RMSPropOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
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


def _validate_float(instance, name, upper_bound=1.):
    val = getattr(instance, name)
    try:
        res = float(val)
        assert 0 < val < upper_bound  # the assertion error will be caught and valueerror raised
    except:
        raise ValueError('%s must be a float between 0 and %.3f' % (name, upper_bound))
    return res


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
    activation_function : str, optional (default='sigmoid')
        The activation function. Should be one of PERMITTED_ACTIVATIONS.

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs and if
        ``early_stopping`` is enabled.

    batch_size : int, optional (default=128)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    n_hidden : int or list, optional (default=None)
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

    min_change : float, optional (default=1e-6)
        An early stopping criterion. If the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early (``early_stopping``
        must also be enabled for this feature to work).

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=5)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    learning_function : str, optional (default='rms_prop')
        The optimizing function for training. Default is ``'rms_prop'``, which will use
        the ``tf.train.RMSPropOptimizer``. Can be one of {``'rms_prop'``, ``'sgd'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    initial_weight_stddev : float, optional (default=0.001)
        The standard deviation of the initial random, normally distributed weights.

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

    def __init__(self, activation_function='sigmoid', learning_rate=0.05, n_epochs=20, batch_size=128,
                 n_hidden=None, compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5,
                 learning_function='rms_prop', early_stopping=False, initial_weight_stddev=0.001,
                 seed=42):

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

    def fit(self, X, y=None, **kwargs):
        # validate X, then make it into TF structure
        X_original = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        n_samples, n_features = X_original.shape

        # assign X to tf as a placeholder for now
        X = tf.placeholder(tf.float32, [None, n_features])

        # validate floats and other params...
        self.compression_ratio = _validate_float(self, 'compression_ratio', upper_bound=np.inf)
        self.min_change = _validate_float(self, 'min_change')
        self.learning_rate = _validate_float(self, 'learning_rate')
        self.seed = _validate_positive_integer(self, 'seed')
        self.verbose = _validate_positive_integer(self, 'verbose')
        self.display_step = _validate_positive_integer(self, 'display_step')
        self.n_epochs = _validate_positive_integer(self, 'n_epochs')
        sd = _validate_float(self, 'initial_weight_stddev', upper_bound=np.inf)

        # validate activation, set it:
        if isinstance(self.activation_function, six.string_types):
            if self.activation_function not in PERMITTED_ACTIVATIONS:
                raise ValueError('Permitted activation functions: %r' % PERMITTED_ACTIVATIONS)
        else:
            raise TypeError('Activation function must be a string')
        activation = PERMITTED_ACTIVATIONS[self.activation_function]  # make it local

        # validation optimization function:
        if isinstance(self.learning_function, six.string_types):
            if self.learning_function not in PERMITTED_OPTIMIZERS:
                raise ValueError('Permitted learning functions: %r' % PERMITTED_OPTIMIZERS)
        else:
            raise TypeError('Learning function must be a string')
        learning_function = PERMITTED_OPTIMIZERS[self.learning_function]

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        n_hidden = self.n_hidden
        seed = self.seed
        if n_hidden is None:
            n_hidden = max(1, int(self.compression_ratio * n_features))

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

        # both weights and biases are dicts with two stages each:
        #   * encode: learn the compressed feature space
        #   * decode: re-map the compressed feature space to the original space
        weights, biases = {}, {}
        for i, t in enumerate(encode_dimensions):
            enc_a, enc_b = t
            dec_a, dec_b = decode_dimensions[i]

            # tensor flow random variable generation will generate all the same matrices if the seed is the same...
            # let's still find a way to keep predictability (for testing) but also maximize shuffle:
            s1, s2 = seed ^ i, seed * i
            s3, s4 = int(17 * (s1 ** 2) / 31), int(s2 ** 2 / 19 + 43)

            # initialize weights for encode/decode layer. While the encode layers progress through the
            # zipped dimensions, the decode layer steps back up to eventually mapping back to the input space
            weights['h%i_encode' % i] = tf.Variable(tf.random_normal(shape=[enc_a, enc_b], stddev=sd, seed=s1))
            weights['h%i_decode' % i] = tf.Variable(tf.random_normal(shape=[dec_a, dec_b], stddev=sd, seed=s2))

            # the dimensions of the bias vectors are equivalent to the [1] index of the tuple
            biases['b%i_encode' % i] = tf.Variable(tf.random_normal(shape=[enc_b], stddev=sd, seed=s3))
            biases['b%i_decode' % i] = tf.Variable(tf.random_normal(shape=[dec_b], stddev=sd, seed=s4))

        def _encode_decode_from_stages(x, key):
            n_layers = len(weights) // 2  # twice as many layers, since there's the decode layer(s) too..

            result_layer = None
            for i in range(n_layers):
                wts, bss = weights['h%i_%s' % (i, key)], biases['b%i_%s' % (i, key)]

                # if it's the first hidden layer, the input tensor is X, otherwise the last layer
                tensor = x if result_layer is None else result_layer
                result_layer = activation(tf.add(tf.matmul(tensor, wts), bss))

            return result_layer

        def _encode(x):
            return _encode_decode_from_stages(x, 'encode')

        def _decode(x):
            return _encode_decode_from_stages(x, 'decode')

        # actually fit the model, now
        encoder = _encode(X)
        decoder = _decode(encoder)
        y_pred, y_true = decoder, X

        # get loss and optimizer, minimize MSE
        cost_function = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = learning_function(self.learning_rate).minimize(cost_function)

        # initialize global vars for tf
        init = tf.global_variables_initializer()

        # run the training session
        with tf.Session() as sess:
            sess.run(init)
            epoch_times = []
            last_cost = None

            # generate the batches in a generator from sklearn, but store
            # in a list so we don't have to re-gen (since the generator will be
            # empty by the end of the epoch)
            batches = list(gen_batches(n_samples, self.batch_size))

            # print(sess.run(weights['h0_encode']))
            # print(sess.run(weights['h0_decode']))
            # print(sess.run(biases['b0_encode']))
            # print(sess.run(biases['b0_decode']))

            # training cycle. For each epoch
            for epoch in range(self.n_epochs):

                # this needs to be re-done for each epoch, since it's a generator and will
                # otherwise be exhausted after the first epoch...
                start_time = time.time()

                # loop batches
                for batch in batches:

                    # extract the chunk given the slice, and assert it's not length 0 or anything weird...
                    chunk = X_original[batch, :]
                    m, _ = chunk.shape
                    assert m <= self.batch_size and m != 0  # sanity check

                    # train the batch - runs optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost_function], feed_dict={X: chunk})

                # add the time to the times array to compute average later
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)

                # Display logs if display_step and verbose
                if epoch % self.display_step == 0 and self.verbose > 1:
                    print('Epoch: %i, cost=%.6f, time=%.4f (sec)' % (epoch + 1, c, epoch_time))

                # update last_cost, and if it meets the stopping criteria, break.
                # we only do this if we've enabled early_stopping, though.
                if self.early_stopping:
                    if last_cost is None:
                        last_cost = c
                    else:
                        delta = abs(last_cost - c)
                        if delta <= self.min_change:
                            if self.verbose:
                                print('Convergence reached at epoch %i, stopping early' % epoch)
                            break

            # set instance vars
            self.weights_, self.biases_ = sess.run(weights), sess.run(biases)

        if self.verbose:
            print('Optimization complete after %i epoch(s). Average epoch time: %.4f seconds'
                  % (len(epoch_times), np.average(epoch_times)))
        return self

    def _apply_transformation(self, X, key):
        check_is_fitted(self, 'weights_')

        # create the steps:
        n_layers = len(self.weights_) // 2  # twice as many layers, since there's the decode layer too..
        activation = LOCAL_ACTIVATIONS[self.activation_function]

        result_layer = None
        for i in range(n_layers):
            wts, bss = self.weights_['h%i_%s' % (i, key)], self.biases_['b%i_%s' % (i, key)]
            tensor = X if result_layer is None else result_layer
            result_layer = activation(np.dot(tensor, wts) + bss)

        return result_layer

    def transform(self, X):
        return self._apply_transformation(X, key='encode')

    def inverse_transform(self, X):
        return self._apply_transformation(X, key='decode')

    def feed_forward(self, X):
        return self.inverse_transform(self.transform(X))
