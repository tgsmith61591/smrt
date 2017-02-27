# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The autoencoder(s)

from __future__ import division, absolute_import, division
import time
import numpy as np
import tensorflow as tf
from sklearn.externals import six
from sklearn.utils import gen_batches, check_array
from sklearn.utils.validation import check_is_fitted
from abc import ABCMeta, abstractmethod

from . import _ae_utils as aeutil
from ._ae_utils import xavier_initialization
from . import base
from .base import BaseAutoEncoder, ReconstructiveMixin, GenerativeMixin, _validate_float, _validate_positive_integer
from ..utils import overrides

__all__ = [
    'AutoEncoder',
    'VariationalAutoEncoder'
]

DTYPE = base.DTYPE

# this dict maps all the supported activation functions to the tensorflow
# equivalent functions for the session model training. It also maps all the
# supported activation functions to the local variants for offline scoring operations.
PERMITTED_ACTIVATIONS = {
    'relu': {
        'tf': tf.nn.relu,
        'local': aeutil.relu
    },

    'sigmoid': {
        'tf': tf.nn.sigmoid,
        'local': aeutil.sigmoid
    },

    'tanh': {
        'tf': tf.nn.tanh,
        'local': np.tanh
    }
}

# this dict maps all the supported optimizer classes to the tensorflow
# callables. For now, only strings are supported for learning_function
PERMITTED_OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'rms_prop': tf.train.RMSPropOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}


def _validate_activation_optimization(activation_function, learning_function):
    if isinstance(activation_function, six.string_types):
        if activation_function not in PERMITTED_ACTIVATIONS:
            raise ValueError('Permitted activation functions: %r' % list(PERMITTED_ACTIVATIONS.keys()))
    else:
        raise TypeError('Activation function must be a string')
    activation = PERMITTED_ACTIVATIONS[activation_function]['tf']  # make it local

    # validation optimization function:
    if isinstance(learning_function, six.string_types):
        if learning_function not in PERMITTED_OPTIMIZERS:
            raise ValueError('Permitted learning functions: %r' % PERMITTED_OPTIMIZERS)
    else:
        raise TypeError('Learning function must be a string')
    learning_function = PERMITTED_OPTIMIZERS[learning_function]

    return activation, learning_function


def _initial_weights_biases(n_hidden, n_features, compression_ratio, sd, seed, xavier, latent_factors):
    if n_hidden is None:
        n_hidden = max(1, int(compression_ratio * n_features))

    if not isinstance(n_hidden, list):
        if not isinstance(n_hidden, (int, np.int)):
            raise ValueError('n_hidden must be an int or list')
        n_hidden = [n_hidden]

    # otherwise it's iterable. There will be two times as many layers as the length of n_hidden:
    # n_hidden * encode layer, and n_hidden * decode layer. Since the dimensions are
    # piped into one another, stagger them (zipped with a lag), and then reverse for
    # the decode layer. First, though, append n_features to n_hidden
    n_hidden.insert(0, n_features)
    encode_dimensions = list(zip(n_hidden[:-1], n_hidden[1:]))
    decode_dimensions = [(v, k) for k, v in reversed(encode_dimensions)]  # pyramid back to n_features

    # if we are using Xavier, call xavier_initialization. Otherwise we need an alternative
    def _initialize_variable(shape, _seed):
        # since it's a closure, it can access sd
        return tf.random_normal(shape=shape, stddev=sd, seed=_seed, dtype=DTYPE)
    init_func = _initialize_variable if not xavier else xavier_initialization

    # function to generate derivative seeds
    def _get_seeds(s, i):
        s1, s2 = seed ^ i, seed * i
        s3, s4 = int(17 * (s1 ** 2) / 31), int(s2 ** 2 / 19 + 43)
        return s1, s2, s3, s4

    # both weights and biases are dicts with two stages each:
    #   * encode: learn the compressed feature space
    #   * decode: re-map the compressed feature space to the original space
    weights, biases = {}, {}
    for i, t in enumerate(encode_dimensions):
        enc_a, enc_b = t
        dec_a, dec_b = decode_dimensions[i]

        # tensor flow random variable generation will generate all the same matrices if the seed is the same...
        # let's still find a way to keep predictability (for testing) but also maximize shuffle:
        s1, s2, s3, s4 = _get_seeds(seed, i)

        # initialize weights for encode/decode layer. While the encode layers progress through the
        # zipped dimensions, the decode layer steps back up to eventually mapping back to the input space
        weights['h%i_encode' % i] = tf.Variable(init_func([enc_a, enc_b], s1))
        weights['h%i_decode' % i] = tf.Variable(init_func([dec_a, dec_b], s2))

        # the dimensions of the bias vectors are equivalent to the [1] index of the tuple
        # the biases do not use Xavier initialization!
        biases['b%i_encode' % i] = tf.Variable(_initialize_variable([enc_b], s3))
        biases['b%i_decode' % i] = tf.Variable(_initialize_variable([dec_b], s4))

    # if we are doing variational, add the means and std layers
    if latent_factors is not None:
        # i has incremented at this point and is still in scope, so we can still use it to
        # create some seeds derivative of the original seed. Furthermore, enc_b, enc_a are still
        # in scope so we can use them as well.
        s1, s2, s3, s4 = _get_seeds(seed, i)
        s5, s6, s7, s8 = _get_seeds(seed, i + 1)  # kind of (very) hacky way to seed more

        # output of encode weights, bias
        weights['out_mean_encode'] = tf.Variable(init_func([enc_b, latent_factors], s1))
        weights['out_log_sigma_encode'] = tf.Variable(init_func([enc_b, latent_factors], s5))
        biases['out_mean_encode'] = tf.Variable(_initialize_variable([latent_factors], s2))
        biases['out_log_sigma_encode'] = tf.Variable(_initialize_variable([latent_factors], s6))

        # output of decode weights, bias
        weights['out_mean_decode'] = tf.Variable(init_func([dec_b, n_features], s3))
        weights['out_log_sigma_decode'] = tf.Variable(init_func([dec_b, n_features], s7))
        biases['out_mean_decode'] = tf.Variable(_initialize_variable([n_features], s4))
        biases['out_log_sigma_decode'] = tf.Variable(_initialize_variable([n_features], s8))

    return weights, biases


class _SymmetricAutoEncoder(BaseAutoEncoder):
    """Base class for the two provided autoencoders, which are architecturally symmetric
    in terms of hidden layers. The encode/decode functions will not work for non-symmetrically-architected
    neural networks.
    """
    def __init__(self, activation_function='sigmoid', learning_rate=0.05, n_epochs=20, batch_size=128,
                 n_hidden=None, compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5,
                 learning_function='rms_prop', early_stopping=False, initial_weight_stddev=0.001,
                 seed=42, xavier_init=True):

        super(_SymmetricAutoEncoder, self).__init__(activation_function=activation_function,
                                                    learning_rate=learning_rate, n_epochs=n_epochs,
                                                    batch_size=batch_size, n_hidden=n_hidden,
                                                    compression_ratio=compression_ratio, min_change=min_change,
                                                    verbose=verbose, display_step=display_step,
                                                    learning_function=learning_function,
                                                    early_stopping=early_stopping,
                                                    initial_weight_stddev=initial_weight_stddev, seed=seed,
                                                    xavier_init=xavier_init)

    @overrides(BaseAutoEncoder)
    def _encoding_function(self, x, weights, biases, activation):
        return self._encode_decode_from_stages(x, weights, biases, activation, 'encode')

    @overrides(BaseAutoEncoder)
    def _decoding_function(self, x, weights, biases, activation):
        return self._encode_decode_from_stages(x, weights, biases, activation, 'decode')

    @abstractmethod
    def _encode_decode_from_stages(self, x, weights, biases, activation, key):
        """Encode or decode given the provided key ('encode' or 'decode'). Since this is
        only for symmetrically-designed encoders, it basically traverses up one side or the
        other.
        """

    def _train(self, X, X_original, weights, biases, n_samples, cost_function, optimizer):
        # initialize global vars for tf - replace them if they already exist
        init = tf.global_variables_initializer()
        self.clean_session()
        sess = self.sess = tf.InteractiveSession()

        # run the training session
        sess.run(init)
        epoch_times = []
        last_cost = None

        # generate the batches in a generator from sklearn, but store
        # in a list so we don't have to re-gen (since the generator will be
        # empty by the end of the epoch)
        batches = list(gen_batches(n_samples, self.batch_size))

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
        self.train_cost_ = c

        if self.verbose:
            print('Optimization complete after %i epoch(s). Average epoch time: %.4f seconds'
                  % (len(epoch_times), np.average(epoch_times)))

        return self

class AutoEncoder(_SymmetricAutoEncoder, ReconstructiveMixin):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."

    The ``ReconstructiveAutoEncoder`` class learns to reconstruct its input, minizing the MSE between
    training examples and the reconstructed output thereof.


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
        the ``tf.train.RMSPropOptimizer``. Can be one of {``'rms_prop'``, ``'sgd'``, ``'adam'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    initial_weight_stddev : float, optional (default=0.001)
        The standard deviation of the initial random, normally distributed weights.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the weight and bias initialization.

    xavier_init : bool, optional (default=True)
        Whether to use Xavier's initialization, as referenced in [2].


    Notes
    -----
    This class is based loosely on an example located at:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py


    Attributes
    ----------
    weights_ : dict
        The weights. This a dictionary that keys encode and decode layers with the suffix 'encode' or 'decode'

    biases_ : dict
        The biases. This a dictionary that keys encode and decode layers with the suffix 'encode' or 'decode'

    train_cost_ : float
        The final cost as a result of the training procedure on the training examples.


    References
    ----------
    [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document
        recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

    [2] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self, activation_function='sigmoid', learning_rate=0.05, n_epochs=20, batch_size=128,
                 n_hidden=None, compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5,
                 learning_function='rms_prop', early_stopping=False, initial_weight_stddev=0.001,
                 seed=42, xavier_init=True):

        super(AutoEncoder, self).__init__(activation_function=activation_function,
                                          learning_rate=learning_rate, n_epochs=n_epochs,
                                          batch_size=batch_size, n_hidden=n_hidden,
                                          compression_ratio=compression_ratio, min_change=min_change,
                                          verbose=verbose, display_step=display_step,
                                          learning_function=learning_function,
                                          early_stopping=early_stopping,
                                          initial_weight_stddev=initial_weight_stddev, seed=seed,
                                          xavier_init=xavier_init)

    def fit(self, X, y=None, **kwargs):
        # validate X, then make it into TF structure
        X_original = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        n_samples, n_features = X_original.shape

        # assign X to tf as a placeholder for now
        self.X_placeholder = X = tf.placeholder(tf.float32, [None, n_features])

        # validate floats and other params...
        self._validate_for_fit()
        activation, learning_function = _validate_activation_optimization(self.activation_function,
                                                                          self.learning_function)

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        n_hidden = self.n_hidden
        seed = self.seed

        # initialize the weights
        weights, biases = _initial_weights_biases(n_hidden, n_features, self.compression_ratio,
                                                  self.initial_weight_stddev, seed, self.xavier_init,
                                                  None)  # no latent factors for this model

        # define the encoder, decoder functions
        encoder = self._encoding_function(X, weights, biases, activation)
        decoder = self._decoding_function(encoder, weights, biases, activation)
        y_pred, y_true = decoder, X

        # get loss and optimizer, minimize MSE
        cost_function = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = learning_function(self.learning_rate).minimize(cost_function)

        # do training
        return self._train(X, X_original, weights, biases, n_samples, cost_function, optimizer)

    # define encoder, decoder
    @overrides(_SymmetricAutoEncoder)
    def _encode_decode_from_stages(self, x, weights, biases, activation, key):
        n_layers = len(weights) // 2  # twice as many layers, since there's the decode layer(s) too..

        result_layer = None
        for i in range(n_layers):
            wts, bss = weights['h%i_%s' % (i, key)], biases['b%i_%s' % (i, key)]

            # if it's the first hidden layer, the input tensor is X, otherwise the last layer
            tensor = x if result_layer is None else result_layer
            result_layer = activation(tf.add(tf.matmul(tensor, wts), bss))

        return result_layer

    def _apply_transformation(self, X, key):
        check_is_fitted(self, 'weights_')

        # create the steps:
        n_layers = len(self.weights_) // 2  # twice as many layers, since there's the decode layer too..
        activation = PERMITTED_ACTIVATIONS[self.activation_function]['local']

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

    @overrides(ReconstructiveMixin)
    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))


class VariationalAutoEncoder(_SymmetricAutoEncoder, GenerativeMixin, ReconstructiveMixin):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."

    The ``ReconstructiveAutoEncoder`` class, as it is intended in ``smrt``, is used to ultimately identify
    the more minority-class-phenotypical training examples to "jitter" and reconstruct as
    synthetic training set observations. The auto-encoder only uses TensorFlow for the model :meth:``fit``,
    and retains the numpy arrays of model weights and biases for offline model scoring.


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
        the ``tf.train.RMSPropOptimizer``. Can be one of {``'rms_prop'``, ``'sgd'``, ``'adam'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    initial_weight_stddev : float, optional (default=0.001)
        The standard deviation of the initial random, normally distributed weights.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the weight and bias initialization.

    xavier_init : bool, optional (default=True)
        Whether to use Xavier's initialization, as referenced in [1].

    n_latent_factors : int or float, optional (default=None)

    eps : float, optional (default=1e-10)
        A small amount of noise to add to the loss to avoid a potential computation of
        ``log(0)``.


    Notes
    -----
    This class is based loosely on the following examples:
        * http://kvfrans.com/variational-autoencoders-explained/
        * http://jmetzen.github.io/2015-11-27/vae.html


    Attributes
    ----------
    weights_ : dict
        The weights. This a dictionary that keys encode and decode layers with the suffix 'encode' or 'decode'

    biases_ : dict
        The biases. This a dictionary that keys encode and decode layers with the suffix 'encode' or 'decode'

    train_cost_ : float
        The final cost as a result of the training procedure on the training examples.


    References
    ----------
    [1] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    [2] http://jmetzen.github.io/2015-11-27/vae.html
    """
    def __init__(self, activation_function='sigmoid', learning_rate=0.05, n_epochs=20, batch_size=128,
                 n_hidden=None, compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5,
                 learning_function='rms_prop', early_stopping=False, initial_weight_stddev=0.001,
                 seed=42, xavier_init=True, n_latent_factors=None, eps=1e-10):

        super(VariationalAutoEncoder, self).__init__(activation_function=activation_function,
                                                     learning_rate=learning_rate, n_epochs=n_epochs,
                                                     batch_size=batch_size, n_hidden=n_hidden,
                                                     compression_ratio=compression_ratio, min_change=min_change,
                                                     verbose=verbose, display_step=display_step,
                                                     learning_function=learning_function, early_stopping=early_stopping,
                                                     initial_weight_stddev=initial_weight_stddev, seed=seed,
                                                     xavier_init=xavier_init)

        # the number of latent factors to learn
        self.n_latent_factors = n_latent_factors

    def fit(self, X, y=None, **kwargs):
        # validate X, then make it into TF structure
        X_original = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True)
        n_samples, n_features = X_original.shape

        # assign X to tf as a placeholder for now
        self.X_placeholder = X = tf.placeholder(tf.float32, [None, n_features])

        # validate floats and other params...
        self._validate_for_fit()
        activation, learning_function = _validate_activation_optimization(self.activation_function,
                                                                          self.learning_function)

        # validate n_latent_factors
        if isinstance(self.n_latent_factors, (int, np.int)):
            n_latent_factors = _validate_positive_integer(self, 'n_latent_factors')
        else:
            # otherwise, if it's a float, we are going to compress the n_features by that amount
            if isinstance(self.n_latent_factors, (float, np.float)):
                compress = _validate_float(self, 'n_latent_factors', 1.0)
            else:
                compress = self.compression_ratio  # this is already validated
            n_latent_factors = max(2, int(round(compress * n_features)))

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        n_hidden = self.n_hidden
        seed = self.seed

        # initialize the weights
        weights, biases = _initial_weights_biases(n_hidden, n_features, self.compression_ratio,
                                                  self.initial_weight_stddev, seed, self.xavier_init,
                                                  n_latent_factors)

        # define the encoder, decoder functions
        z_mean, z_log_sigma_sq = self._encoding_function(X, weights, biases, activation)  # encode
        epsilon = tf.random_normal([self.batch_size, n_latent_factors], 0, 1, dtype=DTYPE, seed=seed)  # one sample
        z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), epsilon))
        X_reconstruction_mean = self._decoding_function(X, weights, biases, activation)  # decode

        # Create the loss function optimizer. This dual-part loss function is adapted from code found at [2]
        # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed
        #     Bernoulli distribution induced by the decoder in the data space). This can be interpreted as the number
        #     of "nats" required for reconstructing the input when the activation in latent is given.
        reconstruction_loss = -tf.reduce_sum(X * tf.log(self.eps + X_reconstruction_mean)
                                             + (1 - X) * tf.log(self.eps + 1 - X_reconstruction_mean), 1)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence between the distribution in
        #     latent space induced by the encoder on the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required for transmitting the the latent space
        #     distribution given the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
        cost_function = tf.reduce_mean(reconstruction_loss + latent_loss)
        optimizer = learning_function(self.learning_rate).minimize(cost_function)

        # actually do training
        return self._train(X, X_original, weights, biases, n_samples, cost_function, optimizer)

    # define encoder, decoder
    @overrides(_SymmetricAutoEncoder)
    def _encode_decode_from_stages(self, x, weights, biases, activation, key):
        n_layers = len(weights) // 2  # twice as many layers, since there's the decode layer(s) too..

        result_layer = None
        for i in range(n_layers):
            wts, bss = weights['h%i_%s' % (i, key)], biases['b%i_%s' % (i, key)]

            # if it's the first hidden layer, the input tensor is X, otherwise the last layer
            tensor = x if result_layer is None else result_layer
            result_layer = activation(tf.add(tf.matmul(tensor, wts), bss))

        # the variational autoencoder has a second step, which is to compute the means
        # and standard deviations of the layer output
        mean_lookup, sigma_lookup = 'out_mean_%s' % key, 'out_log_sigma_%s' % key
        z_mean = tf.add(tf.matmul(result_layer, weights[mean_lookup]), biases[mean_lookup])

        if key == 'encode':
            z_log_sigma_sq = tf.add(tf.matmul(result_layer, weights[sigma_lookup]), biases[sigma_lookup])
            return z_mean, z_log_sigma_sq
        else:
            return z_mean
