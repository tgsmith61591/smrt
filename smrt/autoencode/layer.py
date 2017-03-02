# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Layers for the autoencoder(s)

from __future__ import print_function, absolute_import, division
from sklearn.base import BaseEstimator
from sklearn.externals import six
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from ..utils import overrides, get_random_state, next_seed
from .base import _validate_positive_integer, _validate_float
from . import base

__all__ = [
    'GaussianDenseLayer',
    'SymmetricalAutoEncoderTopography',
    'SymmetricalVAETopography',
    'XavierDenseLayer'
]

DTYPE = base.DTYPE


def _chain_layers(layers, tensor):
    result = tensor
    for layer in layers:
        result = layer.feed_forward(result)
    return result


class _BaseSymmetricalTopography(BaseEstimator):
    def __init__(self, X_placeholder, n_hidden, input_shape, activation, layer_type,
                 dropout, scope, bias_strategy, random_state, **kwargs):
        # validate layer dims
        if not isinstance(n_hidden, list):
            if not isinstance(n_hidden, (int, np.int)):
                raise ValueError('n_hidden must be an int or list')
            n_hidden = [n_hidden]

        # validate layer types
        if layer_type not in PERMITTED_LAYER_TYPES:
            raise ValueError('layer_type must be one of %r' % list(PERMITTED_LAYER_TYPES.keys()))
        LayerClass = PERMITTED_LAYER_TYPES[layer_type]

        # validate random state
        random_state = get_random_state(random_state)

        # set encode/decode
        self.encode, self.decode = self._initialize_layers(X_placeholder=X_placeholder,
                                                           n_hidden=n_hidden,
                                                           input_shape=input_shape,
                                                           LayerClass=LayerClass,
                                                           activation=activation,
                                                           dropout=dropout,
                                                           scope=scope,
                                                           bias_strategy=bias_strategy,
                                                           random_state=random_state,
                                                           **kwargs)

    @abstractmethod
    def _initialize_layers(self, X_placeholder, n_hidden, input_shape, LayerClass, activation,
                           dropout, scope, bias_strategy, random_state, **kwargs):
        """Initialize all the layers"""
        # We know it's a list. There will be two times as many layers as the length of n_hidden:
        # n_hidden * encode layer, and n_hidden * decode layer. Since the dimensions are
        # piped into one another, stagger them (zipped with a lag), and then reverse for
        # the decode layer.

    def get_weights_biases(self):
        return list(zip(*[(layer.w_, layer.b_) for layer in (self.encode_layers_ + self.decode_layers_)]))


class SymmetricalAutoEncoderTopography(_BaseSymmetricalTopography):
    """The architecture of the neural network. This connects layers together given
    the ``layer_type``.

    Parameters
    ----------
    X_placeholder : TensorFlow placeholder
        The placeholder for ``X``.

    n_hidden : int or list
        The shape of the hidden layers. This will be reflected, i.e., if the provided value
        is ``[100, 50]``, the full topography will be ``[100, 50, 100]``

    input_shape : int
        The number of neurons in the input layer.

    activation : callable
        The activation function.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    scope : str, optional (default='dense_layer')
        The scope used for TensorFlow variable sharing.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random TensorFlow variables in weight initialization.


    Attributes
    ----------
    encode_layers_ : list
        The encode layers

    decode_layers_ : list
        The decode layers
    """
    def __init__(self, X_placeholder, n_hidden, input_shape, activation, layer_type='xavier', dropout=1.,
                 scope='dense_layer', bias_strategy='zeros', random_state=None):
        super(SymmetricalAutoEncoderTopography, self).__init__(X_placeholder=X_placeholder,
                                                               n_hidden=n_hidden,
                                                               input_shape=input_shape,
                                                               activation=activation,
                                                               layer_type=layer_type,
                                                               dropout=dropout,
                                                               scope=scope,
                                                               bias_strategy=bias_strategy,
                                                               random_state=random_state)

    @overrides(_BaseSymmetricalTopography)
    def _initialize_layers(self, X_placeholder, n_hidden, input_shape, LayerClass, activation,
                           dropout, scope, bias_strategy, random_state, **kwargs):
        n_hidden.insert(0, input_shape)
        encode_dimensions = list(zip(n_hidden[:-1], n_hidden[1:]))
        decode_dimensions = [(v, k) for k, v in reversed(encode_dimensions)]  # pyramid back to n_features

        # this procedure creates a symmetrical topography
        encode_layers, decode_layers = [], []
        n_layers = len(encode_dimensions)
        for i in range(n_layers):
            encode_fan = encode_dimensions[i]
            decode_fan = decode_dimensions[i]

            # build them simultaneously without duplicated code
            enc_dec_layers = tuple(
                LayerClass(fan_in=dims[0], fan_out=dims[1],
                           activation=activation, dropout=dropout,
                           bias_strategy=bias_strategy, scope=scope,
                           seed=next_seed(random_state))
                for dims in (encode_fan, decode_fan)
            )

            # split the tuple
            encode_layers.append(enc_dec_layers[0])
            decode_layers.append(enc_dec_layers[1])

        # the encode/decode operations
        encoder = _chain_layers(encode_layers, X_placeholder)
        decoder = _chain_layers(decode_layers, encoder)
        return encoder, decoder


class SymmetricalVAETopography(_BaseSymmetricalTopography):
    """The architecture of the neural network. This connects layers together given
    the ``layer_type``.

    Parameters
    ----------
    X_placeholder : TensorFlow placeholder
        The placeholder for ``X``.

    n_hidden : int or list
        The shape of the hidden layers. This will be reflected, i.e., if the provided value
        is ``[100, 50]``, the full topography will be ``[100, 50, 100]``

    input_shape : int
        The number of neurons in the input layer.

    activation : callable
        The activation function.

    n_latent_factors : int or float
        The size of the latent factor layer learned by the ``VariationalAutoEncoder``

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    scope : str, optional (default='dense_layer')
        The scope used for TensorFlow variable sharing.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random TensorFlow variables in weight initialization.


    Attributes
    ----------
    encode_layers_ : list
        The encode layers

    decode_layers_ : list
        The decode layers
    """
    def __init__(self, X_placeholder, n_hidden, input_shape, activation, n_latent_factors, layer_type='xavier',
                 dropout=1., scope='dense_layer', bias_strategy='zeros', random_state=None):

        # validate n_latent_factors
        self.n_latent_factors = n_latent_factors
        if isinstance(self.n_latent_factors, (int, np.int)):
            self.n_latent_factors = _validate_positive_integer(self, 'n_latent_factors')
        else:
            # otherwise, if it's a float, we are going to compress the n_features by that amount
            if isinstance(self.n_latent_factors, (float, np.float)):
                compress = _validate_float(self, 'n_latent_factors', 1.0)
                self.n_latent_factors = max(2, int(round(compress * input_shape)))
            else:
                raise TypeError('n_latent_factors must be an int or a float')

        # python lets us call the super constructor anywhere in the constructor
        super(SymmetricalVAETopography, self).__init__(X_placeholder=X_placeholder,
                                                       n_hidden=n_hidden,
                                                       input_shape=input_shape,
                                                       activation=activation,
                                                       layer_type=layer_type,
                                                       dropout=dropout,
                                                       scope=scope,
                                                       bias_strategy=bias_strategy,
                                                       random_state=random_state,
                                                       **{'n_latent_factors': self.n_latent_factors})

    def _gaussian_sample(self, mu, log_sigma, random_state):
        with tf.name_scope('gaussian_sample'):
            epsilon = tf.random_normal(tf.shape(log_sigma), name='epsilon', seed=next_seed(random_state), dtype=DTYPE)
            return tf.add(mu, tf.multiply(epsilon, tf.exp(log_sigma)))  # N(mu, I * sigma**2)

    @overrides(_BaseSymmetricalTopography)
    def _initialize_layers(self, X_placeholder, n_hidden, input_shape, LayerClass, activation, dropout,
                           scope, bias_strategy, random_state, **kwargs):
        n_latent = kwargs.pop('n_latent_factors')  # will be there because we're injecting it in the super constructor

        # AE makes it easy to string layers together, but the VAE is a bit more
        # complex. So we'll use the _chain method that will string layers together
        # I.e., _chain([layer_1, layer_2]) -> layer_2(layer_1(x))

        # inject input_shape like in AE
        n_hidden.insert(0, input_shape)
        encode_dimensions = list(zip(n_hidden[:-1], n_hidden[1:]))
        decode_dimensions = [(v, k) for k, v in reversed(encode_dimensions)]

        encoding_layers = [
            LayerClass(fan_in=fan_in, fan_out=fan_out,
                       activation=activation, dropout=dropout,
                       bias_strategy=bias_strategy, scope=scope,
                       seed=next_seed(random_state))
            for fan_in, fan_out in encode_dimensions
        ]

        # chain in reverse:
        encode = _chain_layers(encoding_layers, X_placeholder)

        # add the latent distribution ("hidden code")
        # z ~ N(z_mean, exp(z_log_sigma) ** 2)
        z_mean, z_log_sigma = tuple(
            _chain_layers(LayerClass(fan_in=n_hidden[-1], fan_out=n_latent, activation=activation,
                                     dropout=dropout, bias_strategy=bias_strategy, scope=scp,
                                     seed=next_seed(random_state)),
                          encode)
            for scp in ('z_mean', 'z_log_sigma')
        )

        # kingma & welling: only 1 draw necessary
        z = self._gaussian_sample(z_mean, z_log_sigma)

        # define decode layers
        decoding_layers = [
            LayerClass(fan_in=fan_in, fan_out=fan_out,
                       activation=activation, dropout=dropout,
                       bias_strategy=bias_strategy, scope=scope,
                       seed=next_seed(random_state))
            for fan_in, fan_out in decode_dimensions
        ]

        decode = _chain_layers(decoding_layers, z)

        # set some internals...
        self.z_mean_, self.z_log_sigma_, self.z_ = z_mean, z_log_sigma, z

        return encode, decode


class _BaseDenseLayer(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base dense layer"""
    def __init__(self, fan_in, fan_out, activation, dropout, scope, bias_strategy, seed):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.dropout = tf.placeholder_with_default(dropout, shape=[], name='dropout')  # make placeholder
        self.scope = scope
        self.seed = seed

        # validate strategy
        if bias_strategy not in PERMITTED_BIAS_STRATEGIES:
            raise ValueError("bias_strategy must be one of %r" % list(PERMITTED_BIAS_STRATEGIES.keys()))
        self.bias_strategy = PERMITTED_BIAS_STRATEGIES[bias_strategy]

        # initialize
        w, self.b_ = self._initialize_weights_biases()

        # add the dropout term
        self.w_ = tf.nn.dropout(w, self.dropout)

    def feed_forward(self, tensor):
        with tf.name_scope(self.scope):
            return self.activation(tf.add(tf.matmul(tensor, self.w_), self.b_))

    @abstractmethod
    def _initialize_weights_biases(self):
        """Initialize based on which type"""


class GaussianDenseLayer(_BaseDenseLayer):
    """A fully connected layer of neurons initialized via random normal distributions.

    Parameters
    ----------
    fan_in : int
        The dimension of the input.

    fan_out : int
        The dimension of the output.

    activation : callable
        The activation function.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    scope : str, optional (default='dense_layer')
        The scope used for TensorFlow variable sharing.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    seed : int, optional (default=42)
        The seed for random variable generation.


    References
    ----------
    [1] Based on code at https://github.com/fastforwardlabs/vae-tf
    """
    def __init__(self, fan_in, fan_out, activation, dropout=1., scope='dense_layer', bias_strategy='zeros', seed=42):
        super(GaussianDenseLayer, self).__init__(fan_in=fan_in, fan_out=fan_out, activation=activation,
                                                 dropout=dropout, scope=scope, bias_strategy=bias_strategy,
                                                 seed=seed)

    @overrides(_BaseDenseLayer)
    def _initialize_weights_biases(self):
        """Initialize weights in a normalized sense (adaptation of Xavier initialization)"""
        sd = tf.cast((2 / self.fan_in) ** 0.5, DTYPE)

        initial_w = tf.random_normal([self.fan_in, self.fan_out], stddev=sd, seed=self.seed, dtype=DTYPE)
        initial_b = self.bias_strategy([self.fan_out], dtype=DTYPE)

        return (tf.Variable(initial_w, trainable=True, name='weights'),
                tf.Variable(initial_b, trainable=True, name='biases'))


class XavierDenseLayer(_BaseDenseLayer):
    """A fully connected layer of neurons initialized via Xavier initialization distributions.

    Parameters
    ----------
    fan_in : int
        The dimension of the input.

    fan_out : int
        The dimension of the output.

    activation : callable
        The activation function.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    scope : str, optional (default='dense_layer')
        The scope used for TensorFlow variable sharing.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    seed : int, optional (default=42)
        The seed for random variable generation.


    References
    ----------
    [1] Based on code at https://github.com/fastforwardlabs/vae-tf
    """
    def __init__(self, fan_in, fan_out, activation, dropout=1., scope='dense_layer', bias_strategy='zeros', seed=42):
        super(XavierDenseLayer, self).__init__(fan_in=fan_in, fan_out=fan_out, activation=activation,
                                               dropout=dropout, scope=scope, seed=seed,
                                               bias_strategy=bias_strategy)

    @overrides(_BaseDenseLayer)
    def _initialize_weights_biases(self):
        """Initialize weights via Xavier initialization"""
        low = -1. * np.sqrt(6.0 / (self.fan_in + self.fan_out))
        high = 1. * np.sqrt(6.0 / (self.fan_in + self.fan_out))

        initial_w = tf.random_uniform(shape=[self.fan_in, self.fan_out], minval=low,
                                      maxval=high, dtype=DTYPE, seed=self.seed)
        initial_b = self.bias_strategy([self.fan_out], dtype=DTYPE)

        return (tf.Variable(initial_w, trainable=True, name='weights'),
                tf.Variable(initial_b, trainable=True, name='biases'))

# these are strategy/type mappings for mapping a str to a callable
PERMITTED_LAYER_TYPES = {
    'gaussian': GaussianDenseLayer,
    'xavier': XavierDenseLayer,
}

PERMITTED_BIAS_STRATEGIES = {
    'ones': tf.ones,
    'zeros': tf.zeros
}
