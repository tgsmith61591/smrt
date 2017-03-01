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
from ..utils import overrides, get_random_state
from . import base

__all__ = [
    'GaussianDenseLayer',
    'SymmetricalAutoEncoderTopography',
    'XavierDenseLayer'
]

DTYPE = base.DTYPE


class SymmetricalAutoEncoderTopography(BaseEstimator):
    """The architecture of the neural network. This connects layers together given
    the ``layer_type``.

    Parameters
    ----------
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

    random_state : int, ``np.random.RandomState`` or None
        The numpy random state for seeding random TensorFlow variables.


    Attributes
    ----------
    encode_ : list
        The encode layers

    decode_ : list
        The decode layers
    """
    def __init__(self, n_hidden, input_shape, activation, layer_type='xavier', dropout=1.,
                 scope='dense_layer', random_state=None):
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

        # otherwise it's a list. There will be two times as many layers as the length of n_hidden:
        # n_hidden * encode layer, and n_hidden * decode layer. Since the dimensions are
        # piped into one another, stagger them (zipped with a lag), and then reverse for
        # the decode layer. First, though, append n_features to n_hidden
        n_hidden.insert(0, input_shape)
        encode_dimensions = list(zip(n_hidden[:-1], n_hidden[1:]))
        decode_dimensions = [(v, k) for k, v in reversed(encode_dimensions)]  # pyramid back to n_features

        # this procedure creates a symmetrical topography
        encode, decode = [], []
        n_layers = len(encode_dimensions)
        for i in range(n_layers):
            encode_fan = encode_dimensions[i]
            decode_fan = decode_dimensions[i]

            # build them simultaneously without duplicated code
            enc_dec_layers = tuple(
                LayerClass(fan_in=dims[0], fan_out=dims[1],
                           activation=activation, dropout=dropout,
                           scope=scope, seed=random_state.randint(0, 1000))
                for dims in (encode_fan, decode_fan)
            )

            # split the tuple
            encode.append(enc_dec_layers[0])
            decode.append(enc_dec_layers[1])
            
        self.encode_ = encode
        self.decode_ = decode


class _BaseDenseLayer(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base dense layer"""
    def __init__(self, fan_in, fan_out, activation, dropout, scope, seed):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation = activation
        self.dropout = dropout
        self.scope = scope
        self.seed = seed

        # initialize
        self.w_, self.b_ = self._initialize_weights_biases()

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
    n_units

    activation

    dropout

    scope

    seed


    References
    ----------
    [1] Based on code at https://github.com/fastforwardlabs/vae-tf
    """
    def __init__(self, fan_in, fan_out, activation, dropout=1., scope='dense_layer', seed=42):
        super(GaussianDenseLayer, self).__init__(fan_in=fan_in, fan_out=fan_out, activation=activation,
                                                 dropout=dropout, scope=scope, seed=seed)

    @overrides(_BaseDenseLayer)
    def _initialize_weights_biases(self):
        """Initialize weights in a normalized sense (adaptation of Xavier initialization)"""
        sd = tf.cast((2 / self.fan_in) ** 0.5, DTYPE)

        initial_w = tf.random_normal([self.fan_in, self.fan_out], stddev=sd, seed=self.seed, dtype=DTYPE)
        initial_b = tf.zeros([self.fan_out], dtype=DTYPE)

        return (tf.Variable(initial_w, trainable=True, name='weights'),
                tf.Variable(initial_b, trainable=True, name='biases'))


class XavierDenseLayer(_BaseDenseLayer):
    """A fully connected layer of neurons initialized via Xavier initialization distributions.

    Parameters
    ----------
    n_units

    activation

    dropout

    scope

    seed


    References
    ----------
    [1] Based on code at https://github.com/fastforwardlabs/vae-tf
    """
    def __init__(self, fan_in, fan_out, activation, dropout=1., scope='dense_layer', seed=42):
        super(XavierDenseLayer, self).__init__(fan_in=fan_in, fan_out=fan_out, activation=activation,
                                                dropout=dropout, scope=scope, seed=seed)

    @overrides(_BaseDenseLayer)
    def _initialize_weights_biases(self):
        """Initialize weights via Xavier initialization"""
        low = -1. * np.sqrt(6.0 / (self.fan_in + self.fan_out))
        high = 1. * np.sqrt(6.0 / (self.fan_in + self.fan_out))

        initial_w = tf.random_uniform(shape=[self.fan_in, self.fan_out], minval=low,
                                      maxval=high, dtype=DTYPE, seed=self.seed)
        initial_b = tf.zeros([self.fan_out], dtype=DTYPE)

        return (tf.Variable(initial_w, trainable=True, name='weights'),
                tf.Variable(initial_b, trainable=True, name='biases'))

PERMITTED_LAYER_TYPES = {
    'gaussian': GaussianDenseLayer,
    'xavier': XavierDenseLayer,
}
