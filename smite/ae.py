# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The autoencoder

from __future__ import division, absolute_import, division
from .utils import get_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.externals import six
from numpy.random import RandomState
import numpy as np
import theano as th
from theano import tensor
from theano.tensor import nnet as nn
import uuid
import time

__all__ [
    'AutoEncoder'
]

DTYPE = th.config.floatX
PERMITTED_ACTIVATIONS = {
    nn.relu, nn.sigmoid
}


class AutoEncoder(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    activation_function

    output_function

    borrow : bool, optional (default=True)
        It is a safe practice (and a good idea) to use ``borrow=True`` in a ``shared``
        variable constructor when the shared variable stands for a large object 
        (in terms of memory footprint) and you do not want to create copies of it in memory.
        Since the ``AutoEncoder`` shares ``X`` in the ``fit`` method, by default ``borrow``
        is True. If you intend to run this on a GPU server, it is recommended you set ``borrow``
        to True.

    hidden_size : int or list, optional (default=None)
        The number of neurons in the hidden layer. If the default (None) is used, ``hidden_size``
        will amount to ``0.6 * n_features`` so the network is forced to learn a compressed feature 
        space.

    n_epochs : int, optional (default=100)

    mini_batch_size : int, optional (default=1)

    learning_rate : float, optional (default=0.1)
        The rate at which the autoencoder will learn. Default is 0.1.

    weight_init_scalar_low

    weight_init_scalar_high

    weight_init_scaling_factor

    verbose

    random_state
    """

    def __init__(self, activation_function=nn.relu, output_function=nn.relu, borrow=True,
                 hidden_size=None, n_epochs=100, mini_batch_size=1, learning_rate=0.1, 
                 weight_init_scalar_low=-4., weight_init_scalar_high=4., 
                 weight_init_scaling_factor=6., verbose=0, random_state=None):

        self.activation = activation
        self.output_function = output_function
        self.borrow = borrow
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.weight_init_scalar_low = weight_init_scalar_low
        self.weight_init_scalar_high = weight_init_scalar_high
        self.weight_init_scaling_factor = weight_init_scaling_factor
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        # get names
        model_id = str(uuid.uuid1())
        X_nm, W_nm, b1_nm, b2_nm = tuple('ae-%s-%s' % (model_id, c) for c in ('X', 'W', 'b1', 'b2'))

        # validate X, share with Theano
        X = th.shared(name=X_nm, 
                      value=check_array(X, accept_sparse=False, 
                                        force_all_finite=True, 
                                        ensure_2d=True),
                      dtype=DTYPE, borrow=self.borrow)
        n_samples, n_features = X.shape

        # validate params
        if not all(func in PERMITTED_ACTIVATIONS for func in (self.activation_function, self.output_function)):
            raise ValueError('Permitted activation/output functions: %r' % PERMITTED_ACTIVATIONS)

        random_state = self.random_state
        if not random_state:
            random_state = RandomState()  # default initialization
        elif not isinstance(random_state, RandomState):
            random_state = RandomState(random_state)  # in case an int is passed

        # if hidden_size is None, make it 0.6 * n_features
        hidden_size = self.hidden_size
        if not hidden_size:
            hidden_size = max(int(0.6 * n_features), 1)
        if hidden_size < 1 or not isinstance(int, np.int, np.int32, np.int64):
            raise ValueError('hidden_size must be a positive integer')

        # don't validate the weight hi/lo scalars; let them fail in theano if needed
        wisf = self.weight_init_scaling_factor
        if not wisf >= 0.:
            raise ValueError('weight_init_scaling_factor must be a positive float')

        # initialize weights, share them to theano
        initial_weights = np.asarray(
            random_state.uniform(low=self.weight_init_scalar_low * np.sqrt(wisf / (hidden_size + n_features)),
                                 high=self.weight_init_scalar_high * np.sqrt(wisf / (hidden_size + n_features)),
                                 size=(n_features, hidden_size)), 
            dtype=DTYPE)
        weights = th.shared(value=initial_weights, name=W_nm, borrow=self.borrow)
        b1 = th.shared(name=b1_nm, value=np.zeros(shape=(hidden_size,), dtype=DTYPE), borrow=self.borrow)
        b2 = th.shared(name=b2_nm, value=np.zeros(shape=(n_features,), dtype=DTYPE), borrow=self.borrow)

        # do fit
        index = tensor.lscalar()
        x = tensor.matrix('x')
        params = [weights, b1, b2]
        hidden = self.activation_function(tensor.dot(x, weights) + b1)
        output = self.output_function(tensor.dot(hidden, tensor.transpose(weights)) + b2)

        # loss function - cross entropy
        cost = (-tensor.sum(x * tensor.log(output) + (1 - x) * tensor.log(1 - output), axis=1)).mean()

        # return gradient with respect to weights, b1, b2
        gparams = tensor.grad(cost, params)

        # build list for updates
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

        train = th.function(inputs=[index],
                            outputs=[cost],
                            updates=updates,
                            givens={
                                x: X[index : index + self.mini_batch_size, :]
                            })

        start_time = time.clock()
        for epoch in range(self.n_epochs):
            if self.verbose > 0:
                print('Epoch: %i' % epoch)
            for row in range(0, n_samples, mini_batch_size):
                train(row)

        end_time = time.clock()
        if self.verbose > 0:
            print('Average epoch time: %.3f' % ((end_time - start_time) / self.n_epochs))

    def get_deep_features(self):

    def transform(self, X):

    def reconstruct(self, X):

