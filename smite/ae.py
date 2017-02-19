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
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T
from theano.tensor import nnet as nn
import uuid
import timeit
import sys

__all__ = [
    'DenoisingAutoEncoder'
]

if sys.version_info[0] >= 3:
    long = int
    NoneType = type(None)
else:
    from types import NoneType

DTYPE = th.config.floatX
PERMITTED_ACTIVATIONS = {
    nn.relu, nn.sigmoid
}


def initial_weight_matrix(n_hidden, n_visible, name, random_state=None, borrow=True):
    """Create the initial weight matrix. Credit goes to:
    http://deeplearning.net/tutorial/dA.html

    Parameters
    ----------
    # todo
    """
    random_state = get_random_state(random_state)

    # W is initialized with `initial_W` which is uniformly sampled
    # from -4 * sqrt(6. / (n_visible + n_hidden)) and
    # 4 * sqrt(6. / (n_hidden + n_visible)) the output of uniform if
    # converted using asarray to dtype
    # theano.config.floatX so that the code is runnable on GPU
    initial_W = np.asarray(
        random_state.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)
        ),
        dtype=th.config.floatX
    )

    return th.shared(value=initial_W, name=name, borrow=borrow)


def initial_bias_vector(n_units, name, borrow=True):
    """Create the initial bias vector. Credit goes to:
    http://deeplearning.net/tutorial/dA.html

    Parameters
    ----------
    # todo
    """
    return th.shared(
        value=np.zeros(
            n_units,
            dtype=th.config.floatX
        ),
        name=name,
        borrow=borrow
    )


def _validate_float(val, name):
    try:
        res = float(val)
        assert 0 < val < 1  # the assertion error will be caught and valueerror raised
    except:
        raise ValueError('%s must be a float between 0 and 1' % name)
    return res


class DenoisingAutoEncoder(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    activation_function

    borrow : bool, optional (default=True)
        It is a safe practice (and a good idea) to use ``borrow=True`` in a ``shared``
        variable constructor when the shared variable stands for a large object 
        (in terms of memory footprint) and you do not want to create copies of it in memory.
        Since the ``AutoEncoder`` shares ``X`` in the ``fit`` method, by default ``borrow``
        is True. If you intend to run this on a GPU server, it is recommended you set ``borrow``
        to True.

    n_hidden : int or list, optional (default=None)
        The number of neurons in the hidden layer. If the default (None) is used, ``hidden_size``
        will amount to ``0.6 * n_features`` so the network is forced to learn a compressed feature 
        space.

    n_visible

    W : ``theano.tensor.TensorType``, optional (default=None)
        If this autoencoder will be used for stacked de-noising, this should point to a set of weights
        that should be shared between the autoencoder instance, and the other architecture. If this
        autoencoder will stand alone, ``W`` should be None (default).

    b_hid : ``theano.tensor.TensorType``, optional (default=None)
        The Theano variable pointing to a set of bias values that will be shared between this autoencoder
        and another architecture, if stacked. If this autoencoder will stand alone, ``b_hid`` should be None
        (default).

    b_vis : ``theano.tensor.TensorType``, optional (default=None)
        The Theano variable pointing to a set of bias values that will be shared between this autoencoder
        and another architecture, if stacked. If this autoencoder will stand alone, ``b_vis`` should be None
        (default).

    n_epochs : int, optional (default=100)

    mini_batch_size : int, optional (default=1)

    learning_rate : float, optional (default=0.1)
        The rate at which the autoencoder will learn. Default is 0.1.

    denoise_amount : float, optional (default=0.3)

    verbose : int, optional (default=0)
        The level of verbosity. 0 will produce no output.

    random_state : int, None or ``numpy.random.RandomState``
        The PRNG used to generate initial weight matrix and for other random operations.
    """

    def __init__(self, activation_function=nn.relu, borrow=True, n_hidden=None, n_visible=784,
                 W=None, b_hid=None, b_vis=None, n_epochs=100, mini_batch_size=1, learning_rate=0.1,
                 denoise_amount=0.3, verbose=0, random_state=None):

        self.activation_function = activation_function
        self.borrow = borrow
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.W = W
        self.b_hid = b_hid
        self.b_vis = b_vis
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.denoise_amt = denoise_amount
        self.verbose = verbose
        self.random_state = get_random_state(random_state)  # set up the random state if not already

    def fit(self, X, y=None, **kwargs):
        # get names. These are assigned to unique UUID values to ensure no collisions, in
        # case the name keys actually matter for anything lookup-related...
        model_id = str(uuid.uuid1())
        X_nm, W_nm, bvis_nm, bhid_nm = tuple('ae-%s-%s' % (model_id, c) for c in ('X', 'W', 'b_vis', 'b_hid'))

        # validate X, share with Theano
        borrow = self.borrow
        X = th.shared(name=X_nm,
                      value=check_array(X, accept_sparse=False,
                                        force_all_finite=True,
                                        ensure_2d=True),
                      dtype=DTYPE, borrow=borrow)
        n_samples, n_features = X.shape

        # validate activation
        if not all(func in PERMITTED_ACTIVATIONS for func in (self.activation_function,)):
            raise ValueError('Permitted activation/output functions: %r' % PERMITTED_ACTIVATIONS)

        # validate the denoising amt, re-assign in class
        self.denoise_amt = denoise_amt = _validate_float(self.denoise_amt, 'denoise_amt')
        self.learning_rate = learning_rate = _validate_float(self.learning_rate, 'learning_rate')

        # validate the n_hidden, n_visible
        n_hidden, n_visible = self.n_hidden, self.n_visible
        if n_hidden is None:
            n_hidden = max(1, int(0.6 * n_features))
        if not all(isinstance(x, (int, np.int)) for x in (n_hidden, n_visible)):
            raise ValueError('n_hidden and n_visible must be ints')

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        random_state = self.random_state
        W, b_vis, b_hid = self.W, self.b_vis, self.b_hid
        if not W:
            W = initial_weight_matrix(n_hidden, n_visible, name=W_nm,
                                      random_state=random_state,
                                      borrow=borrow)

        if not b_vis:
            b_vis = initial_bias_vector(n_visible, name=bvis_nm, borrow=borrow)

        if not b_hid:
            b_hid = initial_bias_vector(n_hidden, name=bhid_nm, borrow=borrow)

        # assign instance vars:
        self.W_ = W
        self.b_ = b_hid
        self.b_prime_ = b_vis
        self.W_prime_ = W.T  # tied weights, so W_prime is W transpose

        # set a random stream for a Theano RNG using the numpy RNG
        theano_rng = RandomStreams(random_state.randint(2 ** 30))

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        cost, updates = self._get_cost_updates(X, theano_rng)
        train_func = th.function([index], cost, updates=updates,
                                 givens={
                                     x:
                                 })

        start_time = timeit.default_timer()
        for epoch in range(self.n_epochs):



    def _get_hidden_values(self, X):
        """Compute values of hidden layer"""
        return self.activation_function(T.dot(X, self.W_) + self.b_)

    def _get_corrupted_input(self, X, theano_rng):
        """
        #todo
        """
        return theano_rng.binomial(size=X.shape, n=self.mini_batch_size,
                                   p=1 - self.denoise_amt,
                                   dtype=th.config.floatX) * X

    def _get_reconstructed_input(self, y):
        return self.activation_function(T.dot(y, self.W_prime_) + self.b_prime_)

    def _get_cost_updates(self, X, theano_rng):
        """ This function computes the cost and the updates for one training
        step of the dA """

        tilde_X = self._get_corrupted_input(X, theano_rng)
        y = self._get_hidden_values(tilde_X)
        z = self._get_reconstructed_input(y)

        # compute the cost (cross-entropy):
        L = -T.sum(X * T.log(z) + (1 - X) * T.log(1 - z), axis=1)
        cost = L.mean()

        # compute gradients with respect to params
        params = [self.W_, self.b_, self.b_prime_]
        gparams = T.grad(cost, params)

        # create list of updates
        updates = [
            (param, param * self.learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

        return (cost, updates)

    def transform(self, X):

    def reconstruct(self, X):

