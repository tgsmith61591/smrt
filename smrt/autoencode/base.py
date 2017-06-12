# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Base utils for the autoencoder(s)

from __future__ import division, absolute_import, division
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_batches
from sklearn.externals import six
import numpy as np
import tensorflow as tf
import time

from ..utils import validate_float, DTYPE, get_random_state, NPDTYPE

__all__ = [
    'BaseAutoEncoder'
]


def _validate_positive_integer(instance, name):
    val = getattr(instance, name)
    try:
        assert not isinstance(val, (bool, np.bool))
        res = int(val)
        assert res >= 0
    except:
        raise ValueError('%s must be an int >= 0' % name)
    return res


def _validate_float(instance, name, upper_bound=1., ltet=False):
    try:
        res = float(getattr(instance, name))
        validate_float(res, name, upper_bound, ltet=ltet)
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
                 layer_type, dropout, l2_penalty, gclip_min, gclip_max, clip):

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
        self.l2_penalty = l2_penalty
        self.gclip_min = gclip_min
        self.gclip_max = gclip_max
        self.clip = clip

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
        self.dropout = _validate_float(self, 'dropout', ltet=True)

        # l2 is allowed to be None
        if self.l2_penalty is not None:
            self.l2_penalty = _validate_float(self, 'l2_penalty')

    @abstractmethod
    def encode(self, X):
        """Pass the ``X`` array through the inferential MLP layers.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The array of samples that will be encoded into the new
            hidden layer space.
        """

    def transform(self, X):
        """Inherited from the ``TransformerMixin``. Pass the ``X`` array
        through the inferential MLP layers.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The array of samples that will be encoded into the new
            hidden layer space.
        """
        return self.encode(X)

    def clean_session(self):
        """If, i.e., re-fitting a model, this method cleans out the existing tensorflow
        session object, if it exists.
        """
        if hasattr(self, 'sess'):
            self.sess.close()
            delattr(self, 'sess')

    @abstractmethod
    def _initialize_graph(self, X, y):
        """Should be called in the ``fit`` method. This initializes the placeholder variables"""

    def _add_regularization(self, cost_function, topography):
        if self.l2_penalty is not None:
            penalties = [tf.nn.l2_loss(w) for w in topography.get_weights_biases()[0]]
            l2_reg = self.l2_penalty * tf.add_n(penalties)
            cost_function += l2_reg

        return cost_function

    def _clip_or_minimize(self, learning_function, rate, cost):
        # https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
        if not self.clip:
            return learning_function(rate).minimize(cost)
        else:
            global_step = tf.Variable(0, trainable=False)
            optimizer = learning_function(rate)
            grads = optimizer.compute_gradients(cost, tf.trainable_variables())
            clipped = [
                (tf.clip_by_value(grad, self.gclip_min, self.gclip_max), var)
                for grad, var in grads
                if grad is not None
            ]

            return optimizer.apply_gradients(clipped, global_step=global_step)

    def fit(self, X, y=None, **run_args):
        """Train the neural network.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors as real numbers, where ``n_samples`` is the number of
            samples and ``n_features`` is the number of input features.

        y : array-like, optional (default=None)
            None. Pass-through for pipe-lining.

        **run_args : dict, optional
            A key-word dictionary of arguments to be passed to the :func:`_train` method.
        """
        # validate array before graph init
        X = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True, dtype=NPDTYPE)

        # set the TF seed
        tf.set_random_seed(self.random_state.seed_value)

        # assign X to tf as a placeholder before graph init
        self.X_placeholder = tf.placeholder(DTYPE, [None, X.shape[1]])

        # initialize the graph for each re-fit
        X, cost_function, optimizer, dropout = self._initialize_graph(X, y)

        # do training
        return self._train(self.X_placeholder, X, cost_function, optimizer, dropout, **run_args)

    def _train(self, X_placeholder, X_original, cost_function, optimizer, dropout, **run_args):
        # initialize global vars for tf - replace them if they already exist
        init = tf.global_variables_initializer()
        self.clean_session()
        sess = self.sess = tf.InteractiveSession()

        # run the training session
        sess.run(init)
        epoch_times = []
        costs = []
        last_cost = None

        # generate the batches in a generator from sklearn, but store
        # in a list so we don't have to re-gen (since the generator will be
        # empty by the end of the epoch)
        n_samples = X_original.shape[0]
        batches = list(gen_batches(n_samples, self.batch_size))

        # training cycle. For each epoch
        for epoch in range(self.n_epochs):
            # track epoch time
            start_time = time.time()

            # loop batches
            for batch in batches:

                # extract the chunk given the slice, and assert it's not length 0 or anything weird...
                chunk = X_original[batch, :]
                m, _ = chunk.shape
                assert m <= self.batch_size and m != 0  # sanity check

                # train the batch - runs optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost_function],
                                feed_dict={X_placeholder: chunk, dropout: self.dropout},
                                **run_args)

            # add the time to the times array to compute average later
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            costs.append(c)

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
                    else:
                        last_cost = c

        # set instance vars
        self.train_cost_ = c
        self.epoch_times_ = epoch_times
        self.epoch_costs_ = costs

        if self.verbose:
            print('Optimization complete after %i epoch(s). Average epoch time: %.4f seconds'
                  % (len(epoch_times), np.average(epoch_times)))

        return self

    @abstractmethod
    def feed_forward(self, X):
        """Pass a matrix, ``X``, through both the encoding and decoding functions."""
