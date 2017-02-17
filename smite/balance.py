# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMITE balancer

from __future__ import division, absolute_import, division
from sknn.ae import AutoEncoder
import numpy as np

__all__ = [
    'balance'
]

MAX_N_CLASSES = 100
MIN_N_SAMPLES = 2


def balance(X, y, layers, ratio=0.2, random_state=None, parameters=None, learning_rule='sgd',
            learning_rate=0.01, learning_momentum=0.9, batch_size=1, n_iter=None, n_stable=10, f_stable=0.001, 
            valid_set=None, valid_size=0.0, normalize=None, regularize=None, weight_decay=None, dropout_rate=None, 
            loss_type=None, callback=None, debug=False, verbose=None, warning=None, **params):
    """
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    layers: list of :class:``sknn.ae.Layer``
        An iterable sequence of each layer each as a :class:``sknn.ae.Layer`` instance that
        contains its type, optional name, and any paramaters required.

            * For hidden layers, you can use the following layer types:
              ``Sigmoid``, ``Tanh``.

        It's possible to mix and match any of the layer types.

    ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0. <= ``ratio`` <= 1.

    random_state: int, optional
        Seed for the initialization of the neural network parameters (e.g.
        weights and biases).  This is fully deterministic.

    parameters: list of tuple or array-like, optional (default=None)
        A list of ``(weights, biases)`` tuples to be reloaded for each layer, in the same
        order as ``layers`` was specified.  Useful for initializing with pre-trained
        networks.

    learning_rule: str, optional (default='sgd')
        Name of the learning rule used during stochastic gradient descent,
        one of ``sgd``, ``momentum``, ``nesterov``, ``adadelta``, ``adagrad`` or
        ``rmsprop`` at the moment.  The default is vanilla ``sgd``.

    learning_rate: float, optional (default=0.01)
        Real number indicating the default/starting rate of adjustment for
        the weights during gradient descent. Different learning rules may
        take this into account differently. Default is ``0.01``.

    learning_momentum: float, optional (default=0.9)
        Real number indicating the momentum factor to be used for the
        learning rule 'momentum'. Default is ``0.9``.

    batch_size: int, optional (default=1)
        Number of training samples to group together when performing stochastic
        gradient descent (technically, a "minibatch").  By default each sample is
        treated on its own, with ``batch_size=1``.  Larger batches are usually faster.

    n_iter: int, optional (default=None)
        The number of iterations of gradient descent to perform on the
        neural network's weights when training with ``fit()``.

    n_stable: int, optional (default=10)
        Number of interations after which training should return when the validation
        error remains (near) constant.  This is usually a sign that the data has been
        fitted, or that optimization may have stalled.  If no validation set is specified,
        then stability is judged based on the training error.  Default is ``10``.

    f_stable: float, optional (default=0.001)
        Threshold under which the validation error change is assumed to be stable, to
        be used in combination with `n_stable`. This is calculated as a relative ratio
        of improvement, so if the results are only 0.1% better training is considered
        stable. The training set is used as fallback if there's no validation set. Default
        is ``0.001`.

    valid_set: tuple of array-like, optional (default=None)
        Validation set (X_v, y_v) to be used explicitly while training.  Both
        arrays should have the same size for the first dimention, and the second
        dimention should match with the training data specified in ``fit()``.

    valid_size: float, optional (default=0.0)
        Ratio of the training data to be used for validation.  0.0 means no
        validation, and 1.0 would mean there's no training data!  Common values are
        0.1 or 0.25.

    normalize: string, optional (default=None)
        Enable normalization for all layers. Can be either `batch` for batch normalization
        or (soon) `weights` for weight normalization.  Default is no normalization.

    regularize: string, optional (default=None)
        Which regularization technique to use on the weights, for example ``L2`` (most
        common) or ``L1`` (quite rare), as well as ``dropout``.  By default, there's no
        regularization, unless another parameter implies it should be enabled, e.g. if
        ``weight_decay`` or ``dropout_rate`` are specified.

    weight_decay: float, optional (default=None)
        The coefficient used to multiply either ``L1`` or ``L2`` equations when computing
        the weight decay for regularization.  If ``regularize`` is specified, this defaults
        to 0.0001.
        
    dropout_rate: float, optional (default=None)
        What rate to use for drop-out training in the inputs (jittering) and the
        hidden layers, for each training example. Specify this as a ratio of inputs
        to be randomly excluded during training, e.g. 0.75 means only 25% of inputs
        will be included in the training.

    loss_type: string, optional (default=None)
        The cost function to use when training the network.  There are two valid options:
            * ``mse`` — Use mean squared error, for learning to predict the mean of the data.
            * ``mae`` — Use mean average error, for learning to predict the median of the data.
            * ``mcc`` — Use mean categorical cross-entropy, particularly for classifiers.
        The default option is ``mse`` for regressors and ``mcc`` for classifiers, but ``mae`` can
        only be applied to layers of type ``Linear`` or ``Gaussian`` and they must be used as
        the output layer (PyLearn2 only).

    callback: callable or dict, optional (default=None)
        An observer mechanism that exposes information about the inner training loop. This is
        either a single function that takes ``cbs(event, **variables)`` as a parameter, or a
        dictionary of functions indexed by on `event` string that conforms to ``cb(**variables)``.
        
        There are multiple events sent from the inner training loop:
        
            * ``on_train_start`` — Called when the main training function is entered.
            * ``on_epoch_start`` — Called the first thing when a new iteration starts.
            * ``on_batch_start`` — Called before an individual batch is processed.
            * ``on_batch_finish`` — Called after that individual batch is processed.
            * ``on_epoch_finish`` — Called the first last when the iteration is done.
            * ``on_train_finish`` — Called just before the training function exits.
        
        For each function, the ``variables`` dictionary passed contains all local variables within
        the training implementation.

    debug: bool, optional (default=False)
        Should the underlying training algorithms perform validation on the data
        as it's optimizing the model?  This makes things slower, but errors can
        be caught more effectively.  Default is off.

    verbose: bool, optional (default=None)
        How to initialize the logging to display the results during training. If there is
        already a logger initialized, either ``sknn`` or the root logger, then this function
        does nothing.  Otherwise:
            * ``False`` — Setup new logger that shows only warnings and errors.
            * ``True`` — Setup a new logger that displays all debug messages.
            * ``None`` — Don't setup a new logger under any condition (default). 
        Using the built-in python ``logging`` module, you can control the detail and style of
        output by customising the verbosity level and formatter for ``sknn`` logger.
        
    warning: None
        You should use keyword arguments after `layers` when initializing this object. If not,
        the code will raise an ``AssertionError``.
    """
    X = check_array(X, accept_sparse=False, dtype=np.float32)
    y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)

    n_samples, _ = X.shape
    y = np.atleast_1d(y)

    if y.ndim == 1:
        # reshape is necessary to preserve the data contiguity against vs
        # [:, np.newaxis] that does not.
        y = np.reshape(y, (-1, 1))

    # get n classes in y, ensure they are <= MAX_N_CLASSES
    # todo

    pass
