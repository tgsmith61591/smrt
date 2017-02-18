# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMITE balancer

from __future__ import division, absolute_import, division
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.utils.validation import check_array
from sknn.ae import AutoEncoder, Layer
import numpy as np

__all__ = [
    'balance',
    'LayerParameters'
]

MAX_N_CLASSES = 100  # max unique classes in y
MIN_N_SAMPLES = 2  # min n_samples per class in y


class LayerParameters(BaseEstimator):
    """A wrapper class for the ``sknn.ae.Layer`` class. ``sknn`` expects
    a list of ``Layer`` instances, but these are mutable and are tied to the underlying
    weight matrix. Since the ``balance`` function will fit many encoders, the ``layers``
    parameter should not allow mutability. Thus, this class defines a template for the
    instantiation of new layers given a set of parameters.

    Parameters
    ----------
    activation : str, optional (default='Sigmoid')
        Select which activation function this layer should use, as a string.
        The possible activation functions for the ``sknn.ae.AutoEncoder`` are:

            * ``'Sigmoid'``
            * ``'Tanh'``

        Note that the ``'Rectifier'`` is not currently supported.

    layer_type : str, optional (default='autoencoder')
        The type of encoding and decoding layer to use, specifically ``denoising`` for randomly
        corrupting data, and a more traditional ``autoencoder`` which is used by default.

    name : str, optional (default=None)
        You optionally can specify a name for this layer, and its parameters
        will then be accessible to scikit-learn via a nested sub-object.  For example,
        if name is set to ``layer1``, then the parameter ``layer1__units`` from the network
        is bound to this layer's ``units`` variable. The name defaults to ``hiddenN`` where N 
        is the integer index of that layer, and the final layer is always ``output`` without 
        an index.

    units: int, optional (default=None)
        The number of units (also known as neurons) in this layer.

    cost: string, optional (default='msre')
        What type of cost function to use during the layerwise pre-training.  This can be either
        ``'msre'`` for mean-squared reconstruction error (default), and ``'mbce'`` for mean binary
        cross entropy.

    tied_weights: bool, optional (default=True)
        Whether to use the same weights for the encoding and decoding phases of the simulation
        and training.  Default is ``True``.

    corruption_level: float, optional (default=0.5)
        The ratio of inputs to corrupt in this layer; ``0.25`` means that 25% of the inputs will be
        corrupted during the training.  The default is ``0.5``.
    """
    def __init__(self, activation='Sigmoid', layer_type='autoencoder', name=None, units=None,
                 cost='msre', tied_weights=True, corruption_level=0.5):
        self.activation = activation.title()  # sknn dev likes his leading capitals...
        self.layer_type = layer_type
        self.name = name
        self.units = units
        self.cost = cost  # don't make lower, since lower should be expected
        self.tied_weights = tied_weights
        self.corruption_level = corruption_level

    def build_new(self):
        """Construct a new ``sknn.ae.Layer`` instance from the class parameters."""
        return Layer(activation=self.activation, type=self.layer_type, name=self.name,
                     units=self.units, cost=self.cost, tied_weights=self.tied_weights,
                     corruption_level=self.corruption_level, warning=None)  # he has this weird arg in there...


def balance(X, y, layers=None, return_encoders=False, balance_ratio=0.2, random_state=None, parameters=None, 
            learning_rule='sgd', learning_rate=0.01, learning_momentum=0.9, batch_size=1, n_iter=None, 
            n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, normalize=None, regularize=None, 
            weight_decay=None, dropout_rate=None, loss_type=None, callback=None, debug=False, verbose=None, 
            **params):
    """SMITE (Sythetic Minority Interpolation TEchnique) is the younger, more sophisticated cousin to
    SMOTE (Synthetic Minority Oversampling TEchnique). Using auto-encoders, SMITE learns the parameters 
    that best reconstruct the observations in each minority class, and then generates synthetic observations
    until the minority class is represented at a minimum of ``balance_ratio`` * majority_class_size. 

    SMITE avoids one of SMOTE's greatest risks In SMOTE, when drawing random observations from whose k-nearest 
    neighbors to reconstruct, the possibility exists that a "border point," or an observation very close to 
    the decision boundary may be selected. This could result in the synthetically-generated observations lying 
    too close to the decision boundary for reliable classification, and could lead to the degraded performance
    of an estimator. SMITE avoids this risk, by ranking observations according to their reconstruction MSE, and
    drawing samples to reconstruct from the lowest-MSE observations (i.e., the most "phenotypical" of a class).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    layers : list of :class:``LayerParameters``, optional (default=None)
        An iterable sequence of ``LayerParameters`` defining the structure of 
        the hidden layers. If layers is not specifed, the default is to create a single hidden
        layer with default ``LayerParameters`` args, and with ``0.6 * n_features``.

    return_encoders : bool, optional (default=False)
        Whether or not to return the dictionary of fit ``sknn.ae.AutoEncoder`` instances.
        If True, the return value will be a tuple, with the first index being the balanced
        ``X`` matrix, and the second index being a dictionary of the fit encoders. If False,
        the return value is simply the balanced ``X`` matrix.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0. < ``ratio`` <= 1.

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
        The cost function to use when training the network.  There are several valid options:

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
    """
    # validate the cheap stuff before copying arrays around...
    assert 0. < balance_ratio <= 1., 'Expected 0 < balance_ratio <= 1, but got balance_ratio=%r' % balance_ratio

    # validate arrays
    X = check_array(X, accept_sparse=False, dtype=np.float32)
    y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)

    n_samples, n_features = X.shape
    y = np.atleast_1d(y)

    if y.ndim == 1:
        # reshape is necessary to preserve the data contiguity against vs
        # [:, np.newaxis] that does not.
        y = np.reshape(y, (-1, 1))

    # get n classes in y, ensure they are <= MAX_N_CLASSES, but first ensure these are actually
    # class labels and not floats or anything...
    y_type = type_of_target(y)
    supported_types = {'multiclass', 'binary'}
    if y_type not in supported_types:
        raise ValueError('SMITE balancer only supports %r, but got %r' % (supported_types, y_type))

    present_classes = unique_labels(y)
    n_classes = len(present_classes)

    # ensure <= MAX_N_CLASSES
    if n_classes > MAX_N_CLASSES:
        raise ValueError('SMITE balancer currently only supports a maximum of %i '
                         'unique class labels, but %i were identified.' % (MAX_N_CLASSES, n_classes))

    # check layers:
    if layers is None:
        layers = [LayerParameters(units=max(1, 0.6 * n_features))]
    else:
        assert isinstance(layers, (list, tuple)), 'expected a list or tuple for layers, but got type=%s' % type(layers)
        assert all(isinstance(x, LayerParameters) for x in layers), 'layers should be a list or tuple of smite.balance.LayerParameters'

    

    pass
