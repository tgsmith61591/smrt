# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# The SMRT balancer

from __future__ import division, absolute_import, division

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .base import _validate_X_y_ratio_classes
from ..utils import get_random_state, validate_float
from ..autoencode import VariationalAutoEncoder
from . import base

__all__ = [
    'smrt_balance'
]

DEFAULT_SEED = base.DEFAULT_SEED
MAX_N_CLASSES = base.MAX_N_CLASSES
MIN_N_SAMPLES = base.MIN_N_SAMPLES


def smrt_balance(X, y, n_hidden, n_latent_factors, return_estimators=False, balance_ratio=0.2,
                 activation_function='sigmoid', learning_rate=0.05, n_epochs=20, batch_size=128, min_change=1e-6,
                 verbose=0, display_step=5, learning_function='rms_prop', early_stopping=False, bias_strategy='zeros',
                 random_state=DEFAULT_SEED, layer_type='xavier', dropout=1., l2_penalty=0.0001,
                 eps=1e-10, shuffle=True, generate_args={}):
    """SMRT (Sythetic Minority Reconstruction Technique) is the younger, more sophisticated cousin to
    SMOTE (Synthetic Minority Oversampling TEchnique). Using variational auto-encoders, SMRT learns the
    latent factors that best reconstruct the observations in each minority class, and then generates synthetic
    observations until the minority class is represented at a minimum of ``balance_ratio`` * majority_class_size.

    SMRT avoids one of SMOTE's greatest risks: In SMOTE, when drawing random observations from whose k-nearest
    neighbors to reconstruct, the possibility exists that a "border point," or an observation very close to 
    the decision boundary may be selected. This could result in the synthetically-generated observations lying 
    too close to the decision boundary for reliable classification, and could lead to the degraded performance
    of an estimator. SMRT avoids this risk implicitly, as the :class:``VariationalAutoencoder`` learns a
    distribution that is generalizable to the lowest-error (i.e., most archetypal) observations.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    n_hidden : int or list
        The hidden layer structure. If an int is provided, a single hidden layer is constructed,
        with ``n_hidden`` neurons. If ``n_hidden`` is an iterable, ``len(n_hidden)`` hidden layers
        are constructed, with as many neurons as correspond to each index, respectively.

    n_latent_factors : int or float
        The size of the latent factor layer learned by the ``VariationalAutoEncoder``

    return_estimators : bool, optional (default=False)
        Whether or not to return the dictionary of fit :class:``smrt.autoencode.AutoEncoder`` instances.
        If True, the return value will be a tuple, with the first index being the balanced
        ``X`` matrix, the second index being the ``y`` values, and the third index being a 
        dictionary of the fit encoders. If False, the return value is simply the balanced ``X`` 
        matrix and the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    activation_function : str, optional (default='sigmoid')
        The activation function. Should be one of PERMITTED_ACTIVATIONS:
        ('elu', 'identity', 'relu', 'sigmoid', 'tanh')

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs.

    batch_size : int, optional (default=128)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

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
        the ``tf.train.RMSPropOptimizer``. Can be one of { ``'adadelta'``, ``'adagrad'``,
        ``'adagrad-da'``, ``'adam'``, ``'momentum'``, ``'proximal-sgd'``, ``'proximal-adagrad'``,
        ``'rms_prop'``, ``'sgd'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random TensorFlow variables in weight initialization.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    l2_penalty : float or None, optional (default=0.0001)
        The L2 penalty (regularization term) parameter.

    eps : float, optional (default=1e-10)
        A small amount of noise to add to the loss to avoid a potential computation of
        ``log(0)``.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.

    generate_args : dict, optional (default=``{}``)
        Any extra keyword arguments to pass to the generate function. These arguments will
        ultimately be passed into ``random_state.normal``. Appropriate args include:

          * 'loc' : float or array_like of floats
            Mean (“centre”) of the distribution.

          * 'scale' : float or array_like of floats
            Standard deviation (spread or “width”) of the distribution.
    """
    # validate the cheap stuff before copying arrays around...
    X, y, n_classes, present_classes, \
    counts, majority_label, target_count = _validate_X_y_ratio_classes(X, y, balance_ratio)

    # make sure it's not just an int
    random_state = get_random_state(random_state)

    # encode y, in case they are not numeric
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric (we need them to be for np.ones)

    # create X copy on which to append. We do this because we do not want to augment
    # synthetic examples of already-reconstructed examples...
    X_copy = X[:, :]

    # start the iteration...
    encoders = dict()  # map the label to the fit encoder
    for i, label in enumerate(present_classes):
        # if the count >= the ratio, skip this label
        # also skip if it's the majority class (which would be covered in first condition, I guess...)
        count = counts[i]
        if label == majority_label or count >= target_count:
            encoders[label] = None
            continue

        # fit the autoencoder
        encoder = VariationalAutoEncoder(n_hidden=n_hidden, n_latent_factors=n_latent_factors,
                                         activation_function=activation_function, learning_rate=learning_rate,
                                         n_epochs=n_epochs, batch_size=batch_size, min_change=min_change,
                                         verbose=verbose, display_step=display_step,
                                         learning_function=learning_function,
                                         early_stopping=early_stopping, bias_strategy=bias_strategy,
                                         random_state=random_state, layer_type=layer_type, dropout=dropout,
                                         l2_penalty=l2_penalty, eps=eps)

        # transform label
        transformed_label = le.transform([label])[0]
        X_sub = X[y_transform == transformed_label, :]

        # fit the model, store it
        encoders[label] = encoder.fit(X_sub)

        # get the number of synthetic obs we need
        obs_req = target_count - X_sub.shape[0]
        synthetic = encoder.generate(n=obs_req, **generate_args)

        # append
        X_copy = np.vstack([X_copy, synthetic])
        y_transform = np.concatenate([y_transform, np.ones(obs_req, dtype=np.int16) * transformed_label])

    # now that X, y_transform have been assembled, inverse_transform the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    output_order = np.arange(X_copy.shape[0])
    if shuffle:
        output_order = random_state.permutation(output_order)

    if return_estimators:
        return X_copy[output_order, :], y[output_order], encoders
    return X_copy[output_order, :], y[output_order]
