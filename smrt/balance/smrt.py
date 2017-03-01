# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# The SMRT balancer

from __future__ import division, absolute_import, division

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from .base import _validate_X_y_ratio_classes
from ..utils import get_random_state, validate_float
from ..autoencode import AutoEncoder, VariationalAutoEncoder
from . import base

__all__ = [
    'smrt_balance'
]

DEFAULT_SEED = base.DEFAULT_SEED
MAX_N_CLASSES = base.MAX_N_CLASSES
MIN_N_SAMPLES = base.MIN_N_SAMPLES


def smrt_balance(X, y, return_estimators=False, balance_ratio=0.2, strategy='perturb', min_error_sample=0.25,
                 activation_function='relu', learning_rate=0.05, n_epochs=200, batch_size=256, n_hidden=None,
                 compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5, seed=DEFAULT_SEED,
                 xavier_init=True, n_latent_factors=None, eps=1e-10, shuffle=True, smote_args={}, **kwargs):
    """SMRT (Sythetic Minority Reconstruction Technique) is the younger, more sophisticated cousin to
    SMOTE (Synthetic Minority Oversampling TEchnique). Using auto-encoders, SMRT learns the parameters
    that best reconstruct the observations in each minority class, and then generates synthetic observations
    until the minority class is represented at a minimum of ``balance_ratio`` * majority_class_size. 

    SMRT avoids one of SMOTE's greatest risks: In SMOTE, when drawing random observations from whose k-nearest
    neighbors to reconstruct, the possibility exists that a "border point," or an observation very close to 
    the decision boundary may be selected. This could result in the synthetically-generated observations lying 
    too close to the decision boundary for reliable classification, and could lead to the degraded performance
    of an estimator. SMRT avoids this risk by ranking observations according to their reconstruction MSE, and
    drawing samples to reconstruct from the lowest-MSE observations (i.e., the most "phenotypical" of a class).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    return_estimators : bool, optional (default=False)
        Whether or not to return the dictionary of fit :class:``smrt.autoencode.AutoEncoder`` instances.
        If True, the return value will be a tuple, with the first index being the balanced
        ``X`` matrix, the second index being the ``y`` values, and the third index being a 
        dictionary of the fit encoders. If False, the return value is simply the balanced ``X`` 
        matrix and the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    min_error_sample : float, optional (default=0.25)
        The ratio of the existing minority records from which to sample. Selects the lowest ``min_error_sample``
        percent (by error) of records for reconstruction.

    activation_function : str or callable, optional (default='relu')
        The activation function. If a str, it should be one of PERMITTED_ACTIVATIONS. If a
        callable, it should be an activation function contained in the ``tensorflow.nn`` module.

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs.

    batch_size : int, optional (default=256)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    n_hidden : int, list or dictionary , optional (default=None)
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
        is less than ``min_change``, the network will stop fitting early.

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=5)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the weight and bias initialization.

    xavier_init : bool, optional (default=True)
        Whether to use Xavier's initialization for the weights.

    n_latent_factors : int or float, optional (default=None)
        The size of the latent factor layer learned by the ``VariationalAutoEncoder``

    eps : float, optional (default=1e-10)
        A small amount of noise to add to the loss to avoid a potential computation of
        ``log(0)``.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.
    """
    # validate the cheap stuff before copying arrays around...
    X, y, n_classes, present_classes, \
    counts, majority_label, target_count = _validate_X_y_ratio_classes(X, y, balance_ratio)

    random_state = get_random_state(seed)
    validate_float(min_error_sample, 'min_error_sample')

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
        if label == majority_label:
            continue

        # if the count >= the ratio, skip this label
        count = counts[i]
        if count >= target_count:
            encoders[label] = None
            continue

        # fit the autoencoder
        encoder = AutoEncoder(activation_function=activation_function, learning_rate=learning_rate, n_epochs=n_epochs,
                              batch_size=batch_size, n_hidden=n_hidden, compression_ratio=compression_ratio,
                              min_change=min_change, verbose=verbose, display_step=display_step, seed=seed,
                              **kwargs)

        # transform label
        transformed_label = le.transform([label])[0]
        X_sub = X[y_transform == transformed_label, :]

        # fit the model, store it
        encoder.fit(X_sub)
        encoders[label] = encoder

        # transform X_sub, rank it
        reconstructed = encoder.reconstruct(X_sub)
        mse = np.asarray([
            mean_squared_error(X_sub[i, :], reconstructed[i, :])
            for i in range(X_sub.shape[0])
        ])

        # rank order:
        mse_order = np.argsort(mse)
        ordered = X_sub[mse_order, :]  # order asc by reconstr error
        reconstructed_ordered = reconstructed[mse_order, :]

        # sample_count = target_count - X_sub.shape[0]  # todo: redo this--pull a subset of these..
        sample_count = int(round(min_error_sample * X_sub.shape[0]))  # the num rows to select from bottom

        # the number of obs we need
        obs_req = target_count - X_sub.shape[0]

        # sample while under the requisite ratio
        while obs_req > 0:  # shouldn't be less than... but just in case

            # if obs_req is lower than the sample_count, go with the min
            sample_count = min(obs_req, sample_count)
            perturb_sample = ordered[:sample_count]  # makes a copy we can perturb
            reconst_sample = reconstructed_ordered[:sample_count]  # the corresponding reconstruction samples

            # perturb the sample. Subtract the reconstruction matrix from the original sample,
            # then We create a random matrix, M x N, (bound in [0, 1]), multiply it by the difference matrix,
            # and add it to the original sample.
            perturb_sample += ((perturb_sample - reconst_sample) * random_state.rand(*perturb_sample.shape))

            # append to X, y_transform
            X_copy = np.vstack([X_copy, perturb_sample])
            y_transform = np.concatenate([y_transform,
                                          np.ones(sample_count, dtype=np.int16) * transformed_label])

            # determine whether we need to recurse for this class (if there were too few samples)
            # update the required amount
            obs_req -= sample_count

    # now that X, y_transform have been assembled, inverse_transform the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    if shuffle:
        output_order = random_state.shuffle(np.arange(X_copy.shape[0]))
    else:
        output_order = np.arange(X_copy.shape[0])

    if return_estimators:
        return X_copy[output_order, :], y[output_order], encoders
    return X_copy[output_order, :], y[output_order]
