# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMRT balancer

from __future__ import division, absolute_import, division
from sklearn.utils import column_or_1d
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array
from .autoencode import AutoEncoder
from .utils import get_random_state

# we have to import from the backend first. Weird design...
import numpy as np

__all__ = [
    'smrt_balance'
]

MAX_N_CLASSES = 100  # max unique classes in y
MIN_N_SAMPLES = 2  # min n_samples per class in y


def _validate_ratios(ratio, name):
    if not 0. < ratio <= 1.:
        raise ValueError('Expected 0 < %s <= 1, but got %r' 
                         % (name, ratio))


def smrt_balance(X, y, return_encoders=False, balance_ratio=0.2, jitter=1.0, activation_function='relu',
                 learning_rate=0.05, n_epochs=200, batch_size=256, n_hidden=None, compression_ratio=0.6,
                 min_change=1e-6, verbose=0, display_step=5, seed=42, shuffle=True):
    """SMRT (Sythetic Minority Reconstruction Technique) is the younger, more sophisticated cousin to
    SMOTE (Synthetic Minority Oversampling TEchnique). Using auto-encoders, SMRT learns the parameters
    that best reconstruct the observations in each minority class, and then generates synthetic observations
    until the minority class is represented at a minimum of ``balance_ratio`` * majority_class_size. 

    SMRT avoids one of SMOTE's greatest risks: In SMOTE, when drawing random observations from whose k-nearest
    neighbors to reconstruct, the possibility exists that a "border point," or an observation very close to 
    the decision boundary may be selected. This could result in the synthetically-generated observations lying 
    too close to the decision boundary for reliable classification, and could lead to the degraded performance
    of an estimator. SMRT avoids this risk, by ranking observations according to their reconstruction MSE, and
    drawing samples to reconstruct from the lowest-MSE observations (i.e., the most "phenotypical" of a class).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    return_encoders : bool, optional (default=False)
        Whether or not to return the dictionary of fit ``sknn.ae.AutoEncoder`` instances.
        If True, the return value will be a tuple, with the first index being the balanced
        ``X`` matrix, the second index being the ``y`` values, and the third index being a 
        dictionary of the fit encoders. If False, the return value is simply the balanced ``X`` 
        matrix and the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    jitter : float, optional (default=1.0)
        A value between 0 and 1. This is used to jitter the sample. We create a random matrix, 
        M x N, (bound in [0, 1]), subtract 0.5 (so there are some negatives), and then multiply
        by ``eps``. Finally, we multiply by the columnar standard deviations, and add to the sample.
        As ``eps`` approaches 1, the jitter is increased; as it approaches 0, it is decreased.

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

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.
    """
    # validate the cheap stuff before copying arrays around...
    _validate_ratios(balance_ratio, 'balance_ratio')
    _validate_ratios(jitter, 'jitter')
    random_state = get_random_state(seed)

    # validate arrays
    X = check_array(X, accept_sparse=False, dtype=np.float32)
    y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)

    n_samples, n_features = X.shape
    y = column_or_1d(y, warn=False)  

    # np.atleast_1d(y)
    # if y.ndim == 1:
        # reshape is necessary to preserve the data contiguity against vs
        # [:, np.newaxis] that does not.
        # y = np.reshape(y, (-1, 1))

    # get n classes in y, ensure they are <= MAX_N_CLASSES, but first ensure these are actually
    # class labels and not floats or anything...
    y_type = type_of_target(y)
    supported_types = {'multiclass', 'binary'}
    if y_type not in supported_types:
        raise ValueError('SMRT balancer only supports %r, but got %r' % (supported_types, y_type))

    present_classes, counts = np.unique(y, return_counts=True)
    n_classes = len(present_classes)

    # ensure <= MAX_N_CLASSES
    if n_classes > MAX_N_CLASSES:
        raise ValueError('SMRT balancer currently only supports a maximum of %i '
                         'unique class labels, but %i were identified.' % (MAX_N_CLASSES, n_classes))

    # get the majority class label, and its count:
    majority_count_idx = np.argmax(counts, axis=0)
    majority_label, majority_count = present_classes[majority_count_idx], counts[majority_count_idx]
    target_count = max(int(balance_ratio * majority_count), 1)

    # if any counts < MIN_N_SAMPLES, raise:
    if any(i < MIN_N_SAMPLES for i in counts):
        raise ValueError('All label counts must be >= %i' % MIN_N_SAMPLES)

    # encode y, in case they are not numeric
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric

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
                              min_change=min_change, verbose=verbose, display_step=display_step, seed=seed)

        # transform label
        transformed_label = le.transform([label])[0]

        # sample while under the requisite ratio
        while True:
            X_sub = X[y_transform == transformed_label, :]

            # the second+ time thru, we don't want to re-fit...
            if label not in encoders:
                encoder.fit(X_sub)
                encoders[label] = encoder

            # transform X_sub, rank it
            reconstructed = encoder.feed_forward(X_sub)
            mse = np.asarray([
                mean_squared_error(X_sub[i, :], reconstructed[i, :]) 
                for i in range(X_sub.shape[0])
            ])

            # rank order:
            ordered = X_sub[np.argsort(mse), :]  # order asc by reconstr error  # todo, use X_sub or reconstructed?
            sample_count = target_count - X_sub.shape[0]  # todo: redo this--pull a subset of these..
            sample = ordered[:sample_count]  # the interpolation sample

            # jitter the sample. We create a random matrix, M x N, (bound in [0, 1]),
            # subtract 0.5 (so there are some negatives), multiply by the columnar standard
            # deviations, and add to the sample.
            sample += (jitter * (sample.std(axis=0) * (random_state.rand(*sample.shape) - 0.5)))

            # transform the sample batch, and the output is the interpolated minority sample
            # interpolated = encoder.transform(sample)

            # append to X, y
            X = np.vstack([X, sample])  # was `interpolated` instead of sample, before
            y_transform = np.concatenate([y_transform, np.ones(sample.shape[0]) * transformed_label])

            # determine whether we need to recurse for this class (if there were too few samples)
            if sample.shape[0] + X_sub.shape[0] >= target_count:
                break

    # now that X, y_transform have been assembled, inverse_transform the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    if shuffle:
        output_order = random_state.shuffle(np.arange(X.shape[0]))
    else:
        output_order = np.arange(X.shape[0])

    if return_encoders:
        return X[output_order, :], y[output_order], encoders
    return X[output_order, :], y[output_order]
