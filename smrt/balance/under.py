# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# The under-sampling balancer

from __future__ import division, absolute_import, division
from .base import _validate_X_y_ratio_classes
from ..utils import get_random_state, DEFAULT_SEED
import numpy as np

__all__ = [
    'under_sample_balance'
]


def _reorder(X, y, random_state, shuffle):
    # reorder if needed
    order = np.arange(X.shape[0])
    if shuffle:
        order = random_state.permutation(order)
    return X[order, :], y[order]


def under_sample_balance(X, y, balance_ratio=0.2, random_state=DEFAULT_SEED, shuffle=True):
    """One strategy for balancing data is to under-sample the majority class until it is
    represented at the prescribed ``balance_ratio``. This can be effective in cases where the
    training set is already quite large, and diminishing its size may not prove detrimental.

    The under-sampling procedure behaves differently than the other (over-sampling) techniques
    in ``smrt``: its objective is only to under-sample the *majority* class, and will down-sample
    it until the *second-most* represented class is present at the prescribed ratio.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_features`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random selections.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.
    """
    random_state = get_random_state(random_state).state

    # validate before copying arrays around...
    X, y, n_classes, present_classes, \
        counts, majority_label, _ = _validate_X_y_ratio_classes(X, y, balance_ratio)

    # get the second-most populous count, compute target
    sorted_counts = np.sort(counts)
    if sorted_counts[-1] == sorted_counts[-2]:  # corner case
        return _reorder(X, y, random_state, shuffle)

    target_count = max(int(sorted_counts[-2] / balance_ratio), 1)

    # select which rows gotta go...
    idcs = np.arange(X.shape[0])
    mask = y == majority_label
    remove = random_state.permutation(idcs[mask])[:mask.sum() - target_count]  # mask.sum() is GREATER than target count

    # remove them
    X = np.delete(X, remove, axis=0)
    y = np.delete(y, remove)

    # reorder if needed
    return _reorder(X, y, random_state, shuffle)
