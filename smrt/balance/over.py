# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# The over-sampling balancer

from __future__ import division, absolute_import, division
from sklearn.preprocessing import LabelEncoder
from .base import _validate_X_y_ratio_classes
from ..utils import get_random_state, DEFAULT_SEED
from . import base
import numpy as np

__all__ = [
    'over_sample_balance'
]


def over_sample_balance(X, y, balance_ratio=0.2, random_state=DEFAULT_SEED):
    """One strategy for balancing data is to over-sample the minority class until it is
    represented at the prescribed ``balance_ratio``. While there is significant literature
    to show that this is not the best technique, and can sometimes lead to over-fitting, there
    are instances wherein it works quite well.

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

    random_state : int or None, optional (default=None)
        The seed to construct the random state to generate random selections.
    """
    random_state = get_random_state(random_state).state

    # validate before copying arrays around...
    X, y, n_classes, present_classes, \
        counts, majority_label, target_count = _validate_X_y_ratio_classes(X, y, balance_ratio)

    # encode y, in case they are not numeric (we need them to be for np.ones)
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric

    # we'll vstack/concatenate to these
    out_X, out_y = X.copy(), y_transform.copy()

    # iterate the present classes
    for label in present_classes:
        if label == majority_label:
            continue

        # get the transformed label
        label_transform = le.transform([label])[0]

        while True:
            # use the out_X, out_y copies. Since we're oversamping,
            # it doesn't matter if we're drawing from the out_X matrix.
            # also, this way we can better keep track of how many we've drawn.
            mask = out_y == label_transform
            n_req = target_count - mask.sum()

            # terminal case
            if n_req == 0:
                break

            # draw a sample, take first n_req:
            idcs = np.arange(out_X.shape[0])[mask]  # get the indices, mask them
            sample = out_X[random_state.permutation(idcs), :][:n_req]

            # vstack
            out_X = np.vstack([out_X, sample])

            # concatenate. Use sample length, since it might be < n_req
            out_y = np.concatenate([out_y, np.ones(sample.shape[0], dtype=np.int16) * label_transform])

    return out_X, le.inverse_transform(out_y)
