# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test random corner cases for both balancers

from __future__ import division, absolute_import, division
from sklearn.utils.multiclass import type_of_target
from smrt.balance import base, smote_balance
from nose.tools import assert_raises
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target


def test_label_corner_cases():
    # the current max classes is 100 (might change though).
    n_classes = base.MAX_N_CLASSES + 1

    # create n_classes labels, append on itself so there are at least two of each
    # so sklearn will find it as a multi-class and not a continuous target
    labels = np.arange(n_classes)
    labels = np.concatenate([labels, labels])

    # assert that it's multiclass and that we're getting the appropriate ValueError!
    y_type = type_of_target(labels)
    assert y_type == 'multiclass', y_type

    # create an X of random. Doesn't even matter.
    x = np.random.rand(labels.shape[0], 4)

    # try to balance, but it will fail because of the number of classes
    assert_raises(ValueError, smote_balance, x, labels)

    # now time for continuous...
    labels = np.linspace(0, 1000, x.shape[0])

    # fails because improper y_type
    assert_raises(ValueError, smote_balance, x, labels)

    # perform a balancing operation with only one observation, and show that it will raise
    labels = np.zeros(x.shape[0])
    labels[0] = 1  # this is the only one.
    y_type = type_of_target(labels)
    assert y_type == 'binary', y_type

    # fails because only one observation of one of the classes
    assert_raises(ValueError, smote_balance, x, labels)


def test_smote_corner_cases():
    # if n_neighbors is < 1...
    assert_raises(ValueError, smote_balance, X, y, n_neighbors=0)

    # show that a bad "strategy" is a ValueError
    assert_raises(ValueError, smote_balance, X, y, strategy='bad-input')

    # show that iris will not actually balance anything, since there is no majority class
    X_smote, y_smote = smote_balance(X, y)
    assert X_smote.shape[0] == y_smote.shape[0] == 150

    # show that whether or not shuffle is used, it will not raise an index error:
    # likewise, show that models can be returned
    for shuffle in (True, False):
        for model in (True, False):
            output = smote_balance(X, y, shuffle=shuffle, return_estimators=model)
            if model:
                assert len(output) == 3
