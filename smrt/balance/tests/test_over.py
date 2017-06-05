# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the over-sampling balancer

from __future__ import division, absolute_import, division
from smrt.testing import load_imbalanced_mnist
from sklearn.datasets import load_iris
from smrt.balance import over_sample_balance
import numpy as np


def test_over_simple():
    iris = load_iris()
    X, y = iris.data, iris.target

    # create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
    indices = [i for i in range(50)] + [i for i in range(50, 60)] + [i for i in range(100, 120)]
    X, y = X[indices, :], y[indices]

    # fit with over-sampling
    X_bal, y_bal = over_sample_balance(X, y, balance_ratio=1.0, random_state=42)
    assert X_bal.shape[0] == 150

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)


def test_over_mnist():
    X, y = load_imbalanced_mnist(majority_label=0, minority_label=1, majority_size=None, minority_size=500)
    assert X.shape[0] == y.shape[0] == 5944, '%i' % X.shape[0]

    X_bal, y_bal = over_sample_balance(X, y, random_state=42, balance_ratio=0.5)

    # should be equal lengths
    assert X_bal.shape[0] == y_bal.shape[0] == 8166, 'Expected equal lengths of 8166,' \
                                                     'but got len(X)==%i, len(y)==%i' \
                                                     % (X_bal.shape[0], y_bal.shape[0])

    assert y_bal[y_bal == 1].shape[0] == X_bal[y_bal == 1].shape[0] == 2722
    assert y_bal[y_bal == 0].shape[0] == X_bal[y_bal == 0].shape[0] == 5444
