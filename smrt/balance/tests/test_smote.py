# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMOTE balancer

from __future__ import division, absolute_import, division
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from smrt.testing import load_imbalanced_mnist
from sklearn.datasets import load_iris
from smrt.balance import smote_balance
import numpy as np


def test_smote_simple():
    iris = load_iris()
    X, y = iris.data, iris.target

    # create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
    indices = [i for i in range(50)] + [i for i in range(50, 60)] + [i for i in range(100, 120)]
    X, y = X[indices, :], y[indices]

    # fit with smote
    X_bal, y_bal = smote_balance(X, y, balance_ratio=1.0, random_state=42, shuffle=False)
    assert X_bal.shape[0] == 150

    # we didn't shuffle so we can assert the first 80 rows are all the same as they were
    assert_array_almost_equal(X_bal[:80, :], X[:80, :])

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)


def test_smote_mnist():
    X, y = load_imbalanced_mnist(majority_label=0, minority_label=1, majority_size=None, minority_size=500)
    assert X.shape[0] == y.shape[0] == 5944, '%i' % X.shape[0]

    X_smote, y_smote = smote_balance(X, y, random_state=42, n_neighbors=10,
                                     shuffle=False, balance_ratio=0.5)

    assert X_smote.shape[0] == y_smote.shape[0] == 8166
    assert y_smote[y_smote == 1].shape[0] == X_smote[y_smote == 1].shape[0] == 2722
    assert y_smote[y_smote == 0].shape[0] == X_smote[y_smote == 0].shape[0] == 5444
