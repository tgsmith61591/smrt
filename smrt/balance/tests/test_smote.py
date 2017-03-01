# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMOTE balancer

from __future__ import division, absolute_import, division
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from sklearn.datasets import load_iris
from smrt.balance import smote_balance
import numpy as np


def test_smote_simple():
    iris = load_iris()
    X, y = iris.data, iris.target

    # create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
    indices = [i for i in range(50)] + [i for i in range(50,60)] + [i for i in range(100, 120)]
    X, y = X[indices, :], y[indices]

    # fit with smote
    X_bal, y_bal = smote_balance(X, y, balance_ratio=1.0, seed=42, shuffle=False)
    assert X_bal.shape[0] == 150

    # we didn't shuffle so we can assert the first 80 rows are all the same as they were
    assert_array_almost_equal(X_bal[:80, :], X[:80, :])

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)
