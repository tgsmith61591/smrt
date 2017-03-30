# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the under-sampling balancer

from __future__ import division, absolute_import, division
from sklearn.datasets import load_iris
from smrt.balance import under_sample_balance
import numpy as np


def test_under_simple():
    iris = load_iris()
    X, y = iris.data, iris.target

    # create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
    indices = [i for i in range(50)] + [i for i in range(50, 60)] + [i for i in range(100, 120)]
    X, y = X[indices, :], y[indices]

    # fit with under-sampling - this strips the most-populous class down to 20
    X_bal, y_bal = under_sample_balance(X, y, balance_ratio=1.0, random_state=42, shuffle=False)
    assert X_bal.shape[0] == 50

    # assert 50 of each label
    labels, counts = np.unique(y_bal, return_counts=True)
    assert counts[labels == 0][0] == 20
    assert counts[labels == 1][0] == 10
    assert counts[labels == 2][0] == 20
