
from __future__ import division, absolute_import, division
from smrt.balance import smrt_balance
from sklearn.datasets import load_iris


def test_iris():
    iris = load_iris()
    X, y = iris.data, iris.target

    # undersample 1, 2
    indices = [i for i in range(50)] + [i for i in range(50,60)] + [i for i in range(100, 120)]
    X, y = X[indices, :], y[indices]
    X, y, encoders = smrt_balance(X, y, return_encoders=True, balance_ratio=1.0)
    assert X.shape[0] == 150, 'Expected 150, got %i' % X.shape[0]
