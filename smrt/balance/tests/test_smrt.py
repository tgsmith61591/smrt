# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMRT balancer

from __future__ import division, absolute_import, division
from smrt.testing import load_imbalanced_mnist
from smrt.balance import smrt_balance


def test_smrt_mnist():
    X, y = load_imbalanced_mnist(majority_label=0, minority_label=1, majority_size=None, minority_size=500)
    assert X.shape[0] == y.shape[0] == 5944, '%i' % X.shape[0]

    X_smrt, y_smrt = smrt_balance(X, y, n_hidden=450, n_latent_factors=10, learning_rate=0.05,
                                  activation_function='sigmoid', verbose=2, display_step=25,
                                  n_epochs=50, batch_size=256, random_state=42, shuffle=False,
                                  balance_ratio=0.5, return_estimators=False)

    assert X_smrt.shape[0] == y_smrt.shape[0] == 8166
    assert y_smrt[y_smrt == 1].shape[0] == X_smrt[y_smrt == 1].shape[0] == 2722
    assert y_smrt[y_smrt == 0].shape[0] == X_smrt[y_smrt == 0].shape[0] == 5444
