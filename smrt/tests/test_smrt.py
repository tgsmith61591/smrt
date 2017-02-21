# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMRT balancer and autoencoder

from __future__ import division, absolute_import, division
from tensorflow.examples.tutorials.mnist import input_data
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from smrt import smrt_balance, AutoEncoder
import numpy as np


def test_autoencoder():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    all_data = np.asarray(mnist.train.images)

    seed = 42
    X_train, X_test = train_test_split(all_data, train_size=0.7, random_state=seed)

    # define
    ae = AutoEncoder(n_epochs=10, learning_rate=0.01, batch_size=256,
                     display_step=5, activation_function='sigmoid',
                     verbose=2, seed=seed)

    # fit
    ae.fit(X_train)

    # transform and reconstruct the test images
    reconstructed = ae.feed_forward(X_test)

    # get the error:
    mse = ((X_test - reconstructed) ** 2).sum(axis=1).sum() / X_test.shape[0]
    assert_almost_equal(mse, 4.40549573864)
