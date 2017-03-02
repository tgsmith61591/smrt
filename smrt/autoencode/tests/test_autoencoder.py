# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the autoencoder

from __future__ import division, absolute_import, division
from tensorflow.examples.tutorials.mnist import input_data
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from smrt.autoencode import AutoEncoder
import numpy as np


def test_autoencoder():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    all_data = np.asarray(mnist.train.images)

    seed = 42
    X_train, X_test = train_test_split(all_data, train_size=0.7, random_state=seed)

    # define
    ae = AutoEncoder(n_hidden=400, n_epochs=10, learning_rate=0.01, batch_size=256,
                     display_step=5, activation_function='sigmoid', verbose=2,
                     random_state=seed, layer_type='gaussian')

    # fit
    ae.fit(X_train)

    # train error
    assert_almost_equal(ae.train_cost_, 0.00380031)

    # transform and reconstruct the test images
    reconstructed = ae.reconstruct(X_test)

    # get the error:
    mse = ((X_test - reconstructed) ** 2).sum(axis=1).sum() / X_test.shape[0]

    # assert_almost_equal(mse, 4.40549573864)
