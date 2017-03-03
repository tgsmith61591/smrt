# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utilities for testing SMRT & SMOTE

from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

__all__ = [
    'load_imbalanced_mnist'
]


def load_imbalanced_mnist(majority_label=0, minority_label=1, majority_size=None, minority_size=500):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    all_data = np.asarray(mnist.train.images)

    # they're one-hot encoded right now. Flatten into a single vector
    labels = np.asarray([np.argmax(row) for row in mnist.train.labels])

    # create masks
    min_mask = labels == minority_label
    maj_mask = labels == majority_label

    # get labels
    y_min = labels[min_mask][:minority_size]
    y_maj = labels[maj_mask][:majority_size]

    # get images
    X_min = all_data[min_mask, :][:minority_size]
    X_maj = all_data[maj_mask, :][:majority_size]

    X = np.vstack([X_min, X_maj])
    y = np.concatenate([y_min, y_maj])

    return X, y
