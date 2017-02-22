# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMOTE balancer

from __future__ import division, absolute_import, division
from tensorflow.examples.tutorials.mnist import input_data
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from smrt import smrt_balance, AutoEncoder
import numpy as np


def test_smote():
    pass
