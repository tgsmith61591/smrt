# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# Commons and bases for the SMRT and SMOTE balancers

from __future__ import absolute_import
import numpy as np
from ..utils import validate_float, NPDTYPE
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array

DEFAULT_SEED = 42
MAX_N_CLASSES = 100  # max unique classes in y
MIN_N_SAMPLES = 2  # min allowed ever.


def _validate_X_y_ratio_classes(X, y, ratio):
    # validate the cheap stuff before copying arrays around...
    validate_float(ratio, 'balance_ratio')

    # validate arrays
    X = check_array(X, accept_sparse=False, dtype=NPDTYPE, ensure_2d=True, copy=True)
    y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)
    y = column_or_1d(y, warn=False)

    # get n classes in y, ensure they are <= MAX_N_CLASSES, but first ensure these are actually
    # class labels and not floats or anything...
    y_type = type_of_target(y)
    supported_types = {'multiclass', 'binary'}
    if y_type not in supported_types:
        raise ValueError('balancers only support %r, but got %r'
                         % ("(" + ', '.join(supported_types) + ")", y_type))

    present_classes, counts = np.unique(y, return_counts=True)
    n_classes = len(present_classes)

    # ensure <= MAX_N_CLASSES
    if n_classes > MAX_N_CLASSES:
        raise ValueError('balancers currently only support a maximum of %i '
                         'unique class labels, but %i were identified.' % (MAX_N_CLASSES, n_classes))

    # get the majority class label, and its count:
    majority_count_idx = np.argmax(counts, axis=0)
    majority_label, majority_count = present_classes[majority_count_idx], counts[majority_count_idx]
    target_count = max(int(ratio * majority_count), 1)

    # define a min_n_samples based on the sample ratio to max_class
    # required = {target_count - counts[i] for i, v in enumerate(present_classes) if v != majority_label}

    # THIS WAS OUR ORIGINAL LOGIC:
    #   * If there were any instances where the number of synthetic examples required for a class
    #     outweighed the number that existed in the class to begin with, we would end up having to
    #     potentially sample from the synthetic examples. We didn't want to have to do that.
    #
    # But it seems like a totally valid use-case. If we're detecting breast cancer, it might be a rare
    # event that needs lots of bolstering. We should allow that, even though we may discourage it.

    # if any counts < MIN_N_SAMPLES, raise:
    if any(i < MIN_N_SAMPLES for i in counts):
        raise ValueError('All label counts must be >= %i' % MIN_N_SAMPLES)

    return X, y, n_classes, present_classes, counts, majority_label, target_count
