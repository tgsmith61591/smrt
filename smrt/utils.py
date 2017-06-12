# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utils for SMRT

from __future__ import absolute_import, division, print_function
from numpy.random import RandomState as _RandomState
import tensorflow as tf
import numpy as np
import sys

__all__ = [
    'SeededRandomState',
    'overrides',
    'validate_float'
]

# variables we'll use for a lot of stuff...
DTYPE = tf.float32
NPDTYPE = np.float64
DEFAULT_L2 = 0.0001
DEFAULT_DROPOUT = 1.
DEFAULT_SEED = 42

if sys.version_info[0] >= 3:
    long = int


class SeededRandomState(object):
    """Class that wraps ``numpy.random.RandomState``. This
    class preserves the seed that created the state, unlike
    the numpy variant.

    Parameters
    ----------
    seed : int, optional (default=42)
        The seed value.
    """
    # todo make picklable
    def __init__(self, seed=DEFAULT_SEED):
        if seed is None:
            seed = DEFAULT_SEED

        # if it's a numpy random state, get the embedded values
        if isinstance(seed, _RandomState):
            seed = seed.get_state()[1][0]

        if not isinstance(seed, (int, np.int, long)):
            raise TypeError('seed must be an int, but got type=%s' % type(seed))

        self.rs = _RandomState(seed)
        self.seed_value = seed

    @property
    def state(self):
        return self.rs


def get_random_state(seed):
    """Get the :class:`SeededRandomState` instance.

    Parameters
    ----------
    seed : int, optional (default=42)
        The seed value.
    """
    if isinstance(seed, SeededRandomState):
        return seed
    return SeededRandomState(seed)


def validate_float(ratio, name, upper_bound=1., ltet=True):
    res = 0. < ratio < upper_bound if not ltet else 0. < ratio <= upper_bound
    if not res:
        raise ValueError('Expected 0 < %s %s %r, but got %r'
                         % (name, ratio, '<=' if ltet else '<', upper_bound))


def overrides(interface_class):
    """Decorator for methods that override super methods. Provides
    runtime validation that the method is, in fact, inherited from the
    superclass. If not, will raise an ``AssertionError``. This decorator
    is adapted from skutil: http://github.com/tgsmith/skutil. In an effort
    not to add numerous dependencies, it's just copied here for use with
    smrt.

    Parameters
    ----------
    interface_class: Class
        The super class that will have its method overwritten.

    Examples
    --------
    The following is valid use:
        >>> class A(object):
        ...     def a(self):
        ...         return 1
        >>> class B(A):
        ...     @overrides(A)
        ...     def a(self):
        ...         return 2
        ...
        ...     def b(self):
        ...         return 0

    The following would be an invalid ``overrides`` statement, since
    ``A`` does not have a ``b`` method to override.
        >>> class C(B): # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     @overrides(A) # should override B, not A
        ...     def b(self):
        ...         return 1
        Traceback (most recent call last):
        AssertionError: A.b must override a super method!
    """

    def overrider(method):
        assert (method.__name__ in dir(interface_class)), '%s.%s must override a super method!' % (
            interface_class.__name__, method.__name__)
        return method
    return overrider
