# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utils for SMRT

from __future__ import absolute_import, division, print_function
from numpy.random import RandomState
import sys

__all__ = [
    'get_random_state',
    'overrides',
    'validate_float'
]

if sys.version_info[0] >= 3:
    long = int


def get_random_state(random_state):
    """Get a ``numpy.random.RandomState`` PRNG given a seed or
    an existing ``RandomState``.
    Parameters
    ----------
    random_state : ``RandomState``, int or None
        The seed or PRNG.
    """
    if random_state is None:
        return RandomState()
    elif isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, (int, long)):
        return RandomState(random_state)
    else:
        raise ValueError('Cannot seed RandomState given class=%s' % type(random_state))


def validate_float(ratio, name, upper_bound=1., gtet=True):
    res = 0. < ratio < upper_bound if not gtet else 0. < ratio <= upper_bound
    if not res:
        raise ValueError('Expected 0 < %s %s %r, but got %r'
                         % (name, ratio, '<=' if gtet else '<', upper_bound))


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
