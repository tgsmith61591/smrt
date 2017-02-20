# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMRT module

import sys
import os

__version__ = '0.1'

try:
    # this var is injected in the setup build to enable
    # the retrieval of the version number without actually
    # importing the un-built submodules.
    __SMRT_SETUP__
except NameError:
    __SMRT_SETUP__ = False

if __SMRT_SETUP__:
    sys.stderr.write('Partial import of SMRT during the build process.' + os.linesep)
else:
    __all__ = [
        'autoencode.py',
        'balance'
    ]

    # top-level imports
    from .balance import smrt_balance
    from .autoencode import AutoEncoder
