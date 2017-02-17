# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The SMITE module

import sys
import os

__version__ = '0.1'

try:
	# this var is injected in the setup build to enable
	# the retrieval of the version number without actually
	# importing the un-built submodules.
	__SMITE_SETUP__
except NameError:
	__SMITE_SETUP__ = False

if __SMITE_SETUP__:
	sys.stderr.write('Partial import of SMITE during the build process.' + os.linesep)
else:
	__all__ = [

	]
