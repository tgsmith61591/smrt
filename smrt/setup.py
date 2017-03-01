# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup for submodules of smrt

from __future__ import absolute_import
import os


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('smrt', parent_package, top_path)

    # modules
    config.add_subpackage('autoencode')
    config.add_subpackage('balance')

    # module tests -- must be added after others!
    config.add_subpackage('autoencode/tests')
    config.add_subpackage('balance/tests')

    # misc repo tests
    config.add_subpackage('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
