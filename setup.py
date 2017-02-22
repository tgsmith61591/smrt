# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the SMRT module

from __future__ import print_function, absolute_import, division
from distutils.command.clean import clean
from setuptools import setup
import os
import sys

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# Hacky, adopted from sklearn. This sets a global variable
# so smrt __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
builtins.__SMRT_SETUP__ = True

# metadata
DISTNAME = 'smrt'
DESCRIPTION = 'Handle class imbalance intelligently by using autoencoders ' \
              'to generate synthetic observations of your minority class.'

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'new BSD'

# import restricted version
import smrt
VERSION = smrt.__version__

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = req.read().split(os.linesep)


# Custom clean command to remove build artifacts -- adopted from sklearn
class CleanCommand(clean):
    description = "Remove build artifacts from the source tree"

    # this is mostly in case we ever add a Cython module to SMRT
    def run(self):
        clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            cython_hash_file = os.path.join(cwd, 'cythonize.dat')
            if os.path.exists(cython_hash_file):
                os.unlink(cython_hash_file)
            print('Will remove generated .c & .so files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(DISTNAME):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    print('Removing file: %s' % filename)
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            # this is for FORTRAN modules, which some of my other packages have used in the past...
            for dirname in dirnames:
                if dirname == '__pycache__' or dirname.endswith('.so.dSYM'):
                    print('Removing directory: %s' % dirname)
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

# setup the config
configuration = dict(name=DISTNAME,
                     maintainer=MAINTAINER,
                     maintainer_email=MAINTAINER_EMAIL,
                     description=DESCRIPTION,
                     license=LICENSE,
                     version=VERSION,
                     classifiers=['Intended Audience :: Science/Research',
                                  'Intended Audience :: Developers',
                                  'Intended Audience :: Scikit-learn users',
                                  'Programming Language :: Python',
                                  'Topic :: Machine Learning',
                                  'Topic :: Software Development',
                                  'Topic :: Scientific/Engineering',
                                  'Operating System :: Microsoft :: Windows',
                                  'Operating System :: POSIX',
                                  'Operating System :: Unix',
                                  'Operating System :: MacOS',
                                  'Programming Language :: Python :: 2.7'
                                 ],
                     keywords='sklearn scikit-learn tensorflow scikit-neuralnetwork auto-encoders class-imbalance',
                     packages=[DISTNAME],
                     install_requires=REQUIREMENTS,
                     cmdclass=cmdclass)


def do_setup():
    return setup(**configuration)


if __name__ == '__main__':
    do_setup()
