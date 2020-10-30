#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import platform
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
import numpy as np

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOX_ENV_NAME' in os.environ and os.environ.get('SETUP_PY_EXT_COVERAGE') == 'yes' and platform.system() == 'Linux':
    CFLAGS = os.environ['CFLAGS'] = '-fprofile-arcs -ftest-coverage -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
    LFLAGS = os.environ['LFLAGS'] = '-lgcov'
else:
    CFLAGS = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DPy_LIMITED_API'
    LFLAGS = ''

if platform.system() == 'Windows':
    CFLAGS += ' /Ox /Ob2 /Oi /Ot /d2FH4-'
else:
    CFLAGS += ' -mavx2 -fpermissive -Wno-unused-variable -Wno-unused-function -std=c++11 -pthread -falign-functions=32'

if platform.system() == 'Linux':
    LFLAGS += ' -lm'

def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


with open("README.md") as readme:
    long_description = readme.read()

import _add_newdocs
_add_newdocs.main()

setup(
    name='accelerated-numpy',
    version='v0.0.1',
    license='MIT',
    description='Faster loops for NumPy using multithreading and other tricks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Matti Picus',
    author_email='mattigit@picus.org.il',
    url='https://github.com/Quansight/numpy-threading-extensions',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    project_urls={
        'Changelog': 'https://github.com/Quansight/numpy-threading-extensions/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/Quansight/numpy-threading-extensions/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.6',
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
        'numpy>=1.19.0',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    ext_modules=[
        Extension(
            'accelerated_numpy._accelerated_numpy',
            sources=['src/accelerated_numpy/_accelerated_numpy.cpp',
                     'src/accelerated_numpy/module_init.cpp',
                     'src/accelerated_numpy/ledger.cpp', 
                     'src/atop/atop.cpp',
                     'src/atop/threads.cpp',
                     'src/atop/ops_binary.cpp',
                     'src/atop/ops_compare.cpp',
                     'src/atop/ops_unary.cpp',
                     'src/atop/ops_trig.cpp',
                     'src/atop/ops_log.cpp',
                    ],
            extra_compile_args=CFLAGS.split(),
            extra_link_args=LFLAGS.split(),
            include_dirs=['src/accelerated_numpy', 'src/atop', np.get_include()],
            py_limited_api=True,
        )
    ],
)
