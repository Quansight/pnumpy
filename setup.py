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

try:
    from setuptools_scm import get_version
except Exception:
    try:
        import pip
        package='setuptools_scm'
        if hasattr(pip, 'main'):
            pip.main(['install', package])
        else:
            pip._internal.main(['install', package])
        from setuptools_scm import get_version
    except Exception:
        print("**could not install pip or setuptools_scm, version is defaulted")

def myversion():
    try:
        #version = get_version()
        version = '2.0.11'
    except Exception:
        version = '2.0.11'
    return version

thisversion=myversion()
#thisversion = '.'.join(thisversion.split('.')[:3])

def writeversion():
    text_file = open("src/pnumpy/_version.py", "w")
    strver = f"__version__='{thisversion}'"
    n = text_file.write(strver)
    text_file.close()
    return thisversion

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUP_PY_EXT_COVERAGE after deps have been safely installed).
if os.environ.get('SETUP_PY_EXT_COVERAGE') == 'yes' and platform.system() == 'Linux':
    CFLAGS = os.environ['CFLAGS'] = '-fprofile-arcs -ftest-coverage -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
    LFLAGS = os.environ['LFLAGS'] = '-lgcov'
else:
    CFLAGS = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
    LFLAGS = ''

if platform.system() == 'Windows':
    CFLAGS += ' /Ox /Ob2 /Oi /Ot /d2FH4- /GS- /arch:AVX2'
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
    name='pnumpy',
    #version=get_git_version(), #'0.0.0',
    version=writeversion(),
    license='MIT',
    description='Faster loops for NumPy using multithreading and other tricks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Quansight',
    author_email='info@quansight.com',
    url='https://quansight.github.io/numpy-threading-extensions/stable/index.html',
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
    #setup_requires=['setuptools_scm'],
    #use_scm_version = {
    #    'version_scheme': 'post-release',
    #    'local_scheme': 'no-local-version',
    #    'write_to': 'src/pnumpy/_version.py',
    #    'write_to_template': '__version__ = "{version}"',
    #},
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
            'pnumpy._pnumpy',
            sources=['src/pnumpy/_pnumpy.cpp',
                     'src/pnumpy/module_init.cpp',
                     'src/pnumpy/common.cpp',
                     'src/pnumpy/ledger.cpp',
                     'src/pnumpy/getitem.cpp',
                     'src/pnumpy/conversions.cpp',
                     'src/pnumpy/recycler.cpp',
                     'src/pnumpy/sorting.cpp',
                     'src/pnumpy/arange.cpp',
                     'src/pnumpy/item_selection.cpp',
                     'src/atop/atop.cpp',
                     'src/atop/threads.cpp',
                     'src/atop/recarray.cpp',
                     'src/atop/sort.cpp',
                     'src/atop/fill.cpp',
                     'src/atop/ops_binary.cpp',
                     'src/atop/ops_compare.cpp',
                     'src/atop/ops_unary.cpp',
                     'src/atop/ops_trig.cpp',
                     'src/atop/ops_log.cpp',
                    ],
            extra_compile_args=CFLAGS.split(),
            extra_link_args=LFLAGS.split(),
            include_dirs=['src/pnumpy', 'src/atop', np.get_include()],
            py_limited_api=True,
        )
    ],
)
