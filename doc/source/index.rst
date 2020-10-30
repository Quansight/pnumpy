.. accelerated_numpy documentation master file, created by
   sphinx-quickstart on Thu Oct 22 12:01:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to accelerated_numpy's documentation!
============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Faster loops for NumPy using multithreading and other tricks. The first release
will target NumPy binary and unary ufuncs. Eventually we will enable overriding
other NumPy functions, and provide an C-based (non-Python) API for extending
via third-party functions.

Installation
------------

This is a binary package and requires compilation. We recommend using pip or
conda to obtain a pre-built version::

    $ pip install accelerated_numpy
    # or
    $ conda install accelerated_numpy

To use the package one it is installed::

    >>> import accelerated_numpy
    >>> accelerated_numpy.initialize()

How to use it?
--------------

.. automodule:: accelerated_numpy
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
