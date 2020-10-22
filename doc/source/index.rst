.. fast_numpy_loops documentation master file, created by
   sphinx-quickstart on Thu Oct 22 12:01:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fast_numpy_loops's documentation!
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

    $ pip install fast_numpy_loops
    # or
    $ conda install fast_numpy_loops

To use the package one it is installed::

    >>> import fast_numpy_loops
    >>> fast_numpy_loops.initialize()

How to use it?
--------------

.. automodule:: fast_numpy_loops
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
