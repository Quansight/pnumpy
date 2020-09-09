========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |commits-since|

.. |travis| image:: https://api.travis-ci.org/mattip/numpy-threading-extensions.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/mattip/numpy-threading-extensions

.. |codecov| image:: https://codecov.io/gh/mattip/numpy-threading-extensions/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/mattip/numpy-threading-extensions

.. |commits-since| image:: https://img.shields.io/github/commits-since/mattip/numpy-threading-extensions/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/mattip/numpy-threading-extensions/compare/v0.0.0...master



.. end-badges

Faster loops for NumPy using multithreading and other tricks

* Free software: MIT license

Installation
============

::

    pip install fast-numpy-loops

You can also install the in-development version with::

    pip install https://github.com/mattip/numpy-threading-extensions/archive/master.zip


Documentation
=============


To use the project:

.. code-block:: python

    import fast-numpy-loops
    fast-numpy-loops.initialize()


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
