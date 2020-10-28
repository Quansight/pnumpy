..  -*- rst -*-

==========
Benchmarks
==========

This package uses `Airspeed Velocity`_ for benchmarks. The benchmarks are adapted
from the ones in the `NumPy github repo`_


Usage
-----

Airspeed Velocity manages building and Python virtualenvs by itself.

Before beginning, ensure that *airspeed velocity* is installed.
By default, `asv` ships with support for anaconda and virtualenv::

    pip install asv
    pip install virtualenv

After contributing new benchmarks, you should test them locally
before submitting a pull request.

To run all benchmarks, navigate to the top-level repo directory via the command
line and execute::

    asv run

The first time this is run, it will build a profile of the machine. The
information is stored in a top-level `.asv` directory that will be ignored by
git.  (Note: running benchmarks could take a while. Each benchmark is run
multiple times to measure the distribution in execution times.)

To run benchmarks across a series of git commits, `asv` supports git-like
syntax. For example to run benchmarks on all commits on a branch off `main`,
do::
    asv run main..mybranch

To view benchmarks once run, use ``asv show <commit>``::

    asv show main

This will display the results in plain text in the console. For a graphical
view, you can create html via ``asv publish`` and then view the result with
``asv preview``.

More on how to use ``asv`` can be found in `ASV documentation`_
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.

.. _ASV documentation: https://asv.readthedocs.io/


Writing benchmarks
------------------

See `Airspeed Velocity`_ documentation for basics on how to write benchmarks.

Some things to consider:

- The benchmark suite should be importable with any version of the project.

- The benchmark parameters etc. should not depend on which version is
  installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's ``time_`` methods for benchmarking times rather than cooking up
  time measurements via ``time.clock``, even if it requires some juggling when
  writing the benchmark.

- Preparing arrays etc. should generally be put in the ``setup`` method rather
  than the ``time_`` methods, to avoid counting preparation time together with
  the time of the benchmarked operation.

- Be mindful that large arrays created with ``np.empty`` or ``np.zeros`` might
  not be allocated in physical memory until the memory is accessed. If this is
  desired behaviour, make sure to comment it in your setup function. If
  you are benchmarking an algorithm, it is unlikely that a user will be
  executing said algorithm on a newly created empty/zero array. One can force
  pagefaults to occur in the setup phase either by calling ``np.ones`` or
  ``arr.fill(value)`` after creating the array,

.. _`Airspeed Velocity`: https://asv.readthedocs.io/
.. _`NumPy github repo`: https://github.com/numpy/numpy
