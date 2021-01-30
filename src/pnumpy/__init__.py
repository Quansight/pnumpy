"""
Pnumpy now calls ``init`` at startup to setup the package. This imports NumPy,
replacing all the inner loops of UFuncs with wrapped versions. Then you can
enable/disable any of the subsystems:

  - threading

    Threading will kick in when the number of elements to be processed is more
    than 50,000. It will break the operation into chunks. Each chunk will be
    executed in its own thread.

  - ledger

    The ledger records data on each loop execution to
    enable more accurate heuristics on memory allocation, threading behavior
    and reporting for logging and benchmarking.

  - recycler

    Once we can change the NumPy memory allocation strategy, we can use the
    data from the ledger to create more performant memory caches.

  - atop

    Provide faster implementations of NumPy inner loops.
"""
from ._version import __version__
__all__ = [
    'initialize', 'atop_enable', 'atop_disable', 'atop_isenabled', 'atop_info', 'atop_setworkers','cpustring',
    'thread_enable', 'thread_disable', 'thread_isenabled', 'thread_getworkers', 'thread_setworkers', 'thread_zigzag',
    'ledger_enable', 'ledger_disable', 'ledger_isenabled', 'ledger_info',
    'recycler_enable', 'recycler_disable', 'recycler_isenabled', 'recycler_info',
    'timer_gettsc','timer_getutc', 'benchmark']

import numpy as np
import numpy.core._multiarray_umath as umath

# TODO check for Apple M1 chip (where AVX2 makes no sense)
# TODO check for numpy version

try:
    # Numpy 1.18 does not have __cpu_features
    # If we cannot find it, we load anyway because 95% have AVX2
    # and we can hook numpy 1.18 ufuncs
    # TODO: check for Apple M1 chip
    __hasavx2 = umath.__cpu_features__['AVX2']
except Exception:
    __hasavx2 = True

if not __hasavx2:
    raise ValueError(f"PNumPy requires a CPU with AVX2 capability to work")

del __hasavx2

import pnumpy._pnumpy as _pnumpy
from pnumpy._pnumpy import atop_enable, atop_disable, atop_isenabled, atop_info, atop_setworkers, cpustring 
from pnumpy._pnumpy import thread_enable, thread_disable, thread_isenabled, thread_getworkers, thread_setworkers, thread_zigzag
from pnumpy._pnumpy import timer_gettsc, timer_getutc
from pnumpy._pnumpy import ledger_enable, ledger_disable, ledger_isenabled, ledger_info
from pnumpy._pnumpy import recycler_enable, recycler_disable, recycler_isenabled, recycler_info
from pnumpy._pnumpy import getitem, lexsort32, lexsort64

from .cpu import cpu_count_linux, init, enable, disable
from .sort import sort, lexsort, argsort, argmin, argmax, searchsorted
from .benchmark import benchmark, benchmark_func
from .recarray import recarray_to_colmajor

# to be removed
def initialize():
    init()

# start the engine by default
# TODO: check environment variable
init()
