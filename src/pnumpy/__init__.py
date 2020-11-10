"""
Call ``initialize`` to setup the package. This imports and scans NumPy,
replacing all the inner loops of UFuncs with wrapped versions. Then you can
enable/disable any of the subsystems:

  - threading

    Threading will kick in when the number of elements to be processed is more
    than 50,000. It will break the operation into chunks. Each chunk will be
    executed in its own thread.
  - ledger

    The ledger (disabled in version 0.1) records data on each loop execution to
    enable more accurate heuristics on memory allocation, threading behavior
    and reporting for logging and benchmarking.
  - recycler

    Once we can change the NumPy memory allocation strategy, we can use the
    data from the ledger to create more performant memory caches.
  - atop

    NumPy is making progress with faster inner loops, but an outside package
    can iterate faster to provide even faster ones. Since the Universal SIMD
    loops are in a state of flux at this time, this is disabled for version
    0.1.


"""
__version__ = '0.0.0'
__all__ = [
    'initialize', 'atop_enable', 'atop_disable', 'atop_isenabled', 'cpustring',
    'thread_enable', 'thread_disable', 'thread_isenabled', 'thread_getworkers', 'thread_setworkers',
    'ledger_enable', 'ledger_disable', 'ledger_isenabled', 'ledger_info',
    'recycler_enable', 'recycler_disable', 'recycler_isenabled', 'recycler_info',
    'timer_gettsc','timer_getutc']

from pnumpy._pnumpy import initialize, atop_enable, atop_disable, atop_isenabled, cpustring 
from pnumpy._pnumpy import thread_enable, thread_disable, thread_isenabled, thread_getworkers, thread_setworkers
from pnumpy._pnumpy import timer_gettsc, timer_getutc
from pnumpy._pnumpy import ledger_enable, ledger_disable, ledger_isenabled, ledger_info
from pnumpy._pnumpy import recycler_enable, recycler_disable, recycler_isenabled, recycler_info

import numpy as np

# TODO: move this to new location
def debug_timeit(func=np.equal, ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64], scalar=False, unary = False, outdtype=None, recycle=True):
    timedelta = np.zeros(len(ctypes), np.int64)
    sizes=[1_000_000]
    for s in sizes:
        slot = 0
        loop_size = 100
        mtimedelta = np.zeros(loop_size, np.int64)
        for c in ctypes:
            if c is np.bool:
               a=np.arange(s, dtype=np.int8).astype(c)
            else:
               a=np.arange(s, dtype=c)
            if scalar is True:
                b=a[len(a)//2]
            else:
                b=a.copy()

            c = None
            if recycle:
                if outdtype is None:
                    c=np.zeros(s, dtype=c)
                else:
                    c=np.zeros(s, dtype=outdtype)

            for loop in range(loop_size):
                if unary is False:
                    starttime = fa.timer_gettsc()
                    func(a,b,out=c)
                    delta= fa.timer_gettsc() - starttime
                else:
                    starttime = fa.timer_gettsc()
                    func(a,out=c)
                    delta= fa.timer_gettsc() - starttime
                mtimedelta[loop] = delta
            timedelta[slot] = np.median(mtimedelta)
            slot = slot + 1
    return timedelta


def debug_benchmark(
    func=np.equal,
    ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64],
    scalar=False,
    unary = False,
    outdtype=None,
    recycle=True,
    atop=True,
    thread=True):

    fa.atop_disable()
    fa.thread_disable()
    # get original time
    t0=debug_timeit(func=func, ctypes=ctypes, scalar=scalar, unary=unary, outdtype=outdtype, recycle=recycle)
    if atop:
        fa.atop_enable()
    if thread:
        fa.thread_enable()
    t1=debug_timeit(func=func, ctypes=ctypes, scalar=scalar, unary=unary, outdtype=outdtype, recycle=recycle)
    return t0/t1

def benchmark(
    ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64],
    recycle=True,
    atop=True,
    thread=True):

    print("equal twoinp", debug_benchmark(np.equal, ctypes=ctypes, scalar=False, unary=False, recycle=recycle, atop=atop, thread=thread,  outdtype='?'))
    print("equal scalar", debug_benchmark(np.equal, ctypes=ctypes, scalar=True, unary=False, recycle=recycle, atop=atop, thread=thread, outdtype='?'))
    print("add   twoinp", debug_benchmark(np.add, ctypes=ctypes, scalar=False, unary=False,recycle=recycle, atop=atop, thread=thread))
    print("add   scalar", debug_benchmark(np.add, ctypes=ctypes, scalar=True, unary=False,recycle=recycle, atop=atop, thread=thread))
    print("abs         ", debug_benchmark(np.abs, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread))
    print("isnan       ", debug_benchmark(np.abs, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread))

