"""
Override all the NumPy UFunc loops with multithreaded variants. The original
loop will be called via a pool of threads.
"""
__version__ = '0.1.0'
__all__ = [
    'initialize', 'cpustring',
    'thread_enable', 'thread_disable', 'thread_isenabled', 'thread_getworkers', 'thread_setworkers',
    'ledger_enable', 'ledger_disable', 'ledger_isenabled', 'ledger_info',
    'recycler_enable', 'recycler_disable', 'recycler_isenabled', 'recycler_info',
    'timer_gettsc','timer_getutc']

from fast_numpy_loops._fast_numpy_loops import initialize, cpustring 
from fast_numpy_loops._fast_numpy_loops import thread_enable, thread_disable, thread_isenabled, thread_getworkers, thread_setworkers

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

    # fa.atop_disable()
    fa.thread_disable()
    # get original time
    t0=debug_timeit(func=func, ctypes=ctypes, scalar=scalar, unary=unary, outdtype=outdtype, recycle=recycle)
    # if atop:
    #    fa.atop_enable()
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

