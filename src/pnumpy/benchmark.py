import os
import sys

__all__ = [
    'benchmark','benchmark_func']

from pnumpy._pnumpy import atop_enable, atop_disable, atop_isenabled 
from pnumpy._pnumpy import thread_enable, thread_disable, thread_isenabled
from pnumpy._pnumpy import timer_gettsc, timer_getutc

import numpy as np

# TODO: move this to new location
def benchmark_timeit(
    func=np.equal,
    ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64],
    scalar=False,
    unary = False,
    reduct = False,
    outdtype=None,
    recycle=True,
    sizes=[1_000_000]):

    timedelta = np.zeros(len(ctypes), np.int64)
    
    for s in sizes:
        slot = 0
        loop_size = 100
        mtimedelta = np.zeros(loop_size, np.int64)
        for c in ctypes:
            if c is np.bool:
               a=np.arange(s, dtype=np.int8).astype(c)+1
            else:
               a=np.arange(s, dtype=c)
               a=a % 253
               a+=1

            if scalar is True:
                b=a[5]
            else:
                b=a.copy()

            c = None
            if recycle:
                if outdtype is None:
                    c=np.ones(s, dtype=c)
                else:
                    c=np.ones(s, dtype=outdtype)

            for loop in range(loop_size):
                if unary is False:
                    starttime = timer_gettsc()
                    func(a,b,out=c)
                    delta= timer_gettsc() - starttime
                else:
                    if reduct is True:
                        starttime = timer_gettsc()
                        func(a)
                        delta= timer_gettsc() - starttime
                    else:
                        starttime = timer_gettsc()
                        func(a,out=c)
                        delta= timer_gettsc() - starttime

                mtimedelta[loop] = delta
            # skip first
            timedelta[slot] = np.median(mtimedelta[1:])
            # print("median is ", timedelta[slot], slot)
            slot = slot + 1
    return timedelta


def benchmark_func(
    func=np.equal,
    ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64],
    scalar=False,
    unary = False,
    reduct = False,
    outdtype=None,
    recycle=True,
    atop=True,
    thread=True,
    sizes=[1_000_000]):

    # disable atop and threading
    atop_disable()
    thread_disable()
    # get original time
    t0=benchmark_timeit(func=func, ctypes=ctypes, scalar=scalar, unary=unary, reduct=reduct,  outdtype=outdtype, recycle=recycle, sizes=sizes)

    # now possibly enable atop and threading
    if atop:
        atop_enable()
    if thread:
        thread_enable()
    t1=benchmark_timeit(func=func, ctypes=ctypes, scalar=scalar, unary=unary, reduct=reduct, outdtype=outdtype, recycle=recycle, sizes=sizes)
    return t0/t1

def benchmark(
    ctypes=[np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64],
    recycle=True,
    atop=True,
    thread=True,
    sizes=[1_000_000]):
    '''
    Inputs
    ------

    Returns
    -------

    Examples
    --------
    '''

    def ctype_string(ct):
        s=f'{sizes[0]} rows,'
        for i in ct:
            s=s+f'{i.__name__},'
        return s

    def output_data(rowname, data):
        s=f'{rowname},'
        for i in data:
            s=s+f'{i:5.2f},'
        print(s)

    print(ctype_string(ctypes))
    output_data("a==b ", benchmark_func(np.equal, ctypes=ctypes, scalar=False, unary=False, recycle=recycle, atop=atop, thread=thread,  outdtype='?', sizes=sizes))
    output_data("a==5 ", benchmark_func(np.equal, ctypes=ctypes, scalar=True, unary=False, recycle=recycle, atop=atop, thread=thread, outdtype='?', sizes=sizes))
    output_data("a+b", benchmark_func(np.add, ctypes=ctypes, scalar=False, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("a+5", benchmark_func(np.add, ctypes=ctypes, scalar=True, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("a/5", benchmark_func(np.true_divide, ctypes=ctypes, scalar=True, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("abs", benchmark_func(np.abs, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("isnan", benchmark_func(np.isnan, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("sin",   benchmark_func(np.sin, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("log",   benchmark_func(np.log, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("sum",   benchmark_func(np.sum, ctypes=ctypes, scalar=False, reduct=True, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("min",   benchmark_func(np.min, ctypes=ctypes, scalar=False, reduct=True, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
