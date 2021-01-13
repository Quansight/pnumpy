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
    '''
    Internal routine to benchmark a function.

    '''

    def time_func(recycle, c):
        if unary is False:
            starttime = timer_gettsc()
            if recycle:
                result=func(a,b,out=c)
            else:
                result=func(a,b)
            delta= timer_gettsc() - starttime

        else:
            starttime = timer_gettsc()
            if reduct is True:
                result=func(a)
            else:
                if recycle:
                    result=func(a,out=c)
                else:
                    result=func(a)

            delta= timer_gettsc() - starttime
        return delta, result

    timedelta = np.zeros(len(ctypes), np.int64)
    
    for s in sizes:
        slot = 0
        loop_size = 100
        mtimedelta = np.zeros(loop_size, np.int64)
        for ctype in ctypes:
            if ctype is np.bool:
               a=np.arange(s, dtype=np.int8).astype(ctype)+1
            else:
               a=np.arange(s, dtype=ctype)
               a=a % 253
               a+=1

            if scalar is True:
                b=a[5]
            else:
                b=a.copy()

            # dry run
            delta, c=time_func(False, None)

            # main timing loop
            for loop in range(loop_size):
                delta, result = time_func(recycle, c)
                del result

                mtimedelta[loop] = delta
            
            timedelta[slot] = np.median(mtimedelta)
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
    '''
    Benchmark one function.

    Examples
    --------
    benchmark_func(np.add)
    benchmark_func(np.add, sizes=[2**16])
    benchmark_func(np.sqrt, unary=True)
    '''
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
    Performs a simple benchmark of the ratio of normal numpy (no threading) vs parallel numpy (threaded).
    The output is formatted to be copied and pasted in a csv file.
    A result above 1.0 indicates an improvement, below 1.0 indicates worse peformance.

    Inputs
    ------

    Returns
    -------
    output text formatted for a .csv file

    Examples
    --------
    pn.benchmark()
    pn.benchmark(thread=False)
    pn.benchmark(sizes=[2**16])
    pn.benchmark(ctypes=[np.float32, np.float64])
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
    output_data("a==b", benchmark_func(np.equal, ctypes=ctypes, scalar=False, unary=False, recycle=recycle, atop=atop, thread=thread,  outdtype='?', sizes=sizes))
    output_data("a==5", benchmark_func(np.equal, ctypes=ctypes, scalar=True, unary=False, recycle=recycle, atop=atop, thread=thread, outdtype='?', sizes=sizes))
    output_data("a+b", benchmark_func(np.add, ctypes=ctypes, scalar=False, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("a+5", benchmark_func(np.add, ctypes=ctypes, scalar=True, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("a/5", benchmark_func(np.true_divide, ctypes=ctypes, scalar=True, unary=False,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("abs", benchmark_func(np.abs, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("isnan", benchmark_func(np.isnan, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("sin",   benchmark_func(np.sin, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("log",   benchmark_func(np.log, ctypes=ctypes, scalar=False, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("sum",   benchmark_func(np.sum, ctypes=ctypes, scalar=False, reduct=True, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
    output_data("min",   benchmark_func(np.min, ctypes=ctypes, scalar=False, reduct=True, unary=True,recycle=recycle, atop=atop, thread=thread, sizes=sizes))
