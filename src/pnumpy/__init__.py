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
    'initialize', 'atop_enable', 'atop_disable', 'atop_isenabled', 'atop_info', 'atop_setworkers','cpustring',
    'thread_enable', 'thread_disable', 'thread_isenabled', 'thread_getworkers', 'thread_setworkers', 'thread_zigzag',
    'ledger_enable', 'ledger_disable', 'ledger_isenabled', 'ledger_info',
    'recycler_enable', 'recycler_disable', 'recycler_isenabled', 'recycler_info',
    'timer_gettsc','timer_getutc']
import pnumpy._pnumpy as _pnumpy
from pnumpy._pnumpy import atop_enable, atop_disable, atop_isenabled, atop_info, atop_setworkers, cpustring 
from pnumpy._pnumpy import thread_enable, thread_disable, thread_isenabled, thread_getworkers, thread_setworkers, thread_zigzag
from pnumpy._pnumpy import timer_gettsc, timer_getutc
from pnumpy._pnumpy import ledger_enable, ledger_disable, ledger_isenabled, ledger_info
from pnumpy._pnumpy import recycler_enable, recycler_disable, recycler_isenabled, recycler_info
from pnumpy._pnumpy import getitem, lexsort32, lexsort64

from .benchmark import benchmark, benchmark_func
from .recarray import recarray_to_colmajor
import numpy as np

def init():
    import platform
    if platform.system() == 'Linux':
        from .cpu import cpu_count_linux
        logical,physical = cpu_count_linux()
        _pnumpy.initialize()
    else:
        _pnumpy.initialize()

def initialize():
    init()

def lexsort(*args, **kwargs):
    """
    Perform an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, which can be interpreted as columns in a
    spreadsheet, lexsort returns an array of integer indices that describes
    the sort order by multiple columns. The last key in the sequence is used
    for the primary sort order, the second-to-last key for the secondary sort
    order, and so on. The keys argument must be a sequence of objects that
    can be converted to arrays of the same shape. If a 2D array is provided
    for the keys argument, it's rows are interpreted as the sorting keys and
    sorting is according to the last row, second last row etc.

    Parameters
    ----------
    keys : (k, N) array or tuple containing k (N,)-shaped sequences
        The `k` different "columns" to be sorted.  The last column (or row if
        `keys` is a 2D array) is the primary sort key.
    axis : int, optional
        Axis to be indirectly sorted.  By default, sort over the last axis.

    Returns
    -------
    indices : (N,) ndarray of ints
        Array of indices that sort the keys along the specified axis.

    Threading
    ---------
    Up to 8 threads

    See Also
    --------
    argsort : Indirect sort.
    ndarray.sort : In-place sort.
    sort : Return a sorted copy of an array.

    Examples
    --------
    Sort names: first by surname, then by name.

    >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
    >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
    >>> ind = np.lexsort((first_names, surnames))
    >>> ind
    array([1, 2, 0])

    >>> [surnames[i] + ", " + first_names[i] for i in ind]
    ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']

    Sort two columns of numbers:

    >>> a = [1,5,1,4,3,4,4] # First column
    >>> b = [9,4,0,4,0,2,1] # Second column
    >>> ind = np.lexsort((b,a)) # Sort by a, then by b
    >>> ind
    array([2, 0, 4, 6, 5, 3, 1])

    >>> [(a[i],b[i]) for i in ind]
    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]

    Note that sorting is first according to the elements of ``a``.
    Secondary sorting is according to the elements of ``b``.

    A normal ``argsort`` would have yielded:

    >>> [(a[i],b[i]) for i in np.argsort(a)]
    [(1, 9), (1, 0), (3, 0), (4, 4), (4, 2), (4, 1), (5, 4)]

    Structured arrays are sorted lexically by ``argsort``:

    >>> x = np.array([(1,9), (5,4), (1,0), (4,4), (3,0), (4,2), (4,1)],
    ...              dtype=np.dtype([('x', int), ('y', int)]))

    >>> np.argsort(x) # or np.argsort(x, order=('x', 'y'))
    array([2, 0, 4, 6, 5, 3, 1])
    """

    try:
        return lexsort32(*args, **kwargs)
    except Exception:
        return np.lexsort(*args, **kwargs)
