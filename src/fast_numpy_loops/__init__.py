__version__ = '0.0.0'
from fast_numpy_loops._fast_numpy_loops import initialize, enable, disable, isenabled, cpustring
from fast_numpy_loops._fast_numpy_loops import thread_enable, thread_disable, thread_isenabled

# for better reduce loops
import numpy as np
np.setbufsize(8192*1024)
