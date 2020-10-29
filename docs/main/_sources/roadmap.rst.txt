Roadmap
=======

Version 0.1 of the package exposes the first optimization: multithreaded ufunc
loops. Future versions of the package will extend the capabilites. Some of the
proposed enhancements will require new API from NumPy.

The ledger
----------

At the core of the next generation of enhancements lies the ledger, which will
record various data about the use of UFuncs in the process. This will allow
us to learn your use case: do you call UFuncs with small arrays? Do you use
strides or only contiguous memory? After we have profiled your use, we can
tweak the different enhancements below to better speed up your code.

Replacing loops with faster code
--------------------------------

NumPy only now is beginning to use SIMD instructions to speed up loops. We have
a few further enhancements to the current NumPy implementations. The code lives
in the `atop` directory.

Using a better memory allocator
-------------------------------

NumPy uses a small cache for data memory. We will provide a better cache and
will use the ledger to predict when it is worthwhille to compact the fragmented
cached memory.




