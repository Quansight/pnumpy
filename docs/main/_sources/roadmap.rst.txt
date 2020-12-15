Roadmap
=======

Version 2.0 of the package exposes the first optimization: multithreaded ufunc
loops. 

Future versions of the package will extend these capabilites to cover more of 
the NumPy functionality. Some of these proposed enhancements will require new 
API from NumPy.

The ledger
----------

At the core of the next generation of enhancements lies the ledger, which will
record various data about the use of UFuncs in the process. This will allow
us to learn your use case:  

- Do you call UFuncs with small arrays?   
- Do you use strides or only contiguous memory?   

After we have profiled your use, we can
tweak the different enhancements below to better speed up your code.

Replacing loops with faster code
--------------------------------

NumPy is only now beginning to use `SIMD <https://en.wikipedia.org/wiki/SIMD>`_  
instructions to speed up loops. We have a few further enhancements to the 
current NumPy implementations. Check out the code in the `atop` directory.

Using a better memory allocator
-------------------------------

NumPy uses a small cache for data memory. We will provide a better cache and
will use the ledger to predict when it is worthwhile to compact the fragmented
cached memory.




