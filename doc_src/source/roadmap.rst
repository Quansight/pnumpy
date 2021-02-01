Roadmap
=======

Version 2.0 of the package uses multithreaded ufunc loops and parallel sorts.

Future versions of the package will extend these capabilites to cover more of 
the NumPy functionality. Some of these proposed enhancements will require new 
APIs from NumPy.

Conversions
-----------

Currently NumPy does not expose a hook for dtype conversions.  When available,
PNumPy will parallelize those conversions.

Vectorized loops
----------------

NumPy is only now beginning to use `SIMD <https://en.wikipedia.org/wiki/SIMD>`_  
instructions to speed up loops. We have a few further enhancements to the 
current NumPy implementations. Check out the code in the `atop` directory.

Using a better memory allocator
-------------------------------

NumPy uses a small cache for data memory but does not have one for larger
arrays.  When the new API is available, we will provide a better cache.

Ledger
------
What PNumPy hooks can be recorded and timed.  This built in profiler will help
you to tweak and speed up your code.




