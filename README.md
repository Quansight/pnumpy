# PNumPy
Parallel NumPy is a new multithreaded package which seamlessly speeds up NumPy for large arrays (> 64,000 elements).
Most importantly, there is *no change required to your existing NumPy code*.

This first release speeds up NumPy binary and unary ufuncs such **add, multiply, isnan, abs, sin, log, sum, min and many more**.
Sped up functions also include: **sort, argsort, lexsort, boolean indexing, and fancy indexing**.
In the near future we will speed up: **astype, where, putmask, arange, searchsorted**.

[![CI Status](https://github.com/Quansight/numpy-threading-extensions/workflows/tox/badge.svg)](https://github.com/Quansight/numpy-threading-extensions/actions)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation
```
pip install pnumpy
```

You can also install the latest development versions with
```
pip install https://github.com/Quansight/numpy-threading-extensions/archive/main.zip
```

## Documentation

See the [full documentation](https://quansight.github.io/numpy-threading-extensions/stable/index.html)

To use the project:

```python
import pnumpy as pn
```

Parallel NumPy will automatically speed up various numpy functions silently under the hood.
To see the improvements yourself run
```
pn.benchmark()
```

To get a partial list of functions sped up run
```
pn.atop_info()
```

To disable or enable pnumpy run
```
pn.disable()
pn.enable()
```

To cap the number of additional worker threads to 3 run
```
pn.thread_setworkers(3)
```

## FAQ
**Q: If I type np.sort(a) where a is an array, will it be sped up?**

*A: If len(a) > 65536 and pnumpy has been imported, it will automatically be sped up*

**Q: How is sort sped up?**

*A: PNumPy uses additional threads to divide up the sorting job.  For example it might perform an 8 way quicksort followed by a 4 way mergesort*

## Development

To run all the tests run:

```
python -m pip install pytest
python -m pytest tests
```
