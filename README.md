# numpy-threading-extensions
Faster loops for NumPy using multithreading and other tricks. The first release
will target NumPy binary and unary ufuncs. Eventually we will enable overriding
other NumPy functions, and provide an C-based (non-Python) API for extending
via third-party functions.

[![Travis CI Build Status](https://api.travis-ci.org/Quansight/numpy-threading-extensions.svg)](https://travis-ci.org/Quansight/numpy-threading-extensions)

[![Coverage Status](https://codecov.io/gh/Quansight/numpy-threading-extensions/branch/main/graphs/badge.svg)](https://codecov.io/github/Quansight/numpy-threading-extensions)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation
```
pip install pnumpy
```

You can also install the in-development version 0.0.1 with:
```
pip install https://github.com/Quansight/numpy-threading-extensions/archive/v0.0.1.zip
```
or latest with
```
pip install https://github.com/Quansight/numpy-threading-extensions/archive/main.zip
```

## Documentation

To use the project:

```python
    import pnumpy
    pnumpy.initialize()
```

## Development

To run all the tests run::

```
    python -m pip install pytest
    python -m pytest .
```
