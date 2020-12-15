# PNUMPY
Faster loops for NumPy using multithreading and other tricks. The first release
will target NumPy binary and unary ufuncs. Eventually we will enable overriding
other NumPy functions, and provide an C-based (non-Python) API for extending
via third-party functions.

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
    import pnumpy
    pnumpy.initialize()
```

## Development

To run all the tests run::

```
    python -m pip install pytest
    python -m pytest tests
```
