import numpy as np
import pytest
from packaging.utils import Version
import pnumpy

old_numpy = Version(np.__version__) < Version('1.18')


def pytest_ignore_collect(path, config):
    from numpy.core._multiarray_umath import __cpu_features__ as cpu
    if not cpu['AVX2']:
        import warnings
        warnings.warn('tests/conftest.py: skipping since pnumpy requires AVX2')
        return True

@pytest.fixture(scope='session')
def initialize_pnumpy():
    pnumpy.initialize()

@pytest.fixture(scope='function')
def rng():
    if old_numpy:
        class OldRNG(np.random.RandomState):
            pass
        rng = OldRNG(1234)
        rng.random = rng.random_sample
        rng.integers = rng.randint
        return rng
    else:
        return np.random.default_rng(1234)
