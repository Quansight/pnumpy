import numpy as np
import pytest
from packaging.utils import Version
import fast_numpy_loops

old_numpy = Version(np.__version__) < Version('1.18')

@pytest.fixture(scope='session')
def initialize_fast_numpy_loops():
    fast_numpy_loops.initialize()

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
