import numpy as np

import fast_numpy_loops


def test_main():
    pass


def test_initialize():
    ret = fast_numpy_loops.initialize(['add'])
    assert ret == 'int,int->int'


def test_result(rng):
    m = rng.random((10, 10))
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.add(m, m) == o)
