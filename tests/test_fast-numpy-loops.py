import numpy as np

import fast_numpy_loops


def test_initialize():
    ret = fast_numpy_loops.initialize('add')
    assert ret == 'int32,int32->int32'


def test_result(rng):
    # Set subtract to work with the add loop
    ret = fast_numpy_loops.initialize('subtract')
    assert ret == 'int32,int32->int32'
    m = rng.integers(100, size=(10, 10), dtype=np.int32)
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.subtract(m, m) == o)
