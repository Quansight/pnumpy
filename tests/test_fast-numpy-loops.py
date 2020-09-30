import numpy as np
import fast_numpy_loops


def test_enable():
    # enable/disable return the previous value
    old = fast_numpy_loops.isenabled()
    assert fast_numpy_loops.enable() == old
    assert fast_numpy_loops.disable() == True
    assert fast_numpy_loops.enable() == False
    old = fast_numpy_loops.isenabled()

def test_result(rng):
    print('numpy version', np.__version__)
  
    m = rng.integers(100, size=(10, 10), dtype=np.int32)
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.add(m, m) == o)

def test_numpy():
    np.test()
