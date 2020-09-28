import numpy as np

print("before numpy_loops import")

import fast_numpy_loops

print("before numpy_loops init")
# only init once
fast_numpy_loops.initialize()

def test_result(rng):
    print('numpy version', np.__version__)
  
    m = rng.integers(100, size=(10, 10), dtype=np.int32)
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.add(m, m) == o)
