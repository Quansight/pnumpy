import numpy as np
import accelerated_numpy as fn
fn.initialize()

def test_enable():
    # enable/disable return the previous value
    old = fn.thread_isenabled()
    fn.thread_enable()
    assert fn.thread_isenabled() == True
    fn.thread_disable()
    assert fn.thread_isenabled() == False

    # restore prior state
    if old:
        fn.thread_enable()
    else:
        fn.thread_disable()
    assert fn.thread_isenabled() == old

def test_result(rng):
    print('numpy version', np.__version__)
    print(fn.cpustring())
  
    m = rng.integers(100, size=(10, 10), dtype=np.int32)
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.add(m, m) == o)

def test_numpy():
    np.test()
