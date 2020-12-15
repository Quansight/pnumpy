import numpy as np
import pnumpy as pn


def test_enable(initialize_pnumpy):
    # enable/disable return the previous value
    old = pn.atop_isenabled()
    pn.atop_enable()
    assert pn.atop_isenabled() == True
    pn.atop_disable()
    assert pn.atop_isenabled() == False

    # restore prior state
    if old:
        pn.atop_enable()
    else:
        pn.atop_disable()
    assert pn.atop_isenabled() == old


def test_result(rng):
    """ test that the basic idea of rng and ufunc result testing works.
    """

    # this is currently the only test that does not require initialize_pnumpy
    # which is useful for CI runs without AVX2. Otherwise all the tests will be
    # skipped, and pytest will notice that all the tests are skipped and will
    # complain.

    print('numpy version', np.__version__)
    print(pn.cpustring())
  
    m = rng.integers(100, size=(10, 10), dtype=np.int32)
    o = np.empty_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            o[i, j] = m[i, j] + m[i, j]
    assert np.all(np.add(m, m) == o)


def test_numpy_off(initialize_pnumpy):
    np.test()


def test_numpy_on(initialize_pnumpy):
    pn.atop_enable()
    np.test()
