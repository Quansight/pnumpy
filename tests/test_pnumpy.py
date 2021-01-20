import numpy as np

HAVE_PNUMPY = True
try:
    import pnumpy as pn
except Exception:
    HAVE_PNUMPY = False


def test_enable(initialize_pnumpy):
    # enable/disable return the previous value
    if HAVE_PNUMPY:
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

    if HAVE_PNUMPY:
        print('numpy version', np.__version__)
        print(pn.cpustring())
  
        m = rng.integers(100, size=(10, 10), dtype=np.int32)
        o = np.empty_like(m)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                o[i, j] = m[i, j] + m[i, j]
        assert np.all(np.add(m, m) == o)


def test_numpy_off(initialize_pnumpy):
    if HAVE_PNUMPY:
        np.test()


def test_numpy_on(initialize_pnumpy):
    if HAVE_PNUMPY:
        pn.atop_enable()
        np.test()
