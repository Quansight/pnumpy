import numpy as np
HAVE_PNUMPY = True
try:
    import pnumpy
except Exception:
    HAVE_PNUMPY = False

import pytest

# pnumpy.initialize() is called from conftest.py


def type2dtype(types):
    """
    Maps the ufunc.type to the input, output dtypes.
    type2dtype('ii->?') -> (np.int32, np.int32), (np.bool_,)
    """
    inp, out = types.split('->')
    return tuple(np.dtype(c) for c in inp), tuple(np.dtype(c) for c in out)


def get_ufuncs_and_types():
    """Create a dictionary with keys of ufunc names and values of all the
    supported type signatures
    """
    ufuncs = [x for x in dir(np) if isinstance(getattr(np, x), np.ufunc)]
    if 'matmul' in ufuncs:
        ufuncs.remove('matmul')
    # Maybe use a collections.defaultdict instead?
    ret = dict([[x, []] for x in ufuncs])
    for s in ret:
        ret[s] = [type2dtype(t) for t in getattr(np, s).types]
    return ret


def fill_random(a, rng):
    """Fill an ndarray with random values. This will uniformly cover the
    bit-valued number space, which in the case of floats is differnt from
    rng.uniform()
    """
    if a.dtype == 'object':
        v = a.reshape(-1)
        # Slow !!!
        for i in range(v.size):
            v[i] = float(rng._bit_generator.random_raw(1))
    else:
        v = a.view(np.uint64)
        v[:] = rng._bit_generator.random_raw(v.size).reshape(v.shape)
    return a

typemap = get_ufuncs_and_types()

def data(in_dtypes, out_dtypes, shape, rng):
    """ Return two tuples: input and output, with random data dtypes and
    shape
    """
    ret_in = [fill_random(np.empty(shape, dtype=d), rng) for d in in_dtypes]
    ret_out = tuple([fill_random(np.empty(shape, dtype=d), rng) for d in out_dtypes])
    return ret_in, ret_out

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(['name', 'types'], ([k, v] for k,v in typemap.items()))
def test_threads(name, types, initialize_pnumpy, rng):
    """ Test that enabling the threading does not change the results
    """
    if HAVE_PNUMPY:
        ufunc = getattr(np, name)
        for in_dtypes, out_dtypes in types:
            # Skip object dtypes
            if any([o == 'object' for o in out_dtypes]):
                continue
            if any([o == 'object' for o in in_dtypes]):
                continue
            in_data, out_data = data(in_dtypes, out_dtypes, [1024, 1024], rng)
            if (name in ('power',) and 
                    issubclass(in_data[1].dtype.type, np.integer)):
                in_data[1] = np.abs(in_data[1])
                in_data[1][in_data[1] < 0] = 0
            if len(out_data) == 1:
                out_data = out_data[0]
            out1 = ufunc(*in_data, out=out_data)
            pnumpy.thread_enable()
            assert pnumpy.thread_isenabled()
            out2 = ufunc(*in_data, out=out_data)
            pnumpy.thread_disable()
            # may not work on datetime
            if not any([o == 'datetime64' for o in out_dtypes]):
                np.testing.assert_allclose(out1, out2, equal_nan=True)
