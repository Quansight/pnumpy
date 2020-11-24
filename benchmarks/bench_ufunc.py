import numpy as np
import pnumpy
pnumpy.initialize()


########

import random

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences
random.seed(1)
# but will seed it nevertheless
np.random.seed(1)

# Size of square 2d arrays to use in benchmarks
nxy = 1024

# a set of interesting types to test
TYPES1 = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64',  'complex64',
    'longfloat', 'complex128',
]
if 'complex256' in np.typeDict:
    TYPES1.append('complex256')


def memoize(func):
    result = []
    def wrapper():
        if not result:
            result.append(func())
        return result[0]
    return wrapper


# values which will be used to construct our sample data matrices
# replicate 10 times to speed up initial imports of this helper
# and generate some redundancy

@memoize
def get_values():
    rnd = np.random.RandomState(1)
    values = np.tile(rnd.uniform(0, 100, size=nxy * nxy//32), 32)
    return values


@memoize
def get_squares():
    values = get_values()
    squares = {t: np.array(values,
                              dtype=getattr(np, t)).reshape((nxy, nxy))
               for t in TYPES1}

    # adjust complex ones to have non-degenerated imagery part -- use
    # original data transposed for that
    for t, v in squares.items():
        if t.startswith('complex'):
            v += v.T*1j
    return squares


@memoize
def get_vectors():
    # vectors
    vectors = {t: s[0] for t, s in get_squares().items()}
    return vectors


@memoize
def get_indexes():
    indexes = list(range(nxy))
    # so we do not have all items
    indexes.pop(5)
    indexes.pop(95)

    indexes = np.array(indexes)
    return indexes


@memoize
def get_indexes_rand():
    rnd = random.Random(1)

    indexes_rand = get_indexes().tolist()       # copy
    rnd.shuffle(indexes_rand)         # in-place shuffle
    indexes_rand = np.array(indexes_rand)
    return indexes_rand

########



ufuncs = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
          'arctan', 'arctan2', 'arctanh', 'bitwise_and', 'bitwise_not',
          'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj', 'conjugate',
          'copysign', 'cos', 'cosh', 'deg2rad', 'degrees', 'divide', 'divmod',
          'equal', 'exp', 'exp2', 'expm1', 'fabs', 'float_power', 'floor',
          'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'gcd', 'greater',
          'greater_equal', 'heaviside', 'hypot', 'invert', 'isfinite',
          'isinf', 'isnan', 'isnat', 'lcm', 'ldexp', 'left_shift', 'less',
          'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp',
          'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
          'logical_xor', 'matmul', 'maximum', 'minimum', 'mod', 'modf', 'multiply',
          'negative', 'nextafter', 'not_equal', 'positive', 'power',
          'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift',
          'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt',
          'square', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc']


for name in dir(np):
    if isinstance(getattr(np, name, None), np.ufunc) and name not in ufuncs:
        print("Missing ufunc %r" % (name,))

for ufunc in ufuncs:
    class UFunc():
        params = [0, 2, 4]
        param_names = ['nthreads']
        timeout = 10

        def setup(self, nthreads):
            np.seterr(all='ignore')
            if nthreads > 0:
                pnumpy.thread_enable()
                pnumpy.thread_setworkers(nthreads)
            else:
                pnumpy.thread_disable()
            try:
                self.f = getattr(np, ufunc)
            except AttributeError:
                raise NotImplementedError()
            self.args = []
            for t, a in get_squares().items():
                arg = (a,) * self.f.nin
                try:
                    self.f(*arg)
                except TypeError:
                    continue
                self.args.append(arg)

        def time_ufunc_types(self, dummy):
            [self.f(*arg) for arg in self.args]

    UFunc.__name__ = 'UFunc_' + ufunc
    globals()[UFunc.__name__] = UFunc

class Custom():
    def setup(self):
        self.b = np.ones(200000, dtype=bool)

    def time_nonzero(self):
        np.nonzero(self.b)

    def time_not_bool(self):
        (~self.b)

    def time_and_bool(self):
        (self.b & self.b)

    def time_or_bool(self):
        (self.b | self.b)


class CustomInplace():
    def setup(self):
        self.c = np.ones(500000, dtype=np.int8)
        self.i = np.ones(150000, dtype=np.int32)
        self.f = np.zeros(150000, dtype=np.float32)
        self.d = np.zeros(150000, dtype=np.float64)
        # fault memory
        self.f *= 1.
        self.d *= 1.

    def time_char_or(self):
        np.bitwise_or(self.c, 0, out=self.c)
        np.bitwise_or(0, self.c, out=self.c)

    def time_char_or_temp(self):
        0 | self.c | 0

    def time_int_or(self):
        np.bitwise_or(self.i, 0, out=self.i)
        np.bitwise_or(0, self.i, out=self.i)

    def time_int_or_temp(self):
        0 | self.i | 0

    def time_float_add(self):
        np.add(self.f, 1., out=self.f)
        np.add(1., self.f, out=self.f)

    def time_float_add_temp(self):
        1. + self.f + 1.

    def time_double_add(self):
        np.add(self.d, 1., out=self.d)
        np.add(1., self.d, out=self.d)

    def time_double_add_temp(self):
        1. + self.d + 1.


class CustomScalar():
    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.ones(200000, dtype=dtype)

    def time_add_scalar2(self, dtype):
        np.add(self.d, 1)

    def time_divide_scalar2(self, dtype):
        np.divide(self.d, 1)

    def time_divide_scalar2_inplace(self, dtype):
        np.divide(self.d, 1, out=self.d)

    def time_less_than_scalar2(self, dtype):
        (self.d < 1)

