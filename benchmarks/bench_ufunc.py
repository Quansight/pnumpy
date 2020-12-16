import functools
import os
from typing import ClassVar, List, Mapping, Optional, Tuple

import numpy as np
import pnumpy
pnumpy.initialize()


########
# TODO: Additional benchmarking aspects that should be incorporated into the benchmarks below:
#       * start-of-array address alignment to 1/2/4/8/16/32/64/128-byte boundary to look for perf issues
#         caused by things like missing some loop-peeling step (to process the first part of an array to
#         get to an SIMD-vector-aligned boundary).
#       * striding -- e.g. compare the performance of some function operating on an array of N elements
#         vs. an array with the same number of elements but using a non-default stride (e.g. create an array
#         2/3/4x the size then use slicing syntax with an explicit step value); maybe also include a 3rd
#         comparison point where we use slicing syntax on the original array to take every other element
#         (since that array/view would cover the same amount of memory but a smaller number of elements).
#       * Can we report additional information at the process level (of the asv invocation)? Or would this
#         need to be implemented in asv itself? It would be nice to use 'hwloc' to capture system topology
#         info and save it as e.g. some metadata of the benchmark results, so we can consider doing some
#         post-processing of the benchmark results to normalize by things like L2/L3 cache size, number of cores,
#         SMT on/off (and SMT level), number of memory channels, etc.
#       * Chunk/block size (in bytes, *not* number of elements) that arrays are (virtually) split into for
#         parallel processing. The best value for this parameter will vary by function, CPU/memory characteristics,
#         and (likely) the number of worker threads being used for that function.
#
#         See academic literature on automatic, empirical performance optimization of software for
#         examples of how parameterizing settings like the # of worker threads and the chunk/block
#         size per function on each users' machine can be used to squeeze out a little extra performance.
#         References (not an exhaustive list):
#           * "Automated Empirical Optimization of Software and the ATLAS Project"
#             https://doi.org/10.1016%2FS0167-8191%2800%2900087-9
#           * "Automated empirical tuning of scientific codes for performance and power consumption"
#             https://dl.acm.org/doi/abs/10.1145/1944862.1944880
#           * "Stencil computation optimization and auto-tuning on state-of-the-art multicore architectures"
#             https://www.osti.gov/servlets/purl/964371-KongKj/
#           * "Combining models and guided empirical search to optimize for multiple levels of the memory hierarchy"
#             http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.532.9511
#           * "Statistical Models for Empirical Search-Based Performance Tuning"
#             https://dl.acm.org/doi/abs/10.1177/1094342004041293
########

import random

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences
random.seed(1)
# but will seed it nevertheless
np.random.seed(1)

# Shape (not memory size) of rectangular 2d arrays to use in benchmarks;
# these are also reshaped to 1d.
# Choose non-power-of-two values for the dimensions; the product should be large enough
# so generated arrays (of most/all dtypes) are larger than a typical machine's L3 cache.
#nx, ny = (8009, 8011)
#nx, ny = (4001, 4003)
nx, ny = (1019, 1021)
nxy = nx * ny

# a set of interesting types to test
TYPES1: List[str] = [
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
    values = np.tile(rnd.uniform(0, 100, size=nx), ny)
    return values


@memoize
def get_squares():
    values = get_values()
    squares = {
        # TODO: This approach already causes the 'values' array data to be copied here,
        #       so maybe it's better to just decorate get_values() with functools.lru_cache() and
        #       allow it to accept a dtype parameter? That'd also give us better control over
        #       the distribution of the values in the generated arrays, since we'd be able to e.g.
        #       set the lower/upper bounds of the distribution based on the range of the dtype
        #       (from np.iinfo/np.finfo).
        t: np.array(values, dtype=getattr(np, t)).reshape((nx, ny))
        for t in TYPES1
    }

    # adjust complex ones to have non-degenerate imaginary part -- use
    # original data transposed for that.
    for t, v in squares.items():
        if t.startswith('complex'):
            # TODO: Does this still have the desired effect after transposing back again?
            #       That's a workaround for the shapes not being compatible in the original form.
            v += (v.T*1j).T
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


# TODO: Replace this -- we should be able to just define a setup_cache() method
#       within the BitwiseOps class instead.
@functools.lru_cache(maxsize=5)
def get_shaped_ones(shape: Tuple[int, ...], dtype):
    return np.ones(shape, dtype=dtype)

########


# TODO: Get this list by introspecting numpy rather than hard-coding, so we ensure every ufunc
#       gets benchmarked -- that way, if a new ufunc is implemented in numpy it's automatically
#       included starting with the next benchmark run.
#np_dict = {n: getattr(np, n) for n in dir(np)}
#np_ufuncs = {k: v for k, v in np_dict.items() if isinstance(v, np.ufunc)}
#np_funcs = {k: v for k, v in np_dict.items() if type(v).__qualname__ == 'function'}
ufuncs = [
    'abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
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
    'square', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc'
]
for name in dir(np):
    if isinstance(getattr(np, name, None), np.ufunc) and name not in ufuncs:
        print("Missing ufunc %r" % (name,))



class BenchUtil:
    @staticmethod
    def set_atop_thread_count(ufunc_name: str, elem_type: np.dtype, num_threads: Optional[int]) -> None:
        """
        Set the number of worker threads used by atop for a given ufunc and dtype; if set to zero/None, disables threading.

        TODO: What number of worker threads does atop use if given a mix of dtypes? E.g. the following:
              num_elems = 100_000
              left = np.ones(num_elems, dtype=np.int32)
              right = np.ones(num_elems, dtype=np.int16)
              np.add(left, right, dtype=np.int64)
        """
        if isinstance(elem_type, (str, type)):
            elem_type = np.dtype(elem_type)

        if num_threads is not None and num_threads > 0:
            pnumpy.atop_enable()
            # Current implementation in atop requires the number of threads to be set
            # after calling atop_enable(), otherwise setting the number of worker threads
            # doesn't have any effect.
            if not pnumpy.atop_setworkers(ufunc_name, elem_type.num, num_threads):
                # TODO: Call failed, should probably raise an exception or at least log it.
                pass
        else:
            pnumpy.atop_disable()

    @staticmethod
    def set_pnumpy_thread_count(num_threads: Optional[int]) -> None:
        """
        Set the number of worker threads used by pnumpy; if set to zero/None, disables threading.
        """
        if num_threads is not None and num_threads > 0:
            pnumpy.thread_enable()
            pnumpy.thread_setworkers(num_threads)
        else:
            pnumpy.thread_disable()


# Create the benchmark classes -- one for each ufunc.
for _ufunc_name in ufuncs:
    ### IMPORANT ###
    # DO NOT reference the loop variable (the ufunc name) within the class we're creating below;
    # it appears the loop variable is captured by reference so every benchmark class will run
    # the last ufunc in the ufuncs list instead of the ufunc it's supposed to benchmark.
    # Instead, pass the object to the code within the class by setting a class-level attribute
    # and retrieving that inside the class.
    class UFunc():
        target_ufunc: ClassVar[np.ufunc]

        __slots__ = ('args', 'f')

        params = (
            # array rank
            [1, 2],

            # dtype
            # Parameterizing by dtype lets us easily see if some ufunc isn't being optimized for a
            # specific dtype, and to compare the performance of the ufunc across dtypes
            # (e.g. int16 vs. int32, or int32 vs. uint32).
            list(get_squares().keys()),

            # nthreads
            # TODO: Choose these values based on the number of cores in this machine (os.cpu_count()).
            #       This can be created outside of the ufunc loop and captured here instead of recreating.
            #       Might want to use the 'hwloc' library for this so we can reliably detect CPU cores
            #       that support SMT (and determine whether it's enabled, and whether it's SMT2/4/8).
            #       Also need to account for whether this process has been restricted to run on a subset
            #       of CPU cores -- on Unix, can check that with ``len(os.sched_getaffinity(0))``.
            [0, 2, 4],

            # atop enabled?
            [False, True]

            # TODO: Add parameter for array pooling ("recycling") on/off?
        )
        param_names = ['rank', 'input_dtype', 'nthreads', 'atop']
        timeout = 10

        @property
        def ufunc_obj(self) -> np.ufunc:
            """Convenience property / shorthand for getting the ufunc for this benchmark, since it's stored as a class attribute."""
            return self.__class__.target_ufunc

        def setup(self, rank, input_dtype: str, nthreads: int, atop):
            np.seterr(all='ignore')

            # Configure threading layer.
            BenchUtil.set_pnumpy_thread_count(nthreads)
            BenchUtil.set_atop_thread_count(self.ufunc_obj.__name__, np.dtype(input_dtype), (nthreads if atop else None))

            try:
                # Get the ufunc from the class attribute; if it hasn't been set,
                # raise a NotImplementedError to tell ASV to skip this benchmark.
                self.f = self.ufunc_obj
            except AttributeError:
                raise NotImplementedError(f"The installed version of numpy does not have a '{self.__class__.__name__}' function.")

            # Build up the benchmark arguments.
            a = get_squares()[input_dtype]

            # If rank == 1, flatten the array for testing performance on 1D arrays,
            # but do this in a way that avoids copying the original data to keep this fast.
            if rank == 1:
                # Creating a view then assigning the shape ensures we don't
                # silently copy the data when we're just trying to flatten the array;
                # see np.reshape() docs for details.
                a_view = a.view()
                a_view.shape = (nx * ny,)
                a = a_view
            elif rank == 2:
                # Nothing to do here, the arrays are already built as 2D.
                assert len(a.shape) == rank
            else:
                raise NotImplementedError(f"Benchmark does not know how to create rank-{rank} arrays.")

            # TODO: Need to parameterize this to differentiate between the case where we're invoking
            #       e.g. a binary op with the same array on both sides of the operator and the case
            #       where we're passing two different arrays, they'll have different cache/memory behavior.
            #       Maybe also consider (as a separate, 3-valued parameter) whether we'll pass a pre-created array in
            #       for the 'out' parameter and whether that array is one of the operands or a separate array.
            arg = (a,) * self.f.nin

            try:
                self.f(*arg)
            except TypeError:
                # Function couldn't be called with these arguments; can't proceed so raise NotImplementedError
                # which means ASV will skip the current parameter tuple.
                raise NotImplementedError(f"Constructed parameters were not compatible with the '{self.ufunc_obj.__name__}' ufunc.")

            self.args = arg

        def teardown(self, rank, input_dtype, nthreads, atop):
            # Disable threading / revert to 'off' state.
            BenchUtil.set_pnumpy_thread_count(None)
            BenchUtil.set_atop_thread_count(self.ufunc_obj.__name__, np.dtype(input_dtype), None)

            # TODO: Undo the np.seterr(all='ignore') call made in the setup() function.

        def time_ufunc_types(self, _dummy1, _dummy2, _dummy3, _dummy4):
            # N.B. The splat operator adds a very tiny amount of overhead here, but it should
            #       be negligble and dominated by the run-time of the function itself.
            _ = self.f(*self.args)

    # (See comment at the beginning of this loop.)
    # Get the actual ufunc object from numpy then set a class attribute with it
    # to pass the object into the class while avoiding name-capture issues.
    try:
        UFunc.target_ufunc = getattr(np, _ufunc_name)
    except AttributeError:
        # Just pass here; if the function isn't present in the loaded numpy module,
        # we'll get another AttributeError when executing the class' setup() method
        # which indicates to ASV that the benchmark should be skipped (without failing
        # in a way that prevents the rest of the benchmarks from running).
        pass

    # Fix the name of the benchmark class created for this ufunc,
    # then make it visible in the global namespace.
    # TODO: Do we need to fix __qualname__ to match?
    UFunc.__name__ = f'UFunc_{_ufunc_name}'
    globals()[UFunc.__name__] = UFunc


class BitwiseOps:
    # TODO: Include 'nonzero' here? Or is that not a ufunc/array_function?
    _ufunc_names = frozenset(['bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor'])

    # TODO: Define parameters for dtype, threads, array rank, array_pooling.
    params = (
        # array rank
        [1, 2],

        # dtype
        # Parameterizing by dtype lets us easily see if some ufunc isn't being optimized for a
        # specific dtype, and to compare the performance of the ufunc across dtypes
        # (e.g. int16 vs. int32, or int32 vs. uint32).
        [np.dtype(typecode) for typecode in np.typecodes['AllInteger']],

        # nthreads
        # TODO: Choose these values based on the number of cores in this machine (os.cpu_count()).
        #       This can be created outside of the ufunc loop and captured here instead of recreating.
        #       Might want to use the 'hwloc' library for this so we can reliably detect CPU cores
        #       that support SMT (and determine whether it's enabled, and whether it's SMT2/4/8).
        #       Also need to account for whether this process has been restricted to run on a subset
        #       of CPU cores -- on Unix, can check that with ``len(os.sched_getaffinity(0))``.
        [0, 2, 4],

        # atop enabled?
        [False, True]

        # TODO: Add parameter for array pooling ("recycling") on/off?
    )
    param_names = ['rank', 'input_dtype', 'nthreads', 'atop']
    timeout = 10


    def setup(self, rank, input_dtype: str, nthreads: int, atop):
        # TODO: Increase array size; perhaps also add a parameter to vary the array size; may also want to
        #       cache the created arrays (and maybe clean them up / purge the cache during fixture teardown).
        # TODO: Need to test (perhaps controlled by a parameter) with a second array here; as-is, the benchmarks
        #       below test bitwise operations where the same array is on both sides of the expression; this will
        #       have different cache/memory utilization compared to the more-typical case where we have two
        #       different arrays. A sufficiently-clever implementation of these functions could also detect the
        #       self-application case and use a different way of producing the result (which is interesting and
        #       worth including in a benchmark, it just needs to be differentiated from the typical case).
        # TODO: Create this array in a 'setup_cache' function so ASV will cache it and speed up the benchmark run.

        # Configure threading layer.
        BenchUtil.set_pnumpy_thread_count(nthreads)
        for _ufunc_name in self.__class__._ufunc_names:
            BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(input_dtype), (nthreads if atop else None))

        # Get the shape of the array to create based on the rank parameter.
        if rank == 1:
            b_shape = (nxy,)
        elif rank == 2:
            b_shape = (nx, ny)
        else:
            raise NotImplementedError(f"Benchmark does not know how to create rank-{rank} arrays.")

        self.b = get_shaped_ones(b_shape, dtype=input_dtype)

    def teardown(self, rank, input_dtype, nthreads, atop):
        # Disable threading / revert to 'off' state.
        BenchUtil.set_pnumpy_thread_count(None)
        for _ufunc_name in self.__class__._ufunc_names:
            BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(input_dtype), None)

    def time_nonzero(self, _dummy1, _dummy2, _dummy3, _dummy4):
        # TODO: Does this need to switch between np and pnumpy/pn?
        np.nonzero(self.b)

    def time_bitwise_and_bool(self, _dummy1, _dummy2, _dummy3, _dummy4):
        (self.b & self.b)

    def time_bitwise_not_bool(self, _dummy1, _dummy2, _dummy3, _dummy4):
        (~self.b)

    def time_bitwise_or_bool(self, _dummy1, _dummy2, _dummy3, _dummy4):
        (self.b | self.b)

    def time_bitwise_xor_bool(self, _dummy1, _dummy2, _dummy3, _dummy4):
        (self.b ^ self.b)


class CustomInplace:
    _ufunc_names = frozenset(['add', 'bitwise_or'])
    _ufunc_dtypes = frozenset(['int8', 'int32', 'float32', 'float64'])

    def setup(self):
        # TEMP: Disable threading / revert to 'off' state until we can parameterize
        #       this fixture with thread counts, etc.
        BenchUtil.set_pnumpy_thread_count(None)
        for _ufunc_name in self.__class__._ufunc_names:
            for _ufunc_dtype in self.__class__._ufunc_dtypes:
                BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(_ufunc_dtype), None)

        self.c = np.ones(30_000_000, dtype=np.int8)
        self.i = np.ones(3_000_000, dtype=np.int32)
        self.f = np.zeros(3_000_000, dtype=np.float32)
        self.d = np.zeros(3_000_000, dtype=np.float64)
        # fault memory
        self.f *= 1.
        self.d *= 1.

    def teardown(self):
        # Disable threading / revert to 'off' state.
        BenchUtil.set_pnumpy_thread_count(None)
        for _ufunc_name in self.__class__._ufunc_names:
            for _ufunc_dtype in self.__class__._ufunc_dtypes:
                BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(_ufunc_dtype), None)

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


class CustomScalar:
    _ufunc_names = frozenset(['add', 'bitwise_or'])

    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        # TEMP: Disable threading / revert to 'off' state until we can parameterize
        #       this fixture with thread counts, etc.
        BenchUtil.set_pnumpy_thread_count(None)
        for _ufunc_name in self.__class__._ufunc_names:
            for _ufunc_dtype in self.__class__.params:
                BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(_ufunc_dtype), None)

        self.d = np.ones(7_000_000, dtype=dtype)

    def teardown(self, dtype):
        # Disable threading / revert to 'off' state.
        BenchUtil.set_pnumpy_thread_count(None)
        for _ufunc_name in self.__class__._ufunc_names:
            for _ufunc_dtype in self.__class__.params:
                BenchUtil.set_atop_thread_count(_ufunc_name, np.dtype(_ufunc_dtype), None)

    def time_add_scalar2(self, dtype):
        np.add(self.d, 1)

    def time_divide_scalar2(self, dtype):
        np.divide(self.d, 1)

    def time_divide_scalar2_inplace(self, dtype):
        np.divide(self.d, 1, out=self.d)

    def time_less_than_scalar2(self, dtype):
        (self.d < 1)


# TODO: Logical binary ops (e.g. logical_and).
# TODO: Logical unary ops (e.g. logical_not)
# TODO: Logical reductions (any, all, alltrue).
# TODO: Fancy/bool indexing benchmarks; also: where (both 1- and 3-arg forms), putmask, copyto, choose, take.
# TODO: Reductions (e.g. sum, var, mean, amin, amax, argmin, argmax, nanargmin, nanargmax, nansum)
# TODO: Set ops (e.g. isin, unique)
# TODO: Sorting
