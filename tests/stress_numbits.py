import time
from numpy.random import randint
from numpy import allclose
import numbits

SIZE_ARRAY = 32767000
MAX_LOOPS = 100

def one_round(arg_array, arg_nbits, arg_max_loops):
    down_counter = arg_max_loops
    start_time = time.time()
    while down_counter > 0:
        b = numbits.unpack(arg_array, nbits=arg_nbits)
        down_counter -= 1
    et = time.time() - start_time
    print('try_nbits({}): unpack array_shape={}, loop_count={}, et={:.3f}'
          .format(arg_nbits, arg_array.shape, arg_max_loops, et))
    down_counter = arg_max_loops
    start_time = time.time()
    while down_counter > 0:
        c = numbits.pack(b, nbits=arg_nbits)
        down_counter -= 1
    et = time.time() - start_time
    assert allclose(arg_array, c)
    print('try_nbits({}):   pack array_shape={}, loop_count={}, et={:.3f}'
          .format(arg_nbits, arg_array.shape, arg_max_loops, et))

a_array = randint(256, size=SIZE_ARRAY, dtype='uint8')
one_round(a_array, 1, MAX_LOOPS)
one_round(a_array, 2, MAX_LOOPS)
one_round(a_array, 4, MAX_LOOPS)
