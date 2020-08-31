import time
from argparse import ArgumentParser
from numpy.random import randint
from numpy import allclose
import numbits


def one_round(arg_array, arg_nbits, arg_max_loops):
    down_counter = arg_max_loops
    start_time = time.time()
    while down_counter > 0:
        b = numbits.unpack(arg_array, nbits=arg_nbits)
        down_counter -= 1
    et = time.time() - start_time
    print('nbits={}: unpack array_shape={}, loop_count={}, et={:.3f}s'
          .format(arg_nbits, arg_array.shape, arg_max_loops, et))
    down_counter = arg_max_loops
    start_time = time.time()
    while down_counter > 0:
        c = numbits.pack(b, nbits=arg_nbits)
        down_counter -= 1
    et = time.time() - start_time
    assert allclose(arg_array, c)
    print('nbits={}:   pack array_shape={}, loop_count={}, et={:.3f}s'
          .format(arg_nbits, arg_array.shape, arg_max_loops, et))


def main(args=None):
    p = ArgumentParser(description='Measure the performance of numbits.')
    p.add_argument('array_size', type=int, help='Size of array (E.g. frequency array).')
    p.add_argument('max_loops', type=int, help='Maximum number of loops (E.g. number of samples).')
    
    if args is None:
        args = p.parse_args()
    else:
        args = p.parse_args(args)

    a_array = randint(256, size=args.array_size, dtype='uint8')

    t1 = time.time()
    one_round(a_array, 1, args.max_loops)
    one_round(a_array, 2, args.max_loops)
    one_round(a_array, 4, args.max_loops)
    et = time.time() - t1
    print('Elapsed run time: {}s'.format(et))


if __name__ == '__main__':
    main()
