import numpy as np
import perfplot
import numbits


def numbits_1bit(arr):
    return numbits.unpack(arr, 1, parallel=False)


def numbits_1bit_parallel(arr):
    return numbits.unpack(arr, 1, parallel=True)


def np_1bit(arr):
    return np.unpackbits(arr)


b = perfplot.bench(
    setup= lambda n: np.random.randint(256, size=n, dtype='uint8'),
    n_range=[2**k for k in range(8, 15)],
    kernels=[
        numbits_1bit,
        numbits_1bit_parallel,
        np_1bit,
    ],
    xlabel="len(x)",
)
b.save("benchmark_unpack_1bit.png")
