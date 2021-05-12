import numbits
import numpy as np


def test_numbits():

    """
    Test the pack and unpack functions from the numbits package.
    """

    a = np.arange(255, dtype="uint8")

    b1 = numbits.unpack(a, nbits=1)
    c1 = numbits.pack(b1, nbits=1)
    np.allclose(a, c1)

    b2 = numbits.unpack(a, nbits=2)
    c2 = numbits.pack(b2, nbits=2)
    np.allclose(a, c2)

    b4 = numbits.unpack(a, nbits=4)
    c4 = numbits.pack(b4, nbits=4)
    np.allclose(a, c4)