import numbits
import pytest
import numpy as np

class Testnumbits(object):
    def test_unpackbits(self):
        input_arr = np.array([0, 2, 7, 23], dtype=np.uint8)
        expected_bit1 = np.unpackbits(input_arr, bitorder="big")
        expected_bit2 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 3, 0, 1, 1, 3], dtype=np.uint8
        )
        expected_bit4 = np.array([0, 0, 0, 2, 0, 7, 1, 7], dtype=np.uint8)
        np.testing.assert_array_equal(expected_bit1, numbits.unpack(input_arr, nbits=1))
        np.testing.assert_array_equal(expected_bit2, numbits.unpack(input_arr, nbits=2))
        np.testing.assert_array_equal(expected_bit4, numbits.unpack(input_arr, nbits=4))

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_unpackbits_empty(self, nbits):
        input_arr = np.empty((0,), dtype=np.uint8)
        output = numbits.unpack(input_arr, nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_packbits(self, nbits):
        input_arr = np.arange(255, dtype=np.uint8)
        output = numbits.pack(numbits.unpack(input_arr, nbits=nbits), nbits=nbits)
        np.testing.assert_array_equal(input_arr, output)

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