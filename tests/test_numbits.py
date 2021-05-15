import numbits
import numpy as np


def test_numbits():
    """
    Test the pack and unpack functions from the numbits package.
    """

    a_1 = np.array([0, 1] * 4 * 10, dtype='int8')
    a_2 = np.array([0, 1, 2, 3] * 10, dtype='int8')
    a_4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 10, dtype='int8')

    a_1p = numbits.pack(a_1, 1)
    a_2p = numbits.pack(a_2, 2)
    a_4p = numbits.pack(a_4, 4)

    a_1up = numbits.unpack(a_1p, 1)
    a_2up = numbits.unpack(a_2p, 2)
    a_4up = numbits.unpack(a_4p, 4)

    assert np.allclose(a_1, a_1up)
    assert np.allclose(a_2, a_2up)
    assert np.allclose(a_4, a_4up)

def test_requant_ci8_cu2():
    """ Test conversion from Complex 8-bit to complex 2-bit data """
    a = np.random.normal(size=1024, scale=127).astype('int8')
    a_2b_pack = numbits.requant_ci8_cu2(a)
    b = numbits.unpack(a_2b_pack, 2)

    # TODO: Verify output


