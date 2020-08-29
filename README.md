## numbits

Pack and unpack 1, 2 and 4 bit data to/from 8-bit numpy arrays. Motivated by radio astronomy, 
where low bitwidths are common. 

Project built with [pybind11](https://github.com/pybind/pybind11). Pack/unpack code based
on [sigpyproc](https://github.com/FRBs/sigpyproc3).

### Installation

**On Unix (Linux, OS X)**

 - clone this repository
 - `python setup.py build_ext -i` to build shared object .so locally, or
 - `python setup.py install` to install globally

### Test call

```python
import numpy as np
import numbits
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='uint8')
b = numbits.unpack(a, nbits=2)

>>> array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 2, 0], dtype=uint8)

```
