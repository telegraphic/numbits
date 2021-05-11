# numbits

![Tests][tests]
[![Coverage Status][coveralls-badge]][coveralls]

## Pack and unpack 1, 2 and 4 bit data to/from 8-bit numpy arrays

Motivated by radio astronomy, where low bitwidths are common.

Project built with [**pybind11**][pybind]. Pack/unpack code based on [**sigpyproc**][sigpyproc].

## Installation

### On Unix (Linux, OS X)

You can either:

* Install numbits from PyPI with:

    ```bash
    pip install numbits
    ```

or you can:

* Clone this repository. and then:
  
  * Build shared object .so locally, using:

    ```bash
    python setup.py build_ext -i
    ```

  * Or install the package globally, using:

    ```bash
    python -m pip install .
    ```

    or:

    ```bash
    python setup.py install
    ```

### Test call

```python
import numpy as np
import numbits
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype='uint8')
b = numbits.unpack(a, nbits=2)

>> b
>> array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 2, 0], dtype=uint8)
```

[tests]: https://github.com/astrogewgaw/numbits/actions/workflows/tests.yaml/badge.svg
[coveralls]: https://coveralls.io/github/astrogewgaw/numbits?branch=feature
[coveralls-badge]: https://coveralls.io/repos/github/astrogewgaw/numbits/badge.svg?branch=feature
[pybind]: https://github.com/pybind/pybind11
[sigpyproc]: https://github.com/FRBs/sigpyproc3
