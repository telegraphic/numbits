c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` numbits.cpp -o numbits`python3-config --extension-suffix`
