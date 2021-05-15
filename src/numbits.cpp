#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define HI4BITS 240
#define LO4BITS 15
#define HI2BITS 192
#define UPMED2BITS 48
#define LOMED2BITS 12
#define LO2BITS 3

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

/*----------------------------------------------------------------------------*/

/*
Function to unpack 1,2 and 4 bit data
data is unpacked into an empty buffer
NOTE: Only unpacks big endian bit ordering
*/
py::array_t<uint8_t> unpack(py::array_t<uint8_t> inarray, int nbits)
{
  // Setup input/output buffers.
  py::buffer_info inbuf = inarray.request();
  int nbytes = inbuf.size;

  auto outarray = py::array_t<uint8_t>(inbuf.size * 8 / nbits);
  py::buffer_info outbuf = outarray.request();

  uint8_t *indata = (uint8_t *)inbuf.ptr;
  uint8_t *outdata = (uint8_t *)outbuf.ptr;

  int ii, jj;
  switch (nbits)
  {
  case 1:
    for (ii = 0; ii < nbytes; ii++)
    {
      for (jj = 0; jj < 8; jj++)
      {
        outdata[(ii * 8) + jj] = (indata[ii] >> jj) & 1;
      }
    }
    break;
  case 2:
    for (ii = 0; ii < nbytes; ii++)
    {
      outdata[(ii * 4) + 3] = indata[ii] & LO2BITS;
      outdata[(ii * 4) + 2] = (indata[ii] & LOMED2BITS) >> 2;
      outdata[(ii * 4) + 1] = (indata[ii] & UPMED2BITS) >> 4;
      outdata[(ii * 4) + 0] = (indata[ii] & HI2BITS) >> 6;
    }
    break;
  case 4:
    for (ii = 0; ii < nbytes; ii++)
    {
      outdata[(ii * 2) + 1] = indata[ii] & LO4BITS;
      outdata[(ii * 2) + 0] = (indata[ii] & HI4BITS) >> 4;
    }
    break;
  }
  return outarray;
}

/*
Function to pack bit data into an empty buffer
*/
py::array_t<uint8_t> pack(py::array_t<uint8_t> inarray, int nbits)
{
  // Setup input/output buffers.
  py::buffer_info inbuf = inarray.request();
  int nbytes = inbuf.size;

  auto outarray = py::array_t<uint8_t>(inbuf.size * nbits / 8);
  py::buffer_info outbuf = outarray.request();

  uint8_t *indata = (uint8_t *)inbuf.ptr;
  uint8_t *outdata = (uint8_t *)outbuf.ptr;

  int ii, pos;
  int bitfact = 8 / nbits;
  unsigned char val;

  switch (nbits)
  {
  case 1:
    for (ii = 0; ii < nbytes / bitfact; ii++)
    {
      pos = ii * 8;
      val = (indata[pos + 7] << 7) |
            (indata[pos + 6] << 6) |
            (indata[pos + 5] << 5) |
            (indata[pos + 4] << 4) |
            (indata[pos + 3] << 3) |
            (indata[pos + 2] << 2) |
            (indata[pos + 1] << 1) |
            indata[pos + 0];
      outdata[ii] = val;
    }
    break;
  case 2:
    for (ii = 0; ii < nbytes / bitfact; ii++)
    {
      pos = ii * 4;
      val = (indata[pos] << 6) |
            (indata[pos + 1] << 4) |
            (indata[pos + 2] << 2) |
            indata[pos + 3];
      outdata[ii] = val;
    }
    break;
  case 4:
    for (ii = 0; ii < nbytes / bitfact; ii++)
    {
      pos = ii * 2;
      val = (indata[pos] << 4) | indata[pos + 1];
      outdata[ii] = val;
    }
    break;
  }
  return outarray;
}

/* Function to rescale from complex 8-bit to 2-bit

Take input data, rescale by stdev, and then optimally
requantize using the method in

VLBA Correlator Memo 75: TWO-BIT CORRELATORS: MISCELLANEOUS RESULTS
Fred Schwab, November 19, 1986

The step function q(x) has values +/-1 and +/- N:

       { + N, for x > a
       | + 1, for 0 < x < a
q(x) = | - 1, for -a < x < 0
       { - N, for x < -a

The optimal values are N = 3.3358750 and a = 0.98159883.
With these values, for noise-like signals the correlator efficiency
is approx ~0.8825.

*/
py::array_t<uint8_t> requant_ci8_cu2(py::array_t<int8_t> input) {
    // Setup pointers and buffers to access data in ndarray
    py::buffer_info buf = input.request();

    // Input array should have size 4x that of output array.
    auto output = py::array_t<uint8_t>(buf.size / 4);
    py::buffer_info bufo = output.request();
    auto *v   = (int8_t *) buf.ptr;
    auto *vo  = (uint8_t *) bufo.ptr;

    // Compute STDEV for real and imag
    double sum_re = 0.0,  sum_im = 0.0;
    double sq_sum_re = 0.0,  sq_sum_im = 0.0;
    size_t X = buf.shape[0] / 2;
    for(size_t idx = 0; idx < X; idx++){
        sum_re += v[2*idx];
        sq_sum_re += v[2*idx] * v[2*idx];
        sum_im += v[2*idx+1];
        sq_sum_im += v[2*idx+1] * v[2*idx+1];
    }
    double mean_re = sum_re / X;
    double mean_im = sum_im / X;
    double stdev_re = std::sqrt(sq_sum_re / X - mean_re * mean_re);
    double stdev_im = std::sqrt(sq_sum_im / X - mean_im * mean_im);

    // Do 2-bit conversion
    for(size_t idx = 0; idx < buf.shape[0] / 4; idx++) {

        // We are going to add all 2-bits together into one 8-bit number
        // So break out each into indexes
        size_t idxr = 4*idx;
        size_t idxi = 4*idx + 1;
        size_t idxr2 = 4*idx + 2;
        size_t idxi2 = 4*idx + 3;
        //std::cout << v[idxr] << " " << v[idxi] << " ";

        // Real part
        if(v[idxr] <  -0.98159883*stdev_re) {
            vo[idx] += 0 * 64;
        } else if(v[idxr] < 0){
            vo[idx] += 1 * 64;
        } else if(v[idxr] < 0.98159883*stdev_re) {
            vo[idx] += 2 * 64;
        } else {
            vo[idx] += 3 * 64;
        }

        if(v[idxr2] <  -0.98159883*stdev_re) {
            vo[idx] += 0 * 4;
        } else if(v[idxr2] < 0){
            vo[idx] += 1 * 4;
        } else if(v[idxr2] < 0.98159883*stdev_re) {
            vo[idx] += 2 * 4;
        } else {
            vo[idx] += 3 * 4;
        }

        // Imag part
        if(v[idxi] <  -0.98159883*stdev_im) {
            vo[idx] += 0 * 16;
        } else if(v[idxi] < 0) {
            vo[idx] += 1 * 16;
        } else if(v[idxi] < 0.98159883*stdev_im) {
            vo[idx] += 2 * 16;
        } else {
            vo[idx] += 3 * 16;
        }

        if(v[idxi2] <  -0.98159883*stdev_im) {
            vo[idx] += 0;
        } else if(v[idxi2] < 0) {
            vo[idx] += 1;
        } else if(v[idxi2] < 0.98159883*stdev_im) {
            vo[idx] += 2;
        } else {
            vo[idx] += 3;
        }

    }
    return output;

}


PYBIND11_MODULE(numbits, m) {
    m.doc() = "Pack and unpack 1, 2 and 4 bit data"; // optional module docstring
    m.def("unpack", &unpack, "Unpack 1, 2 and 4 bit data (unsigned) into an 8-bit numpy array",
          py::arg("inarray"), py::arg("nbits"));
        py::arg("nbits"),
    m.def("pack", &pack, "Pack 1, 2 and 4 bit data into an 8-bit numpy array",
         py::arg("inarray"), py::arg("nbits"));
    m.def("requant_ci8_cu2", &requant_ci8_cu2, "Requantize 8-bit complex data into 2-bit complex data",
         py::arg("inarray"));
}
