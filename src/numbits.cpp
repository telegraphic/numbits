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
        outdata[(ii * 8) + (7 - jj)] = (indata[ii] >> jj) & 1;
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
      val = (indata[pos + 0] << 7) |
            (indata[pos + 1] << 6) |
            (indata[pos + 2] << 5) |
            (indata[pos + 3] << 4) |
            (indata[pos + 4] << 3) |
            (indata[pos + 5] << 2) |
            (indata[pos + 6] << 1) |
            indata[pos + 7];
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

PYBIND11_MODULE(numbits, m)
{
  // Optional module docstring.
  m.doc() = "Pack and unpack 1, 2 and 4 bit data";

  m.def("unpack",
        &unpack,
        py::arg("array"),
        py::arg("nbits"),
        "Unpack 1, 2 and 4 bit data into an 8-bit numpy array.");

  m.def("pack",
        &pack,
        py::arg("array"),
        py::arg("nbits"),
        "Pack 1, 2 and 4 bit data into an 8-bit numpy array.");
}