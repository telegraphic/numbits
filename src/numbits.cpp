#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

#define HI4BITS 240
#define LO4BITS 15
#define HI2BITS 192
#define UPMED2BITS 48
#define LOMED2BITS 12
#define LO2BITS 3

/*----------------------------------------------------------------------------*/

template <bool parallel, bool bigEndian>
void unpack_1bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii, jj;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes; ii++) {
    for (jj = 0; jj < 8; jj++) {
      if constexpr (bigEndian) {
        outbuffer[(ii << 3) + (7 - jj)] = (inbuffer[ii] >> jj) & 1;
      } else {
        outbuffer[(ii << 3) + jj] = (inbuffer[ii] >> jj) & 1;
      }
    }
  }
}

template <bool parallel, bool bigEndian>
void unpack_1bit_un(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes; ii++) {
    if constexpr (bigEndian) {
      outbuffer[(ii << 3) + 7] = inbuffer[ii] & 1;
      outbuffer[(ii << 3) + 6] = (inbuffer[ii] & 2) >> 1;
      outbuffer[(ii << 3) + 5] = (inbuffer[ii] & 4) >> 2;
      outbuffer[(ii << 3) + 4] = (inbuffer[ii] & 8) >> 3;
      outbuffer[(ii << 3) + 3] = (inbuffer[ii] & 16) >> 4;
      outbuffer[(ii << 3) + 2] = (inbuffer[ii] & 32) >> 5;
      outbuffer[(ii << 3) + 1] = (inbuffer[ii] & 64) >> 6;
      outbuffer[(ii << 3) + 0] = (inbuffer[ii] & 128) >> 7;
    } else {
      outbuffer[(ii << 3) + 0] = inbuffer[ii] & 1;
      outbuffer[(ii << 3) + 1] = (inbuffer[ii] & 2) >> 1;
      outbuffer[(ii << 3) + 2] = (inbuffer[ii] & 4) >> 2;
      outbuffer[(ii << 3) + 3] = (inbuffer[ii] & 8) >> 3;
      outbuffer[(ii << 3) + 4] = (inbuffer[ii] & 16) >> 4;
      outbuffer[(ii << 3) + 5] = (inbuffer[ii] & 32) >> 5;
      outbuffer[(ii << 3) + 6] = (inbuffer[ii] & 64) >> 6;
      outbuffer[(ii << 3) + 7] = (inbuffer[ii] & 128) >> 7;
    }
  }
}

template <bool parallel, bool bigEndian>
void unpack_2bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes; ii++) {
    if constexpr (bigEndian) {
      outbuffer[(ii << 2) + 3] = inbuffer[ii] & LO2BITS;
      outbuffer[(ii << 2) + 2] = (inbuffer[ii] & LOMED2BITS) >> 2;
      outbuffer[(ii << 2) + 1] = (inbuffer[ii] & UPMED2BITS) >> 4;
      outbuffer[(ii << 2) + 0] = (inbuffer[ii] & HI2BITS) >> 6;
    } else {
      outbuffer[(ii << 2) + 0] = inbuffer[ii] & LO2BITS;
      outbuffer[(ii << 2) + 1] = (inbuffer[ii] & LOMED2BITS) >> 2;
      outbuffer[(ii << 2) + 2] = (inbuffer[ii] & UPMED2BITS) >> 4;
      outbuffer[(ii << 2) + 3] = (inbuffer[ii] & HI2BITS) >> 6;
    }
  }
}

template <bool parallel, bool bigEndian>
void unpack_4bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes; ii++) {
    if constexpr (bigEndian) {
      outbuffer[(ii << 1) + 1] = inbuffer[ii] & LO4BITS;
      outbuffer[(ii << 1) + 0] = (inbuffer[ii] & HI4BITS) >> 4;
    } else {
      outbuffer[(ii << 1) + 0] = inbuffer[ii] & LO4BITS;
      outbuffer[(ii << 1) + 1] = (inbuffer[ii] & HI4BITS) >> 4;
    }
  }
}

template <bool parallel, bool bigEndian>
void pack_1bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii, pos;
  const int bitfact = 8;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes / bitfact; ii++) {
    pos = ii * bitfact;
    if constexpr (bigEndian) {
      outbuffer[ii] = (inbuffer[pos + 0] << 7) | (inbuffer[pos + 1] << 6) |
                      (inbuffer[pos + 2] << 5) | (inbuffer[pos + 3] << 4) |
                      (inbuffer[pos + 4] << 3) | (inbuffer[pos + 5] << 2) |
                      (inbuffer[pos + 6] << 1) | inbuffer[pos + 7];
    } else {
      outbuffer[ii] = inbuffer[pos + 0] | (inbuffer[pos + 1] << 1) |
                      (inbuffer[pos + 2] << 2) | (inbuffer[pos + 3] << 3) |
                      (inbuffer[pos + 4] << 4) | (inbuffer[pos + 5] << 5) |
                      (inbuffer[pos + 6] << 6) | (inbuffer[pos + 7] << 7);
    }
  }
}

template <bool parallel, bool bigEndian>
void pack_2bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii, pos;
  const int bitfact = 4;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes / bitfact; ii++) {
    pos = ii * bitfact;
    if constexpr (bigEndian) {
      outbuffer[ii] = (inbuffer[pos + 0] << 6) | (inbuffer[pos + 1] << 4) |
                      (inbuffer[pos + 2] << 2) | inbuffer[pos + 3];
    } else {
      outbuffer[ii] = inbuffer[pos + 0] | (inbuffer[pos + 1] << 2) |
                      (inbuffer[pos + 2] << 4) | (inbuffer[pos + 3] << 6);
    }
  }
}

template <bool parallel, bool bigEndian>
void pack_4bit(const uint8_t *inbuffer, uint8_t *outbuffer, int nbytes) {
  int ii, pos;
  const int bitfact = 2;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (ii = 0; ii < nbytes / bitfact; ii++) {
    pos = ii * bitfact;
    if constexpr (bigEndian) {
      outbuffer[ii] = (inbuffer[pos] << 4) | inbuffer[pos + 1];
    } else {
      outbuffer[ii] = inbuffer[pos] | (inbuffer[pos + 1] << 4);
    }
  }
}

typedef void (*PackUnpackFunc)(const uint8_t *, uint8_t *, int);

std::array<std::array<std::array<PackUnpackFunc, 2>, 2>, 3> unpackLookup = {
    {{{
         {unpack_1bit_un<false, false>, unpack_1bit_un<true, false>}, // 1-bit, little
         {unpack_1bit_un<false, true>, unpack_1bit_un<true, true>}    // 1-bit, big
     }},
     {{
         {unpack_2bit<false, false>, unpack_2bit<true, false>}, // 2-bit, little
         {unpack_2bit<false, true>, unpack_2bit<true, true>}    // 2-bit, big
     }},
     {{
         {unpack_4bit<false, false>, unpack_4bit<true, false>}, // 4-bit, little
         {unpack_4bit<false, true>, unpack_4bit<true, true>}    // 4-bit, big
     }}}};

std::array<std::array<std::array<PackUnpackFunc, 2>, 2>, 3> packLookup = {
    {{{
         {pack_1bit<false, false>, pack_1bit<true, false>}, // 1-bit, little
         {pack_1bit<false, true>, pack_1bit<true, true>}    // 1-bit, big
     }},
     {{
         {pack_2bit<false, false>, pack_2bit<true, false>}, // 2-bit, little
         {pack_2bit<false, true>, pack_2bit<true, true>}    // 2-bit, big
     }},
     {{
         {pack_4bit<false, false>, pack_4bit<true, false>}, // 4-bit, little
         {pack_4bit<false, true>, pack_4bit<true, true>}    // 4-bit, big
     }}}};

int get_bitorder_index(const std::string &bitorder) {
  if (bitorder.empty() || (bitorder[0] != 'l' && bitorder[0] != 'b')) {
    throw std::invalid_argument(
        "Invalid bitorder. Must begin with 'l' or 'b'.");
  }
  return (bitorder[0] == 'b') ? 1 : 0;
}

/*
Function to unpack 1, 2 and 4 bit data into an 8-bit array.
*/
py::array_t<uint8_t>
unpack(const py::array_t<uint8_t, py::array::c_style> &inarray, int nbits,
       const std::string &bitorder, bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  int bitorder_idx = get_bitorder_index(bitorder);
  int nbits_idx = nbits >> 1;

  int nbytes = inarray.size();
  auto outarray = py::array_t<uint8_t>(nbytes * 8 / nbits);

  PackUnpackFunc unpackFunc =
      unpackLookup[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  unpackFunc(inarray.data(), outarray.mutable_data(), nbytes);

  return outarray;
}

void unpack_buffered(const py::array_t<uint8_t, py::array::c_style> &inarray,
                     py::array_t<uint8_t, py::array::c_style> &outarray,
                     int nbits, const std::string &bitorder,
                     bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  int bitorder_idx = get_bitorder_index(bitorder);
  int nbits_idx = nbits >> 1;

  int nbytes = inarray.size();
  int outsize = outarray.size();
  if (outsize != nbytes * 8 / nbits) {
    throw std::invalid_argument("Output buffer size is not correct.");
  }

  PackUnpackFunc unpackFunc =
      unpackLookup[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  unpackFunc(inarray.data(), outarray.mutable_data(), nbytes);
}

/*
Function to pack 1, 2 and 4 bit data into an 8-bit array.
*/
py::array_t<uint8_t>
pack(const py::array_t<uint8_t, py::array::c_style> &inarray, int nbits,
     const std::string &bitorder, bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  int bitorder_idx = get_bitorder_index(bitorder);
  int nbits_idx = nbits >> 1;

  int nbytes = inarray.size();
  auto outarray = py::array_t<uint8_t>(nbytes * nbits / 8);

  PackUnpackFunc packFunc =
      packLookup[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  packFunc(inarray.data(), outarray.mutable_data(), nbytes);

  return outarray;
}

void pack_buffered(const py::array_t<uint8_t, py::array::c_style> &inarray,
                   py::array_t<uint8_t, py::array::c_style> &outarray,
                   int nbits, const std::string &bitorder,
                   bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  int bitorder_idx = get_bitorder_index(bitorder);
  int nbits_idx = nbits >> 1;

  int nbytes = inarray.size();
  int outsize = outarray.size();
  if (outsize != nbytes * nbits / 8) {
    throw std::invalid_argument("Output buffer size is not correct.");
  }

  PackUnpackFunc packFunc =
      packLookup[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  packFunc(inarray.data(), outarray.mutable_data(), nbytes);
}

PYBIND11_MODULE(numbits, m) {
  // Optional module docstring.
  m.doc() = "Pack and unpack 1, 2 and 4 bit data";

  m.def("unpack", &unpack,
        "Unpack 1, 2 and 4 bit data into an 8-bit numpy array.",
        py::arg("inarray"), py::arg("nbits"), py::arg("bitorder") = "big",
        py::arg("parallel") = false);

  m.def("unpack_buffered", &unpack_buffered,
        "Unpack bit-packed data into a pre-allocated buffer",
        py::arg("inarray"), py::arg("outarray"), py::arg("nbits"),
        py::arg("bitorder") = "big", py::arg("parallel") = false);

  m.def("pack", &pack, "Pack 1, 2 and 4 bit data into an 8-bit numpy array.",
        py::arg("inarray"), py::arg("nbits"), py::arg("bitorder") = "big",
        py::arg("parallel") = false);

  m.def("pack_buffered", &pack_buffered,
        "Pack bit-packed data into a pre-allocated buffer", py::arg("inarray"),
        py::arg("outarray"), py::arg("nbits"), py::arg("bitorder") = "big",
        py::arg("parallel") = false);
}