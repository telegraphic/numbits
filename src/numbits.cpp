#include <array>
#include <cstddef>
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
// Lookup table for bit unpacking
template <size_t NBits, bool BigEndian> struct unpack_lookup_table {
  static constexpr size_t Size = 256;
  static constexpr size_t Elements = 8 / NBits;
  alignas(64) uint8_t data[Size][Elements]{}; // 256 * 8/NBits bytes

  constexpr unpack_lookup_table() {
    for (size_t ii = 0; ii < Size; ii++) {
      for (size_t jj = 0; jj < Elements; jj++) {
        if constexpr (BigEndian) {
          data[ii][Elements - 1 - jj] =
              (ii >> (jj * NBits)) & ((1 << NBits) - 1);
        } else {
          data[ii][jj] = (ii >> (jj * NBits)) & ((1 << NBits) - 1);
        }
      }
    }
  }
};

// Compile-time lookup table initialization
constexpr unpack_lookup_table<1, false> unpack_lookup_table_1bit_little{};
constexpr unpack_lookup_table<1, true> unpack_lookup_table_1bit_big{};
constexpr unpack_lookup_table<2, false> unpack_lookup_table_2bit_little{};
constexpr unpack_lookup_table<2, true> unpack_lookup_table_2bit_big{};
constexpr unpack_lookup_table<4, false> unpack_lookup_table_4bit_little{};
constexpr unpack_lookup_table<4, true> unpack_lookup_table_4bit_big{};

template <bool parallel, bool BigEndian>
void unpack_1bit_lookup(const uint8_t *inbuffer, uint8_t *outbuffer,
                        size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
    if constexpr (BigEndian) {
      std::copy(&unpack_lookup_table_1bit_big.data[inbuffer[ii]][0],
                &unpack_lookup_table_1bit_big.data[inbuffer[ii]][8],
                &outbuffer[ii * 8]);
    } else {
      std::copy(&unpack_lookup_table_1bit_little.data[inbuffer[ii]][0],
                &unpack_lookup_table_1bit_little.data[inbuffer[ii]][8],
                &outbuffer[ii * 8]);
    }
  }
}

template <bool parallel, bool BigEndian>
void unpack_2bit_lookup(const uint8_t *inbuffer, uint8_t *outbuffer,
                        size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
    if constexpr (BigEndian) {
      std::copy(&unpack_lookup_table_2bit_big.data[inbuffer[ii]][0],
                &unpack_lookup_table_2bit_big.data[inbuffer[ii]][4],
                &outbuffer[ii * 4]);
    } else {
      std::copy(&unpack_lookup_table_2bit_little.data[inbuffer[ii]][0],
                &unpack_lookup_table_2bit_little.data[inbuffer[ii]][4],
                &outbuffer[ii * 4]);
    }
  }
}

template <bool parallel, bool BigEndian>
void unpack_4bit_lookup(const uint8_t *inbuffer, uint8_t *outbuffer,
                        size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
    if constexpr (BigEndian) {
      std::copy(&unpack_lookup_table_4bit_big.data[inbuffer[ii]][0],
                &unpack_lookup_table_4bit_big.data[inbuffer[ii]][2],
                &outbuffer[ii * 2]);
    } else {
      std::copy(&unpack_lookup_table_4bit_little.data[inbuffer[ii]][0],
                &unpack_lookup_table_4bit_little.data[inbuffer[ii]][2],
                &outbuffer[ii * 2]);
    }
  }
}

template <bool parallel, bool bigEndian>
void unpack_1bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
    for (size_t jj = 0; jj < 8; jj++) {
      if constexpr (bigEndian) {
        outbuffer[(ii << 3) + (7 - jj)] = (inbuffer[ii] >> jj) & 1;
      } else {
        outbuffer[(ii << 3) + jj] = (inbuffer[ii] >> jj) & 1;
      }
    }
  }
}

template <bool parallel, bool bigEndian>
void unpack_2bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
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
void unpack_4bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes; ii++) {
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
void pack_1bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
  size_t pos;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes / 8; ii++) {
    pos = ii * 8;
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
void pack_2bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
  size_t pos;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes / 4; ii++) {
    pos = ii * 4;
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
void pack_4bit(const uint8_t *inbuffer, uint8_t *outbuffer, size_t nbytes) {
  size_t pos;
#ifdef USE_OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (size_t ii = 0; ii < nbytes / 2; ii++) {
    pos = ii * 2;
    if constexpr (bigEndian) {
      outbuffer[ii] = (inbuffer[pos] << 4) | inbuffer[pos + 1];
    } else {
      outbuffer[ii] = inbuffer[pos] | (inbuffer[pos + 1] << 4);
    }
  }
}

using PackUnpackFunc = void (*)(const uint8_t *, uint8_t *, size_t);

constexpr std::array<std::array<std::array<PackUnpackFunc, 2>, 2>, 3>
    unpackLookupDispatcher = {{{{
                                   {unpack_1bit_lookup<false, false>,
                                    unpack_1bit_lookup<true, false>}, // little
                                   {unpack_1bit_lookup<false, true>,
                                    unpack_1bit_lookup<true, true>} // big
                               }},
                               {{
                                   {unpack_2bit_lookup<false, false>,
                                    unpack_2bit_lookup<true, false>}, // little
                                   {unpack_2bit_lookup<false, true>,
                                    unpack_2bit_lookup<true, true>} // big
                               }},
                               {{
                                   {unpack_4bit_lookup<false, false>,
                                    unpack_4bit_lookup<true, false>}, // little
                                   {unpack_4bit_lookup<false, true>,
                                    unpack_4bit_lookup<true, true>} // big
                               }}}};

constexpr std::array<std::array<std::array<PackUnpackFunc, 2>, 2>, 3>
    unpackDispatcher = {
        {{{
             {unpack_1bit<false, false>, unpack_1bit<true, false>}, // little
             {unpack_1bit<false, true>, unpack_1bit<true, true>}    // big
         }},
         {{
             {unpack_2bit<false, false>, unpack_2bit<true, false>}, // little
             {unpack_2bit<false, true>, unpack_2bit<true, true>}    // big
         }},
         {{
             {unpack_4bit<false, false>, unpack_4bit<true, false>}, // little
             {unpack_4bit<false, true>, unpack_4bit<true, true>}    // big
         }}}};

constexpr std::array<std::array<std::array<PackUnpackFunc, 2>, 2>, 3>
    packDispatcher = {
        {{{
             {pack_1bit<false, false>, pack_1bit<true, false>}, // little
             {pack_1bit<false, true>, pack_1bit<true, true>}    // big
         }},
         {{
             {pack_2bit<false, false>, pack_2bit<true, false>}, // little
             {pack_2bit<false, true>, pack_2bit<true, true>}    // big
         }},
         {{
             {pack_4bit<false, false>, pack_4bit<true, false>}, // little
             {pack_4bit<false, true>, pack_4bit<true, true>}    // big
         }}}};

size_t get_bitorder_index(const std::string &bitorder) {
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
unpack_lookup(const py::array_t<uint8_t, py::array::c_style> &inarray,
              size_t nbits, const std::string &bitorder,
              bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  size_t bitorder_idx = get_bitorder_index(bitorder);
  size_t nbits_idx = nbits >> 1;
  size_t nbytes = inarray.size();
  auto outarray = py::array_t<uint8_t>(nbytes * 8 / nbits);

  PackUnpackFunc unpackFunc =
      unpackLookupDispatcher[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  unpackFunc(inarray.data(), outarray.mutable_data(), nbytes);

  return outarray;
}

py::array_t<uint8_t>
unpack(const py::array_t<uint8_t, py::array::c_style> &inarray, size_t nbits,
       const std::string &bitorder, bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  size_t bitorder_idx = get_bitorder_index(bitorder);
  size_t nbits_idx = nbits >> 1;
  size_t nbytes = inarray.size();
  auto outarray = py::array_t<uint8_t>(nbytes * 8 / nbits);

  PackUnpackFunc unpackFunc =
      unpackDispatcher[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  unpackFunc(inarray.data(), outarray.mutable_data(), nbytes);

  return outarray;
}

void unpack_buffered(const py::array_t<uint8_t, py::array::c_style> &inarray,
                     py::array_t<uint8_t, py::array::c_style> &outarray,
                     size_t nbits, const std::string &bitorder,
                     bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  size_t bitorder_idx = get_bitorder_index(bitorder);
  size_t nbits_idx = nbits >> 1;
  size_t nbytes = inarray.size();
  size_t outsize = outarray.size();
  if (outsize != nbytes * 8 / nbits) {
    throw std::invalid_argument("Output buffer size is not correct.");
  }

  PackUnpackFunc unpackFunc =
      unpackDispatcher[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  unpackFunc(inarray.data(), outarray.mutable_data(), nbytes);
}

/*
Function to pack 1, 2 and 4 bit data into an 8-bit array.
*/
py::array_t<uint8_t>
pack(const py::array_t<uint8_t, py::array::c_style> &inarray, size_t nbits,
     const std::string &bitorder, bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  size_t bitorder_idx = get_bitorder_index(bitorder);
  size_t nbits_idx = nbits >> 1;
  size_t nbytes = inarray.size();
  auto outarray = py::array_t<uint8_t>(nbytes * nbits / 8);

  PackUnpackFunc packFunc =
      packDispatcher[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  packFunc(inarray.data(), outarray.mutable_data(), nbytes);

  return outarray;
}

void pack_buffered(const py::array_t<uint8_t, py::array::c_style> &inarray,
                   py::array_t<uint8_t, py::array::c_style> &outarray,
                   size_t nbits, const std::string &bitorder,
                   bool parallel = false) {
  if (nbits != 1 && nbits != 2 && nbits != 4) {
    throw std::invalid_argument(
        "Invalid number of bits. Supported values are 1, 2, and 4.");
  }
  size_t bitorder_idx = get_bitorder_index(bitorder);
  size_t nbits_idx = nbits >> 1;
  size_t nbytes = inarray.size();
  size_t outsize = outarray.size();
  if (outsize != nbytes * nbits / 8) {
    throw std::invalid_argument("Output buffer size is not correct.");
  }

  PackUnpackFunc packFunc =
      packDispatcher[nbits_idx][bitorder_idx][parallel ? 1 : 0];
  packFunc(inarray.data(), outarray.mutable_data(), nbytes);
}

PYBIND11_MODULE(numbits, m) {
  m.doc() = "Pack and unpack 1, 2 and 4 bit data into/from an 8-bit array.";

  m.def(
      "unpack_lookup", &unpack_lookup,
      "Unpack 1, 2 and 4-bit data from an 8-bit numpy array using lookup table",
      py::arg("inarray"), py::arg("nbits"), py::arg("bitorder") = "big",
      py::arg("parallel") = false);

  m.def("unpack", &unpack,
        "Unpack 1, 2 and 4-bit data from an 8-bit numpy array",
        py::arg("inarray"), py::arg("nbits"), py::arg("bitorder") = "big",
        py::arg("parallel") = false);

  m.def("unpack_buffered", &unpack_buffered,
        "Unpack 1, 2 and 4-bit data from an 8-bit numpy array into a "
        "pre-allocated buffer",
        py::arg("inarray"), py::arg("outarray"), py::arg("nbits"),
        py::arg("bitorder") = "big", py::arg("parallel") = false);

  m.def("pack", &pack, "Pack 1, 2 and 4-bit data into an 8-bit numpy array",
        py::arg("inarray"), py::arg("nbits"), py::arg("bitorder") = "big",
        py::arg("parallel") = false);

  m.def("pack_buffered", &pack_buffered,
        "Pack 1, 2 and 4-bit data into an pre-allocated 8-bit numpy array",
        py::arg("inarray"), py::arg("outarray"), py::arg("nbits"),
        py::arg("bitorder") = "big", py::arg("parallel") = false);
}