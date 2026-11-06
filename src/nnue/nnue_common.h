/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Constants used in NNUE evaluation function

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "../misc.h"
#include "../types.h"

#include <immintrin.h>

namespace Stockfish::Eval::NNUE {

using BiasType         = i16;
using ThreatWeightType = i8;
using WeightType       = i8;
using IndexType        = u32;

// Version of the evaluation file
constexpr u32 Version = 0x6A448AFAu;

// Constant used in evaluation value calculation
constexpr int OutputScale     = 16;
constexpr int WeightScaleBits = 6;
constexpr int FtMaxVal        = 255;
constexpr int HiddenOneVal    = 128;

// Size of cache line (in bytes)
constexpr usize CacheLineSize = 64;

// clang-format off
// Orient a square according to perspective
// Reflect kings on files a..d, so that it lands on e..h
alignas(CacheLineSize) inline constexpr u8 OrientTBL[SQUARE_NB] = {
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
};
// clang-format on


constexpr usize MaxSimdWidth = 32;

// Type of input feature after conversion
using TransformedFeatureType = u8;

// Round n up to be a multiple of base
template<typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
}


// Read an integer from a little-endian stream on x86-64.
template<typename IntType>
inline IntType read_little_endian(std::istream& stream) {
    IntType result;

    stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));

    return result;
}


// Write an integer to a little-endian stream on x86-64.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, IntType value) {

    stream.write(reinterpret_cast<const char*>(&value), sizeof(IntType));
}


// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
template<typename IntType>
inline void read_little_endian(std::istream& stream, IntType* out, usize count) {
    stream.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
}


// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, const IntType* values, usize count) {
    stream.write(reinterpret_cast<const char*>(values), sizeof(IntType) * count);
}


}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_COMMON_H_INCLUDED
