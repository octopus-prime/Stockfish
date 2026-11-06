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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// Clipped ReLU
template<IndexType InDims, int WeightScaleBitsLocal = WeightScaleBits>
class ClippedReLU {
   public:
    // Input/output type
    using InputType  = i32;
    using OutputType = u8;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0x538D24C7u;
        hashValue += prevHash;
        // TODO: consider including WeightScaleBitsLocal in the hash value.
        // For now omitted on purpose because not written by trainer (yet)
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream&) { return true; }

    // Write network parameters
    bool write_parameters(std::ostream&) const { return true; }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_hash_value(0));
        return h;
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
