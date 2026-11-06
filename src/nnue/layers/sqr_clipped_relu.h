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

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"
#include "../simd.h"

namespace Stockfish::Eval::NNUE::Layers {

// Clipped ReLU
template<IndexType InDims, int WeightScaleBitsLocal = WeightScaleBits>
class SqrClippedReLU {
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

    // Produce the squared and linear clipped activations together, sharing the input loads and
    // the initial signed 32-to-16-bit saturating narrowing. No shuffle is needed to fix up the
    // pack lane order because the next layer's weights were already reverse pre-permuted to match
    // by AffineTransform::get_weight_index_scrambled().
    void propagate_pair(const InputType* input, OutputType* squared, OutputType* clipped) const {
        static_assert(WeightScaleBitsLocal >= 5 && WeightScaleBitsLocal <= 8,
                      "SqrClippedReLU only support WeightScaleBitsLocal between 5 and 8");
        static_assert(InputDimensions % 32 == 0,
                      "propagate_pair() needs a tail path if L2/L3 change to multiple of 16");

        constexpr IndexType NumChunks       = InputDimensions / 32;
        constexpr int       SimdShiftAmount = WeightScaleBitsLocal * 2 + 7 - 16;

        const auto in      = reinterpret_cast<const __m512i*>(input);
        auto       sqrOut  = reinterpret_cast<__m256i*>(squared);
        auto       clipOut = reinterpret_cast<__m256i*>(clipped);

        const __m512i zero = _mm512_setzero_si512();

        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const __m512i words = _mm512_packs_epi32(_mm512_load_si512(&in[i * 2 + 0]),
                                                     _mm512_load_si512(&in[i * 2 + 1]));
            const __m512i sqrWords =
              _mm512_srli_epi16(_mm512_mulhi_epi16(words, words), SimdShiftAmount);
            _mm256_store_si256(&sqrOut[i], _mm512_cvtsepi16_epi8(sqrWords));

            const __m512i clipWords =
              _mm512_srli_epi16(_mm512_max_epi16(words, zero), WeightScaleBitsLocal);
            _mm256_store_si256(&clipOut[i], _mm512_cvtsepi16_epi8(clipWords));
        }
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
