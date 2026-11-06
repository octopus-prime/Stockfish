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

#ifndef NNUE_SIMD_H_INCLUDED
#define NNUE_SIMD_H_INCLUDED

#include <immintrin.h>

#include "../types.h"
#include "../memory.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE::SIMD {


// Two int8 weight rows can be widened and summed
// into int16 lanes with one unpack + one maddubs (against a vector of ones)
// instead of one sign-extension per row. See apply_feature_pair() in
// nnue_accumulator.cpp and FeatureTransformer::WideningPairOrder.
// Accumulator tiles fit in the CPU's vector registers.

using vec_t    = __m512i;
using vec_i8_t = __m256i;


inline constexpr int NumRegistersSIMD = 16;
inline constexpr int MaxChunkSize     = 64;

static_assert(MaxChunkSize == sizeof(vec_t), "MaxChunkSize must equal one vector.");

inline __mmask16 nonzero_mask(__m512i value) {
    return _mm512_cmpgt_epi32_mask(value, _mm512_setzero_si512());
}

inline __m512i broadcast32(const u8* input) { return _mm512_set1_epi32(load_as<i32>(input)); }

[[maybe_unused]] static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {

    acc = _mm512_dpbusd_epi32(acc, a, b);
}

[[maybe_unused]] static int m256_hadd(__m256i sum, int bias) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {

    acc = _mm256_dpbusd_epi32(acc, a, b);
}


// Compute optimal SIMD register count for feature transformer accumulation.
template<IndexType TransformedFeatureWidth, IndexType HalfDimensions>
class SIMDTiling {
// We use __m* types as template arguments, which causes GCC to emit warnings
// about losing some attribute information. This is irrelevant to us as we
// only take their size, so the following pragma are harmless.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

    template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
    static constexpr int BestRegisterCount() {
        constexpr usize RegisterSize = sizeof(SIMDRegisterType);
        constexpr usize LaneSize     = sizeof(LaneType);

        static_assert(RegisterSize >= LaneSize);
        static_assert(MaxRegisters <= NumRegistersSIMD);
        static_assert(MaxRegisters > 0);
        static_assert(NumRegistersSIMD > 0);
        static_assert(RegisterSize % LaneSize == 0);
        static_assert((NumLanes * LaneSize) % RegisterSize == 0);

        const int ideal = (NumLanes * LaneSize) / RegisterSize;
        if (ideal <= MaxRegisters)
            return ideal;

        // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
        for (int divisor = MaxRegisters; divisor > 1; --divisor)
            if (ideal % divisor == 0)
                return divisor;

        return 1;
    }

#pragma GCC diagnostic pop

   public:
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, WeightType, TransformedFeatureWidth, NumRegistersSIMD>();

    static constexpr IndexType TileHeight = NumRegs * sizeof(vec_t) / 2;

    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
};

}

#endif
