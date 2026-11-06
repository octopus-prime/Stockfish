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

#ifndef NNZ_HELPER_H_INCLUDED
#define NNZ_HELPER_H_INCLUDED

#include <utility>

#include "nnue_common.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE {

template<usize Dimensions>
struct NNZInfo {

    unsigned count = 0;
    // indices of non-zero chunks
    u16 nnz[Dimensions / 4];

    alignas(64) static constexpr auto Indices = []() {
        std::array<std::array<u16, 32>, 2> indices{};
        for (int i = 0; i < 2; ++i)
        {
            indices[i] = {0, 1, 2,  3,  16, 17, 18, 19, 4,  5,  6,  7,  20, 21, 22, 23,
                          8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
            for (u16& m : indices[i])
                m += u16(i * Dimensions / 8);
        }
        return indices;
    }();

    struct NNZCursor {
        NNZInfo& info;
        __m512i  indices;
        unsigned count;

        NNZCursor(NNZInfo& info_, bool perspective, unsigned count_) :
            info(info_),
            count(count_) {
            indices = _mm512_load_si512(&Indices[perspective]);
        }

        void record2(SIMD::vec_t neurons1, SIMD::vec_t neurons2) {
            const __m512i increment = _mm512_set1_epi16(32);

            // Get a bitmask and gather non zero indices
            const __m512i   inputV01 = _mm512_packs_epi32(neurons1, neurons2);
            const __mmask32 nnzMask  = _mm512_test_epi16_mask(inputV01, inputV01);

            // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
            __m512i nnzIndices = _mm512_maskz_compress_epi16(nnzMask, indices);
            _mm512_storeu_si512(info.nnz + count, nnzIndices);

            count += popcount(nnzMask);
            indices = _mm512_add_epi16(indices, increment);
        }

        ~NNZCursor() { info.count = count; }
    };

    NNZCursor make_cursor(bool perspective) { return {*this, perspective, count}; }
};
}  // namespace Stockfish::Eval::NNUE

#endif
