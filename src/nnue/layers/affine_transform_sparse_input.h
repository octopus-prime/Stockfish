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

// Definition of layer AffineTransformSparseInput of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "../../bitboard.h"
#include "../../memory.h"
#include "../simd.h"
#include "../nnue_common.h"
#include "../nnz_helper.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform) with block sparse input.
*/

namespace Stockfish::Eval::NNUE::Layers {

// Sparse input implementation
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType  = u8;
    using OutputType = i32;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(OutputDimensions % 16 == 0,
                  "Only implemented for OutputDimensions divisible by 16.");

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    static constexpr IndexType ChunkSize = 4;

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
        return get_weight_index_scrambled(i);
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);

        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType*                        input,
                   OutputType*                             output,
                   [[maybe_unused]] const NNZInfo<InDims>& nnzInfo) const {

        using invec_t                       = __m512i;
        using outvec_t                      = __m512i;
        constexpr IndexType OutputSimdWidth = sizeof(outvec_t) / sizeof(OutputType);
        constexpr IndexType NumAccums       = OutputDimensions / OutputSimdWidth;
        // If we're using high-latency dot product instructions, split the accumulators
        // into separate dependency chains and merge at the end
        constexpr IndexType NumRegs = 3 * NumAccums;

        const outvec_t* biasvec = reinterpret_cast<const outvec_t*>(biases);
        outvec_t        acc[NumRegs];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = biasvec[k];

        // convince GCC to not do weird pointer arithmetic in the following loops
        const i8* weights_cp = weights;

        const auto* start = nnzInfo.nnz;
        const auto* end   = nnzInfo.nnz + nnzInfo.count;

        for (IndexType k = NumAccums; k < NumRegs; ++k)
            acc[k] = _mm512_setzero_epi32();
        while (start < end - 2)
        {
            const isize   i0  = *start++;
            const isize   i1  = *start++;
            const isize   i2  = *start++;
            const invec_t in0 = SIMD::broadcast32(input + i0 * sizeof(i32));
            const invec_t in1 = SIMD::broadcast32(input + i1 * sizeof(i32));
            const invec_t in2 = SIMD::broadcast32(input + i2 * sizeof(i32));
            const auto    col0 =
              reinterpret_cast<const invec_t*>(&weights_cp[i0 * OutputDimensions * ChunkSize]);
            const auto col1 =
              reinterpret_cast<const invec_t*>(&weights_cp[i1 * OutputDimensions * ChunkSize]);
            const auto col2 =
              reinterpret_cast<const invec_t*>(&weights_cp[i2 * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
            {
                SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
                SIMD::m512_add_dpbusd_epi32(acc[k + NumAccums], in1, col1[k]);
                SIMD::m512_add_dpbusd_epi32(acc[k + 2 * NumAccums], in2, col2[k]);
            }
        }

        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = _mm512_add_epi32(_mm512_add_epi32(acc[k], acc[k + NumAccums]),
                                      acc[k + 2 * NumAccums]);

        while (start < end)
        {
            const isize   i  = *start++;
            const invec_t in = SIMD::broadcast32(input + i * sizeof(i32));
            const auto    col =
              reinterpret_cast<const invec_t*>(&weights_cp[i * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
                SIMD::m512_add_dpbusd_epi32(acc[k], in, col[k]);
        }
        outvec_t* outptr = reinterpret_cast<outvec_t*>(output);
        for (IndexType k = 0; k < NumAccums; ++k)
            outptr[k] = acc[k];
    }

   private:
    using BiasType   = OutputType;
    using WeightType = i8;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
