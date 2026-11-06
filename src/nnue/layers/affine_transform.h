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

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <cstdint>
#include <iostream>

#include "../../memory.h"
#include "../nnue_common.h"
#include "../simd.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform).

    - expected use-case is for when PaddedInputDimensions == 32 and InputDimensions <= 32.
      - that's why AVX512 is hard to implement
    - expected use-case is small layers
    - inputs are processed in chunks of 4, weights are respectively transposed
    - accumulation happens directly to int32s
*/

namespace Stockfish::Eval::NNUE::Layers {


// Fallback implementation for older/other architectures.
// Requires the input to be padded to at least 16 values.

template<IndexType InDims, IndexType OutDims>
class AffineTransform {
   public:
    // Input/output type
    using InputType  = u8;
    using OutputType = i32;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

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
        IndexType inputIndex = i % PaddedInputDimensions;

        // At load time, pre-permute the weights to match the per-128-bit-lane interleaving that
        // the previous layer produces, either via SqrClippedReLU::propagate_pair() or via the
        // separate SqrClippedReLU/ClippedReLU propagate() calls, so no shuffle is needed at runtime.
        const IndexType block = inputIndex / 32;
        const IndexType chunk = (inputIndex % 32) / 4;
        inputIndex            = block * 32 + ((chunk % 4) * 2 + chunk / 4) * 4 + inputIndex % 4;
        return inputIndex / 4 * OutputDimensions * 4 + i / PaddedInputDimensions * 4
             + inputIndex % 4;
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
    void propagate(const InputType* input, OutputType* output) const {

        if constexpr (OutputDimensions > 1)
        {
            using vec_t = __m512i;

            static constexpr IndexType OutputSimdWidth = sizeof(vec_t) / sizeof(OutputType);

            static_assert(OutputDimensions % OutputSimdWidth == 0);

            constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
            constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;

            constexpr IndexType NumRegs = 2 * NumAccums;

            const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
            vec_t        acc[NumRegs];
            for (IndexType k = 0; k < NumAccums; ++k)
                acc[k] = biasvec[k];
            for (IndexType k = NumAccums; k < NumRegs; ++k)
                acc[k] = _mm512_set1_epi32(0);

            IndexType i = 0;
            for (; i < NumChunks; i += 2)
            {
                const vec_t in0 = SIMD::broadcast32(input + i * sizeof(i32));
                const vec_t in1 = SIMD::broadcast32(input + (i + 1) * sizeof(i32));
                const auto  col0 =
                  reinterpret_cast<const vec_t*>(&weights[i * OutputDimensions * 4]);
                const auto col1 =
                  reinterpret_cast<const vec_t*>(&weights[(i + 1) * OutputDimensions * 4]);

                for (IndexType k = 0; k < NumAccums; ++k)
                {
                    SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
                    SIMD::m512_add_dpbusd_epi32(acc[k + NumAccums], in1, col1[k]);
                }
            }

            for (IndexType k = 0; k < NumAccums; ++k)
                acc[k] = _mm512_add_epi32(acc[k], acc[k + NumAccums]);
            for (; i < NumChunks; ++i)
            {
                const vec_t in0 = SIMD::broadcast32(input + i * sizeof(i32));
                const auto  col0 =
                  reinterpret_cast<const vec_t*>(&weights[i * OutputDimensions * 4]);

                for (IndexType k = 0; k < NumAccums; ++k)
                    SIMD::m512_add_dpbusd_epi32(acc[k], in0, col0[k]);
            }

            vec_t* outptr = reinterpret_cast<vec_t*>(output);
            for (IndexType k = 0; k < NumAccums; ++k)
                outptr[k] = acc[k];
        }
        else if constexpr (OutputDimensions == 1)
        {
            using vec_t = __m256i;

            const auto inputVector = reinterpret_cast<const vec_t*>(input);

            static constexpr IndexType InputSimdWidth = sizeof(vec_t) / sizeof(InputType);

            static_assert(PaddedInputDimensions % InputSimdWidth == 0);

            constexpr IndexType NumChunks = PaddedInputDimensions / InputSimdWidth;
            vec_t               sum0      = _mm256_setzero_si256();
            const auto          row0      = reinterpret_cast<const vec_t*>(&weights[0]);

            for (int j = 0; j < int(NumChunks); ++j)
            {
                const vec_t in = inputVector[j];
                SIMD::m256_add_dpbusd_epi32(sum0, in, row0[j]);
            }
            output[0] = SIMD::m256_hadd(sum0, biases[0]);
        }
    }

   private:
    using BiasType   = OutputType;
    using WeightType = i8;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
