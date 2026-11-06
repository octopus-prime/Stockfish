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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iterator>

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE {

// Returns the inverse of a permutation
template<usize Len>
constexpr std::array<usize, Len> invert_permutation(const std::array<usize, Len>& order) {
    std::array<usize, Len> inverse{};
    for (usize i = 0; i < order.size(); i++)
        inverse[order[i]] = i;
    return inverse;
}

// Divide a byte region of size TotalSize to chunks of size
// BlockSize, and permute the blocks by a given order
template<usize BlockSize, typename T, usize N, usize OrderSize>
void permute(std::array<T, N>& data, const std::array<usize, OrderSize>& order) {
    constexpr usize TotalSize = N * sizeof(T);

    static_assert(TotalSize % (BlockSize * OrderSize) == 0,
                  "ChunkSize * OrderSize must perfectly divide TotalSize");

    constexpr usize ProcessChunkSize = BlockSize * OrderSize;

    std::array<std::byte, ProcessChunkSize> buffer{};

    std::byte* const bytes = reinterpret_cast<std::byte*>(data.data());

    for (usize i = 0; i < TotalSize; i += ProcessChunkSize)
    {
        std::byte* const values = &bytes[i];

        for (usize j = 0; j < OrderSize; j++)
        {
            auto* const buffer_chunk = &buffer[j * BlockSize];
            auto* const value_chunk  = &values[order[j] * BlockSize];

            std::copy(value_chunk, value_chunk + BlockSize, buffer_chunk);
        }

        std::copy(std::begin(buffer), std::end(buffer), values);
    }
}

// Input feature converter
class FeatureTransformer {
    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = L1;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType ThreatInputDimensions = ThreatFeatureSet::Dimensions;
    static constexpr IndexType PairInputDimensions   = PairFeatureSet::Dimensions;
    static constexpr IndexType PsqDimensions         = PSQFeatureSet::Dimensions;
    static constexpr IndexType ThreatAndPpDimensions = ThreatInputDimensions + PairInputDimensions;
    static constexpr IndexType InputDimensions       = PsqDimensions + ThreatAndPpDimensions;
    static constexpr IndexType OutputDimensions      = HalfDimensions;
    static constexpr IndexType ThreatWeightSize      = ThreatInputDimensions * HalfDimensions;
    static constexpr IndexType PairWeightSize        = PairInputDimensions * HalfDimensions;
    static constexpr IndexType ThreatAndPpWeightSize = ThreatAndPpDimensions * HalfDimensions;

    using BiasesArray            = std::array<BiasType, HalfDimensions>;
    using WeightArray            = std::array<WeightType, HalfDimensions * PsqDimensions>;
    using ThreatAndPpWeightArray = std::array<ThreatWeightType, ThreatAndPpWeightSize>;

    // Size of forward propagation buffer
    static constexpr usize BufferSize = OutputDimensions * sizeof(OutputType);

    // Store the order by which 128-bit blocks of a 1024-bit data must
    // be permuted so that calling packus on adjacent vectors of 16-bit
    // integers loaded from the data results in the pre-permutation order
    static constexpr auto PackusEpi16Order = []() -> std::array<usize, 8> {
        // _mm512_packus_epi16 after permutation:
        // |   0   |   2   |   4   |   6   | // Vector 0
        // |   1   |   3   |   5   |   7   | // Vector 1
        // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 4, 6, 1, 3, 5, 7};
    }();

    static constexpr auto InversePackusEpi16Order = invert_permutation(PackusEpi16Order);

    // The int8 weight rows are consumed one full vector register (sizeof(vec_t)
    // bytes) at a time: unpack{lo,hi}_epi8 interleave the bytes of two rows and
    // maddubs against a vector of ones sums each pair into an int16 lane. Both
    // unpacks work within 128-bit lanes, so the int16 results of a chunk come out
    // with its 8-byte blocks in the order {0, 2, 4, 6, ...} (lo) and {1, 3, 5, 7,
    // ...} (hi). Storing the 8-byte blocks of every chunk in this order at load
    // time makes the two output registers hold consecutive accumulator lanes, so
    // the accumulator layout is unchanged.
    static constexpr auto WideningPairOrder = []() {
        constexpr usize              NumBlocks = sizeof(SIMD::vec_t) / 8;
        std::array<usize, NumBlocks> order{};
        for (usize i = 0; i < NumBlocks; ++i)
            order[i] = i % 2 == 0 ? i / 2 : NumBlocks / 2 + i / 2;
        return order;
    }();

    static constexpr auto InverseWideningPairOrder = invert_permutation(WideningPairOrder);

    static constexpr u32 combine_hash(std::initializer_list<u32> hashes) {
        u32 hash = 0;
        for (const auto component_hash : hashes)
        {
            hash = (hash << 1) | (hash >> 31);
            hash ^= component_hash;
        }
        return hash;
    }

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        return combine_hash(
                 {ThreatFeatureSet::HashValue, PairFeatureSet::HashValue, PSQFeatureSet::HashValue})
             ^ (OutputDimensions * 2);
    }

    void permute_weights() {
        permute<16>(biases, PackusEpi16Order);
        permute<8>(weights, PackusEpi16Order);

        permute<8>(threatAndPpWeights, PackusEpi16Order);

        // With 128-bit vectors the order is the identity
        if constexpr (sizeof(SIMD::vec_t) > 16)
        {
            permute<8>(weights, WideningPairOrder);
            permute<8>(threatAndPpWeights, WideningPairOrder);
        }
    }

    void unpermute_weights() {
        if constexpr (sizeof(SIMD::vec_t) > 16)
        {
            permute<8>(weights, InverseWideningPairOrder);
            permute<8>(threatAndPpWeights, InverseWideningPairOrder);
        }

        permute<16>(biases, InversePackusEpi16Order);
        permute<8>(weights, InversePackusEpi16Order);
        permute<8>(threatAndPpWeights, InversePackusEpi16Order);
    }

    ThreatWeightType* threatWeightData() { return threatAndPpWeights.data(); }
    ThreatWeightType* pawnPairWeightData() { return threatWeightData() + ThreatWeightSize; }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian(stream, biases.data(), HalfDimensions);

        read_little_endian(stream, threatWeightData(), ThreatWeightSize);
        read_little_endian(stream, pawnPairWeightData(), PairWeightSize);

        read_little_endian(stream, weights.data(), weights.size());

        permute_weights();

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        std::unique_ptr<FeatureTransformer> copy = std::make_unique<FeatureTransformer>(*this);

        copy->unpermute_weights();

        write_little_endian(stream, copy->biases.data(), HalfDimensions);

        write_little_endian(stream, copy->threatWeightData(), ThreatWeightSize);
        write_little_endian(stream, copy->pawnPairWeightData(), PairWeightSize);

        write_little_endian(stream, copy->weights.data(), copy->weights.size());

        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;

        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));

        hash_combine(h, get_raw_data_hash(threatAndPpWeights));

        hash_combine(h, get_hash_value());

        return h;
    }

    const WeightType* psq_weights(int block) const {
        return &weights[usize(PSQFeatureSet::block_offset(block)) * HalfDimensions];
    }

    // Convert input features
    void transform(const Position&                             pos,
                   AccumulatorStack&                           accumulatorStack,
                   AccumulatorCaches&                          cache,
                   OutputType*                                 output,
                   [[maybe_unused]] NNZInfo<OutputDimensions>& nnzInfo) const {
        accumulatorStack.evaluate(pos, *this, cache);
        const auto& accumulatorState = accumulatorStack.latest();

        const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& accumulation    = accumulatorState.accumulation;

        for (IndexType p = 0; p < 2; ++p)
            transform_perspective(accumulation[perspectives[p]], output, p, nnzInfo);
    }

   private:
    static void transform_perspective(const std::array<i16, HalfDimensions>&      accumulation,
                                      OutputType*                                 output,
                                      IndexType                                   perspective,
                                      [[maybe_unused]] NNZInfo<OutputDimensions>& nnzInfo) {

        using namespace SIMD;
        const IndexType offset = (HalfDimensions / 2) * perspective;

        [[maybe_unused]] auto cursor = nnzInfo.make_cursor(perspective);

        constexpr IndexType OutputChunkSize = MaxChunkSize;
        static_assert((HalfDimensions / 2) % OutputChunkSize == 0);
        constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

        [[maybe_unused]] const vec_t   Zero  = _mm512_setzero_epi32();
        [[maybe_unused]] const vec_t   FtMax = _mm512_set1_epi16(FtMaxVal);
        [[maybe_unused]] constexpr int shift = 7;

        const vec_t* in0 = reinterpret_cast<const vec_t*>(&accumulation[0]);
        const vec_t* in1 = reinterpret_cast<const vec_t*>(&accumulation[HalfDimensions / 2]);
        vec_t*       out = reinterpret_cast<vec_t*>(output + offset);

        // Per the NNUE architecture, here we want to multiply pairs of
        // clipped elements and divide the product by 512. To do this,
        // we can naively perform min/max operation to clip each of the
        // four int16 vectors, mullo pairs together, then pack them into
        // one int8 vector. However, there exists a faster way.

        // The idea here is to use the implicit clipping from packus to
        // save us two _mm512_max_epi16 instructions. This clipping works due
        // to the fact that any int16 integer below zero will be zeroed
        // on packus.

        // Consider the case where the second element is negative.
        // If we do standard clipping, that element will be zero, which
        // means our pairwise product is zero. If we perform packus and
        // remove the lower-side clip for the second element, then our
        // product before packus will be negative, and is zeroed on pack.
        // The two operations produce equivalent results, but the second
        // one (using packus) saves one max operation per pair.

        // But here we run into a problem: mullo does not preserve the
        // sign of the multiplication. We can get around this by doing
        // mulhi, which keeps the sign. But that requires an additional
        // tweak.

        // mulhi cuts off the last 16 bits of the resulting product,
        // which is the same as performing a rightward shift of 16 bits.
        // We can use this to our advantage. Recall that we want to
        // divide the final product by 512, which is equivalent to a
        // 9-bit right shift. Intuitively, if we shift the clipped
        // value left by 7, and perform mulhi, which shifts the product
        // right by 16 bits, then we will net a right shift of 9 bits.

        for (IndexType j = 0; j < NumOutputChunks; j += 2)
        {
            vec_t packed[2];
            for (IndexType k = 0; k < 2; ++k)
            {
                const IndexType i = (j + k) * 2;

                vec_t acc0a = in0[i + 0];
                vec_t acc0b = in0[i + 1];
                vec_t acc1a = in1[i + 0];
                vec_t acc1b = in1[i + 1];

                static_assert(FtMaxVal == 255);

                vec_t sum0a =
                  _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0a, FtMax), Zero), shift);
                vec_t sum0b =
                  _mm512_slli_epi16(_mm512_max_epi16(_mm512_min_epi16(acc0b, FtMax), Zero), shift);
                vec_t sum1a = _mm512_min_epi16(acc1a, FtMax);
                vec_t sum1b = _mm512_min_epi16(acc1b, FtMax);

                vec_t pa = _mm512_mulhi_epi16(sum0a, sum1a);
                vec_t pb = _mm512_mulhi_epi16(sum0b, sum1b);

                vec_t result = _mm512_packus_epi16(pa, pb);

                packed[k] = out[j + k] = result;
            }

            cursor.record2(packed[0], packed[1]);
        }
    }

   public:
    alignas(CacheLineSize) BiasesArray biases;
    alignas(CacheLineSize) WeightArray weights;

    // Threats and pawn-pair features are concatenated into one array to allow for a single index to address either.
    // The first pawn-pair feature is at index ThreatFeatureSet::Dimensions.
    static_assert(PairFeatureSet::IndexBase == ThreatFeatureSet::Dimensions);

    alignas(CacheLineSize) ThreatAndPpWeightArray threatAndPpWeights;
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::FeatureTransformer> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::FeatureTransformer& ft) const noexcept {
        return ft.get_content_hash();
    }
};

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
