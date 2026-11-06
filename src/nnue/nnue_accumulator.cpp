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

#include "nnue_accumulator.h"

#include <algorithm>
#include <cassert>
#include <new>

#include "../bitboard.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"  // IWYU pragma: keep
#include "simd.h"

namespace Stockfish::Eval::NNUE {

using namespace SIMD;

namespace {

template<bool Forward>
void update_accumulator_incremental(Color                     perspective,
                                    const FeatureTransformer& featureTransformer,
                                    const Square              ksq,
                                    int                       psqBlock,
                                    AccumulatorState&         target_state,
                                    const AccumulatorState&   computed);

void update_accumulator_incremental_both(const FeatureTransformer& featureTransformer,
                                         Square                    white_ksq,
                                         Square                    black_ksq,
                                         int                       whiteBlock,
                                         int                       blackBlock,
                                         AccumulatorState&         target_state,
                                         const AccumulatorState&   computed);

void update_accumulator_refresh_cache(Color                     perspective,
                                      const FeatureTransformer& featureTransformer,
                                      const Position&           pos,
                                      AccumulatorState&         accumulatorState,
                                      AccumulatorCaches&        cache);

void update_accumulator_hybrid(Color                     perspective,
                               const Position&           pos,
                               const FeatureTransformer& featureTransformer,
                               AccumulatorState&         target,
                               const AccumulatorState&   computed,
                               AccumulatorCaches&        cache);

}

const AccumulatorState& AccumulatorStack::latest() const noexcept { return accumulators[size - 1]; }

AccumulatorState& AccumulatorStack::mut_latest() noexcept { return accumulators[size - 1]; }

void AccumulatorStack::reset() noexcept {
    accumulators[0].dirtyPiece = {};
    new (&accumulators[0].dirtyThreats) DirtyThreats;
    new (&accumulators[0].dirtyPawnPairs) DirtyPawnPairs;
    accumulators[0].computed.fill(false);
    size = 1;
}

Dirties& AccumulatorStack::push(const Position& pos) noexcept {
    assert(size < MaxSize);
    auto& st = accumulators[size];
    st.computed.fill(false);
    st.blockBefore = PSQFeatureSet::block_state(pos);
    new (&st.dirtyThreats) DirtyThreats;
    new (&st.dirtyPawnPairs) DirtyPawnPairs;
    size++;
    return st;
}

void AccumulatorStack::pop() noexcept {
    assert(size > 1);
    size--;
}

void AccumulatorStack::evaluate(const Position&           pos,
                                const FeatureTransformer& featureTransformer,
                                // Silence spurious warning on GCC 10
                                [[maybe_unused]] AccumulatorCaches& cache) noexcept {
    const usize last_white = find_last_usable_accumulator(WHITE, pos);
    const usize last_black = find_last_usable_accumulator(BLACK, pos);

    if (accumulators[last_white].computed[WHITE] && accumulators[last_black].computed[BLACK])
        forward_update_incremental_both(pos, featureTransformer, last_white, last_black);
    else
    {
        evaluate_side(WHITE, pos, featureTransformer, cache, last_white);
        evaluate_side(BLACK, pos, featureTransformer, cache, last_black);
    }
}

void AccumulatorStack::evaluate_side(Color                     perspective,
                                     const Position&           pos,
                                     const FeatureTransformer& featureTransformer,
                                     AccumulatorCaches&        cache,
                                     usize                     last_usable_accum) noexcept {

    constexpr int MIN_PC_COUNT_HYBRID = 15;

    if (accumulators[last_usable_accum].computed[perspective])
        forward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);

    else
    {
        const auto& dirtyPiece = latest().dirtyPiece;

        if (dirtyPiece.pc == make_piece(perspective, KING)
            && latest().blockBefore[perspective] == PSQFeatureSet::block(pos, perspective)
            && accumulators[size - 2].computed[perspective]
            && pos.count<ALL_PIECES>() >= MIN_PC_COUNT_HYBRID
            && ((int(dirtyPiece.from) & 0b100) == (int(dirtyPiece.to) & 0b100))
            && dirtyPiece.add_sq == SQ_NONE)  // excludes castling
        {
            update_accumulator_hybrid(perspective, pos, featureTransformer, mut_latest(),
                                      accumulators[size - 2], cache);
            return;
        }

        update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(), cache);
        backward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);
    }
}

// Find the earliest usable accumulator, this can either be a computed accumulator or the accumulator
// state just after a change that requires full refresh.
usize AccumulatorStack::find_last_usable_accumulator(Color           perspective,
                                                     const Position& pos) const noexcept {

    // All traversed states must use the current block.
    const int psqBlock = PSQFeatureSet::block(pos, perspective);

    for (usize curr_idx = size - 1; curr_idx > 0; curr_idx--)
    {
        if (accumulators[curr_idx].computed[perspective])
            return curr_idx;

        const auto& dirtyPiece = accumulators[curr_idx].dirtyPiece;
        const int   before     = accumulators[curr_idx].blockBefore[perspective];

        // Threat feature set refreshes require a king move across the center, i.e.,
        // a subset of halfka refreshes
        if (PSQFeatureSet::requires_refresh(dirtyPiece, perspective, before, psqBlock))
            return curr_idx;
    }

    return 0;
}

void AccumulatorStack::forward_update_incremental(Color                     perspective,
                                                  const Position&           pos,
                                                  const FeatureTransformer& featureTransformer,
                                                  const usize               begin) noexcept {

    assert(begin < accumulators.size());
    assert(accumulators[begin].computed[perspective]);

    const Square ksq      = pos.square<KING>(perspective);
    const int    psqBlock = PSQFeatureSet::block(pos, perspective);

    for (usize next = begin + 1; next < size; next++)
        update_accumulator_incremental<true>(perspective, featureTransformer, ksq, psqBlock,
                                             accumulators[next], accumulators[next - 1]);

    assert(latest().computed[perspective]);
}

void AccumulatorStack::backward_update_incremental(Color                     perspective,
                                                   const Position&           pos,
                                                   const FeatureTransformer& featureTransformer,
                                                   const usize               end) noexcept {

    assert(end < accumulators.size());
    assert(end < size);
    assert(latest().computed[perspective]);

    const Square ksq      = pos.square<KING>(perspective);
    const int    psqBlock = PSQFeatureSet::block(pos, perspective);

    for (i64 next = i64(size) - 2; next >= i64(end); next--)
        update_accumulator_incremental<false>(perspective, featureTransformer, ksq, psqBlock,
                                              accumulators[next], accumulators[next + 1]);

    assert(accumulators[end].computed[perspective]);
}

void AccumulatorStack::forward_update_incremental_both(const Position&           pos,
                                                       const FeatureTransformer& featureTransformer,
                                                       usize                     white_begin,
                                                       usize black_begin) noexcept {

    assert(white_begin < size);
    assert(black_begin < size);
    assert(accumulators[white_begin].computed[WHITE]);
    assert(accumulators[black_begin].computed[BLACK]);

    const Square white_ksq    = pos.square<KING>(WHITE);
    const Square black_ksq    = pos.square<KING>(BLACK);
    const int    whiteBlock   = PSQFeatureSet::block(pos, WHITE);
    const int    blackBlock   = PSQFeatureSet::block(pos, BLACK);
    const usize  shared_begin = std::max(white_begin, black_begin);

    // Catch up the lagging perspective, then traverse the common suffix once.
    for (usize next = white_begin + 1; next <= shared_begin; ++next)
        update_accumulator_incremental<true>(WHITE, featureTransformer, white_ksq, whiteBlock,
                                             accumulators[next], accumulators[next - 1]);
    for (usize next = black_begin + 1; next <= shared_begin; ++next)
        update_accumulator_incremental<true>(BLACK, featureTransformer, black_ksq, blackBlock,
                                             accumulators[next], accumulators[next - 1]);

    for (usize next = shared_begin + 1; next < size; ++next)
        update_accumulator_incremental_both(featureTransformer, white_ksq, black_ksq, whiteBlock,
                                            blackBlock, accumulators[next], accumulators[next - 1]);

    assert(latest().computed[WHITE]);
    assert(latest().computed[BLACK]);
}

namespace {

constexpr IndexType Dimensions = FeatureTransformer::OutputDimensions;

using Tiling = SIMDTiling<Dimensions, Dimensions>;

struct Tile {
    vec_t inner[Tiling::NumRegs];
    auto& operator[](int i) { return inner[i]; }
};

sf_always_inline Tile load_tile(IndexType j, const i16* data) {
    Tile  acc;
    auto* column = reinterpret_cast<const vec_t*>(&data[j]);
    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
        acc[k] = column[k];
    return acc;
}

sf_always_inline void store_tile(IndexType j, i16* dest, Tile acc) {
    auto* column = reinterpret_cast<vec_t*>(&dest[j]);
    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
        column[k] = acc[k];
}

sf_always_inline void increment_index(IndexType& j) { j += Tiling::TileHeight; }

template<int sign>
sf_always_inline Tile apply([[maybe_unused]] IndexType j, Tile acc, const i16* data) {
    const auto* column = reinterpret_cast<const vec_t*>(data);
    for (IndexType k = 0; k < Tiling::NumRegs; ++k)
        if constexpr (sign == +1)
            acc[k] = _mm512_add_epi16(acc[k], column[k]);
        else
            acc[k] = _mm512_sub_epi16(acc[k], column[k]);
    return acc;
}

template<int sign>
sf_always_inline Tile apply_feature_single([[maybe_unused]] IndexType j, Tile acc, const i8* data) {
    static_assert(sign == 1 || sign == -1);
    [[maybe_unused]] auto* column = reinterpret_cast<const vec_i8_t*>(data);
    static_assert(Tiling::NumRegs % 2 == 0);
    const vec_t zero = _mm512_setzero_epi32();
#if defined(__clang__)
    #pragma clang loop unroll(full)
#else
    #pragma GCC unroll(Tiling::NumRegs / 2)
#endif
    for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
    {
        const vec_t weight = _mm512_load_si512(reinterpret_cast<const vec_t*>(data) + k / 2);
        const vec_t lo     = _mm512_srai_epi16(_mm512_unpacklo_epi8(zero, weight), 8);
        const vec_t hi     = _mm512_srai_epi16(_mm512_unpackhi_epi8(zero, weight), 8);
        if constexpr (sign == +1)
        {
            acc[k]     = _mm512_add_epi16(acc[k], lo);
            acc[k + 1] = _mm512_add_epi16(acc[k + 1], hi);
        }
        else
        {
            acc[k]     = _mm512_sub_epi16(acc[k], lo);
            acc[k + 1] = _mm512_sub_epi16(acc[k + 1], hi);
        }
    }
    return acc;
}

template<int sign>
sf_always_inline Tile apply_feature_pair(Tile acc, const i8* data0, const i8* data1) {
    static_assert(sign == 1 || sign == -1);
    static_assert(Tiling::NumRegs % 2 == 0);
    const vec_t ones = _mm512_set1_epi8(1);
#if defined(__clang__)
    #pragma clang loop unroll(full)
#else
    #pragma GCC unroll(Tiling::NumRegs / 2)
#endif
    for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
    {
        const vec_t w0 = _mm512_load_si512(reinterpret_cast<const vec_t*>(data0) + k / 2);
        const vec_t w1 = _mm512_load_si512(reinterpret_cast<const vec_t*>(data1) + k / 2);
        const vec_t lo = _mm512_maddubs_epi16(ones, _mm512_unpacklo_epi8(w0, w1));
        const vec_t hi = _mm512_maddubs_epi16(ones, _mm512_unpackhi_epi8(w0, w1));
        if constexpr (sign == +1)
        {
            acc[k]     = _mm512_add_epi16(acc[k], lo);
            acc[k + 1] = _mm512_add_epi16(acc[k + 1], hi);
        }
        else
        {
            acc[k]     = _mm512_sub_epi16(acc[k], lo);
            acc[k + 1] = _mm512_sub_epi16(acc[k + 1], hi);
        }
    }
    return acc;
}

// acc += add - sub in one maddubs per half. Bias both rows to u8 so maddubs' unsigned
// operand is valid; the bias cancels and |add - sub| <= 255 can't saturate.
sf_always_inline Tile apply_feature_diff(Tile acc, const i8* add, const i8* sub) {
    static_assert(Tiling::NumRegs % 2 == 0);
    const vec_t bias  = _mm512_set1_epi8(-128);
    const vec_t signs = _mm512_set1_epi16(-255);  // i8 pairs (+1, -1)
#if defined(__clang__)
    #pragma clang loop unroll(full)
#else
    #pragma GCC unroll(Tiling::NumRegs / 2)
#endif
    for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
    {
        const vec_t ua =
          _mm512_xor_si512(_mm512_load_si512(reinterpret_cast<const vec_t*>(add) + k / 2), bias);
        const vec_t us =
          _mm512_xor_si512(_mm512_load_si512(reinterpret_cast<const vec_t*>(sub) + k / 2), bias);
        acc[k] =
          _mm512_add_epi16(acc[k], _mm512_maddubs_epi16(_mm512_unpacklo_epi8(ua, us), signs));
        acc[k + 1] =
          _mm512_add_epi16(acc[k + 1], _mm512_maddubs_epi16(_mm512_unpackhi_epi8(ua, us), signs));
    }
    return acc;
}

// The weight-table base already includes the tile offset.
template<int sign, bool Incremental = false>
sf_always_inline Tile
apply_features(IndexType j, Tile acc, const u16* indices, int count, const i8* weights) {
    static_assert(sign == 1 || sign == -1);
    if constexpr (Incremental)
    {
        assert(count >= 0 && count <= 2);
        if (count == 2)
            return apply_feature_pair<sign>(acc, &weights[usize(indices[0]) * Dimensions],
                                            &weights[usize(indices[1]) * Dimensions]);
        if (count > 0)
            acc = apply_feature_single<sign>(j, acc, &weights[usize(indices[0]) * Dimensions]);
        if (count > 1)
            acc = apply_feature_single<sign>(j, acc, &weights[usize(indices[1]) * Dimensions]);
        return acc;
    }

    int i = 0;
    for (; i + 1 < count; i += 2)
        acc = apply_feature_pair<sign>(acc, &weights[usize(indices[i]) * Dimensions],
                                       &weights[usize(indices[i + 1]) * Dimensions]);
    for (; i < count; ++i)
        acc = apply_feature_single<sign>(j, acc, &weights[usize(indices[i]) * Dimensions]);
    return acc;
}

// Applies one sign's psq and threat/pp rows. With paired widening, an odd row is held back
// in `odd` so it can be combined with the other sign's odd row.
template<int sign, bool Incremental = false>
sf_always_inline Tile apply_features_but_odd(IndexType                 j,
                                             Tile                      acc,
                                             const u16*                psq,
                                             int                       psqCount,
                                             const u16*                thr,
                                             int                       thrCount,
                                             const FeatureTransformer& ft,
                                             const WeightType*         psqBase,
                                             const i8*&                odd) {
    const auto* psqWeights = &psqBase[j];
    const auto* thrWeights = &ft.threatAndPpWeights[j];
    odd                    = nullptr;
    if ((psqCount + thrCount) % 2 != 0)
        odd = thrCount > 0 ? &thrWeights[usize(thr[--thrCount]) * Dimensions]
                           : &psqWeights[usize(psq[--psqCount]) * Dimensions];

    // Keep the last odd psq row paired with the first threat/pp row of the same sign.
    if (psqCount % 2 != 0)
    {
        --psqCount;
        acc = apply_feature_pair<sign>(acc, &psqWeights[usize(psq[psqCount]) * Dimensions],
                                       &thrWeights[usize(thr[0]) * Dimensions]);
        ++thr;
        --thrCount;
    }

    acc = apply_features<sign, Incremental>(j, acc, psq, psqCount, psqWeights);
    return apply_features<sign>(j, acc, thr, thrCount, thrWeights);
}

sf_always_inline Tile apply_odd_features(IndexType j,
                                         Tile      acc,
                                         const i8* removed,
                                         const i8* added) {
    if (removed && added)
        return apply_feature_diff(acc, added, removed);
    if (removed)
        acc = apply_feature_single<-1>(j, acc, removed);
    if (added)
        acc = apply_feature_single<+1>(j, acc, added);
    return acc;
}

sf_always_inline Tile apply_psq_features(IndexType                       j,
                                         Tile                            acc,
                                         const PSQFeatureSet::IndexList& removed,
                                         const PSQFeatureSet::IndexList& added,
                                         const FeatureTransformer&       ft,
                                         const WeightType*               psqWeights) {
    const i8 *oddRemoved, *oddAdded;
    acc = apply_features_but_odd<-1>(j, acc, removed.begin(), removed.ssize(), nullptr, 0, ft,
                                     psqWeights, oddRemoved);
    acc = apply_features_but_odd<+1>(j, acc, added.begin(), added.ssize(), nullptr, 0, ft,
                                     psqWeights, oddAdded);
    return apply_odd_features(j, acc, oddRemoved, oddAdded);
}

template<int sign>
sf_always_inline Tile apply_threat_features(IndexType                          j,
                                            Tile                               acc,
                                            const ThreatFeatureSet::IndexList& list,
                                            const FeatureTransformer&          ft) {
    return apply_features<sign>(j, acc, list.begin(), list.ssize(), &ft.threatAndPpWeights[j]);
}

template<bool Incremental = false>
sf_always_inline Tile apply_combined_features(IndexType                          j,
                                              Tile                               acc,
                                              const PSQFeatureSet::IndexList&    psqRemoved,
                                              const ThreatFeatureSet::IndexList& thrRemoved,
                                              const PSQFeatureSet::IndexList&    psqAdded,
                                              const ThreatFeatureSet::IndexList& thrAdded,
                                              const FeatureTransformer&          ft,
                                              const WeightType*                  psqBase) {
    if constexpr (Incremental)
    {
        assert(psqRemoved.size() == 1 || psqRemoved.size() == 2);
        assert(psqAdded.size() == 1 || psqAdded.size() == 2);
    }

    const i8 *oddRemoved, *oddAdded;
    acc = apply_features_but_odd<-1, Incremental>(j, acc, psqRemoved.begin(), psqRemoved.ssize(),
                                                  thrRemoved.begin(), thrRemoved.ssize(), ft,
                                                  psqBase, oddRemoved);
    acc = apply_features_but_odd<+1, Incremental>(j, acc, psqAdded.begin(), psqAdded.ssize(),
                                                  thrAdded.begin(), thrAdded.ssize(), ft, psqBase,
                                                  oddAdded);
    return apply_odd_features(j, acc, oddRemoved, oddAdded);
}

void apply_combined(Color                              perspective,
                    const FeatureTransformer&          featureTransformer,
                    const AccumulatorState&            from,
                    AccumulatorState&                  to,
                    const PSQFeatureSet::IndexList&    psqAdded,
                    const PSQFeatureSet::IndexList&    psqRemoved,
                    const ThreatFeatureSet::IndexList& thrAdded,
                    const ThreatFeatureSet::IndexList& thrRemoved,
                    const WeightType*                  psqWeights) {

    const auto& fromAcc = from.accumulation[perspective];
    auto&       toAcc   = to.accumulation[perspective];

    for (IndexType j = 0; j < Dimensions; increment_index(j))
    {
        Tile acc = load_tile(j, fromAcc.data());

        acc = apply_combined_features<true>(j, acc, psqRemoved, thrRemoved, psqAdded, thrAdded,
                                            featureTransformer, psqWeights);

        store_tile(j, toAcc.data(), acc);
    }
}

template<bool Forward>
void update_accumulator_incremental(Color                     perspective,
                                    const FeatureTransformer& featureTransformer,
                                    const Square              ksq,
                                    int                       psqBlock,
                                    AccumulatorState&         target_state,
                                    const AccumulatorState&   computed) {

    assert(computed.computed[perspective]);
    assert(!target_state.computed[perspective]);

    // The size must be enough to contain the largest possible update.
    // That might depend on the feature set and generally relies on the
    // feature set's update cost calculation to be correct and never allow
    // updates with more added/removed features than MaxActiveDimensions.
    PSQFeatureSet::IndexList    psqRemoved, psqAdded;
    ThreatFeatureSet::IndexList thrRemoved, thrAdded;

    const auto& dirtyPiece     = Forward ? target_state.dirtyPiece : computed.dirtyPiece;
    const auto& dirtyThreats   = Forward ? target_state.dirtyThreats : computed.dirtyThreats;
    const auto& dirtyPawnPairs = Forward ? target_state.dirtyPawnPairs : computed.dirtyPawnPairs;

    // Used solely for prefetching
    const auto* threatPpBase = &featureTransformer.threatAndPpWeights[0];
    IndexType   pfStride     = FeatureTransformer::OutputDimensions;

    if constexpr (Forward)
    {
        ThreatFeatureSet::append_changed_indices(perspective, ksq, dirtyThreats, thrRemoved,
                                                 thrAdded, threatPpBase, pfStride);
        PairFeatureSet::append_changed_indices(perspective, ksq, dirtyPawnPairs, thrRemoved,
                                               thrAdded, threatPpBase, pfStride);
        PSQFeatureSet::append_changed_indices(perspective, ksq, dirtyPiece, psqRemoved, psqAdded);
    }
    else
    {
        ThreatFeatureSet::append_changed_indices(perspective, ksq, dirtyThreats, thrAdded,
                                                 thrRemoved, threatPpBase, pfStride);
        PairFeatureSet::append_changed_indices(perspective, ksq, dirtyPawnPairs, thrAdded,
                                               thrRemoved, threatPpBase, pfStride);
        PSQFeatureSet::append_changed_indices(perspective, ksq, dirtyPiece, psqAdded, psqRemoved);
    }

    apply_combined(perspective, featureTransformer, computed, target_state, psqAdded, psqRemoved,
                   thrAdded, thrRemoved, featureTransformer.psq_weights(psqBlock));

    target_state.computed[perspective] = true;
}

void update_accumulator_incremental_both(const FeatureTransformer& featureTransformer,
                                         Square                    white_ksq,
                                         Square                    black_ksq,
                                         int                       whiteBlock,
                                         int                       blackBlock,
                                         AccumulatorState&         target_state,
                                         const AccumulatorState&   computed) {

    assert(computed.computed[WHITE]);
    assert(computed.computed[BLACK]);
    assert(!target_state.computed[WHITE]);
    assert(!target_state.computed[BLACK]);

    PSQFeatureSet::IndexList    psq_removed[COLOR_NB], psq_added[COLOR_NB];
    ThreatFeatureSet::IndexList thr_removed[COLOR_NB], thr_added[COLOR_NB];

    const auto* threat_pp_base = &featureTransformer.threatAndPpWeights[0];
    const auto  pf_stride      = FeatureTransformer::OutputDimensions;

    ThreatFeatureSet::append_changed_indices_both(
      white_ksq, black_ksq, target_state.dirtyThreats, thr_removed[WHITE], thr_added[WHITE],
      thr_removed[BLACK], thr_added[BLACK], threat_pp_base, pf_stride);
    PairFeatureSet::append_changed_indices_both(
      white_ksq, black_ksq, target_state.dirtyPawnPairs, thr_removed[WHITE], thr_added[WHITE],
      thr_removed[BLACK], thr_added[BLACK], threat_pp_base, pf_stride);
    PSQFeatureSet::append_changed_indices(WHITE, white_ksq, target_state.dirtyPiece,
                                          psq_removed[WHITE], psq_added[WHITE]);
    PSQFeatureSet::append_changed_indices(BLACK, black_ksq, target_state.dirtyPiece,
                                          psq_removed[BLACK], psq_added[BLACK]);

    apply_combined(WHITE, featureTransformer, computed, target_state, psq_added[WHITE],
                   psq_removed[WHITE], thr_added[WHITE], thr_removed[WHITE],
                   featureTransformer.psq_weights(whiteBlock));
    apply_combined(BLACK, featureTransformer, computed, target_state, psq_added[BLACK],
                   psq_removed[BLACK], thr_added[BLACK], thr_removed[BLACK],
                   featureTransformer.psq_weights(blackBlock));

    target_state.computed[WHITE] = true;
    target_state.computed[BLACK] = true;
}

Bitboard get_changed_pieces(const std::array<Piece, SQUARE_NB>& oldPieces,
                            const std::array<Piece, SQUARE_NB>& newPieces) {
    static_assert(sizeof(Piece) == 1);
    Bitboard sameBB = 0;

    for (int i = 0; i < 64; i += 32)
    {
        const __m256i old_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&oldPieces[i]));
        const __m256i new_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&newPieces[i]));
        const __m256i cmpEqual  = _mm256_cmpeq_epi8(old_v, new_v);
        const u32     equalMask = _mm256_movemask_epi8(cmpEqual);
        sameBB |= static_cast<Bitboard>(equalMask) << i;
    }
    return ~sameBB;
}

// Append the PSQ feature indices that differ between two piece arrays, given
// the bitboards of the pieces present before and after the change.
sf_always_inline void append_changed_psq_indices(Color                               perspective,
                                                 Square                              ksq,
                                                 const std::array<Piece, SQUARE_NB>& oldPieces,
                                                 Bitboard                            oldBB,
                                                 const std::array<Piece, SQUARE_NB>& newPieces,
                                                 Bitboard                            newBB,
                                                 PSQFeatureSet::IndexList&           removed,
                                                 PSQFeatureSet::IndexList&           added) {
    const Bitboard changedBB = get_changed_pieces(oldPieces, newPieces);
    const Bitboard removedBB = changedBB & oldBB;
    const Bitboard addedBB   = changedBB & newBB;

    PSQFeatureSet::write_indices(oldPieces, newPieces, removedBB, addedBB, perspective, ksq,
                                 removed, added);
}

// Updates accumulator for a king move, and also updates the accumulator cache
// for the new king position.
void update_accumulator_hybrid(Color                     perspective,
                               const Position&           pos,
                               const FeatureTransformer& featureTransformer,
                               AccumulatorState&         target,
                               const AccumulatorState&   computed,
                               AccumulatorCaches&        cache) {
    const auto& dirtyPiece = target.dirtyPiece;

    assert(dirtyPiece.pc == make_piece(perspective, KING));
    assert(dirtyPiece.to != SQ_NONE);
    assert((int(dirtyPiece.from) & 0b100) == (int(dirtyPiece.to) & 0b100));
    assert(computed.computed[perspective]);
    assert(!target.computed[perspective]);

    const Square oldKsq = dirtyPiece.from;
    const Square newKsq = dirtyPiece.to;
    assert(oldKsq != newKsq);

    const auto& currentPieces  = pos.piece_array();
    auto        previousPieces = currentPieces;  // copies 64 bytes!

    Bitboard previousPieceBB = pos.pieces();

    assert(previousPieces[newKsq] == dirtyPiece.pc);

    if (dirtyPiece.remove_sq != SQ_NONE)
    {
        assert(dirtyPiece.remove_sq == newKsq);
        previousPieces[newKsq] = dirtyPiece.remove_pc;
    }
    else
    {
        previousPieces[newKsq] = NO_PIECE;
        previousPieceBB &= ~square_bb(newKsq);
    }

    assert(previousPieces[oldKsq] == NO_PIECE);
    previousPieces[oldKsq] = make_piece(perspective, KING);
    previousPieceBB |= square_bb(oldKsq);

    const int   psqBlock   = PSQFeatureSet::block(pos, perspective);
    const auto& oldEntry   = cache[psqBlock][oldKsq][perspective];
    auto&       newEntry   = cache[psqBlock][newKsq][perspective];
    const auto* psqWeights = featureTransformer.psq_weights(psqBlock);

    // "Remove" means we need to remove them from the cache entry,
    // "Add" means add them to the entry to get the accumulator we want
    PSQFeatureSet::IndexList oldRemove, oldAdd, newRemove, newAdd;

    append_changed_psq_indices(perspective, oldKsq, oldEntry.pieces, oldEntry.pieceBB,
                               previousPieces, previousPieceBB, oldRemove, oldAdd);
    append_changed_psq_indices(perspective, newKsq, newEntry.pieces, newEntry.pieceBB,
                               currentPieces, pos.pieces(), newRemove, newAdd);

    ThreatFeatureSet::IndexList thrRemoved, thrAdded;  // also contain pp indices
    const auto*                 threatPpBase = &featureTransformer.threatAndPpWeights[0];
    IndexType                   pfStride     = FeatureTransformer::OutputDimensions;
    ThreatFeatureSet::append_changed_indices(perspective, newKsq, target.dirtyThreats, thrRemoved,
                                             thrAdded, threatPpBase, pfStride);
    PairFeatureSet::append_changed_indices(perspective, newKsq, target.dirtyPawnPairs, thrRemoved,
                                           thrAdded, threatPpBase, pfStride);

    const auto& fromAcc = computed.accumulation[perspective];
    auto&       toAcc   = target.accumulation[perspective];

    for (IndexType j = 0; j < Dimensions; increment_index(j))
    {
        Tile acc = load_tile(j, newEntry.accumulation.data());

        acc = apply_psq_features(j, acc, newRemove, newAdd, featureTransformer, psqWeights);

        store_tile(j, newEntry.accumulation.data(), acc);

        // adding the old accumulator adds (most of) the threats and pp weights that we need
        acc = apply<1>(j, acc, &fromAcc[j]);
        // But we have added a whole bunch of psq weights for the wrong king bucket which
        // we need to remove
        // first we remove the cached psq accumulation for the old king position...
        acc = apply<-1>(j, acc, &oldEntry.accumulation[j]);

        // ... then we adjust
        // The old entry's removed psq features have to be added back, its added ones removed
        acc = apply_combined_features(j, acc, oldAdd, thrRemoved, oldRemove, thrAdded,
                                      featureTransformer, psqWeights);

        store_tile(j, toAcc.data(), acc);
    }

    newEntry.pieces  = currentPieces;
    newEntry.pieceBB = pos.pieces();

    target.computed[perspective] = true;
}

// HalfKA data comes from the Finny table entry, while the threats are built
// from the active threat features
void update_accumulator_refresh_cache(Color                     perspective,
                                      const FeatureTransformer& featureTransformer,
                                      const Position&           pos,
                                      AccumulatorState&         accumulator,
                                      AccumulatorCaches&        cache) {

    const int                psqBlock   = PSQFeatureSet::block(pos, perspective);
    const Square             ksq        = pos.square<KING>(perspective);
    auto&                    entry      = cache[psqBlock][ksq][perspective];
    const auto*              psqWeights = featureTransformer.psq_weights(psqBlock);
    PSQFeatureSet::IndexList removed, added;

    append_changed_psq_indices(perspective, ksq, entry.pieces, entry.pieceBB, pos.piece_array(),
                               pos.pieces(), removed, added);

    entry.pieceBB = pos.pieces();
    entry.pieces  = pos.piece_array();

    ThreatFeatureSet::IndexList active;
    ThreatFeatureSet::append_active_indices(perspective, pos, active);
    PairFeatureSet::append_active_indices(perspective, pos, active);

    accumulator.computed[perspective] = true;

    for (IndexType j = 0; j < Dimensions; increment_index(j))
    {
        Tile acc = load_tile(j, entry.accumulation.data());

        acc = apply_psq_features(j, acc, removed, added, featureTransformer, psqWeights);

        store_tile(j, &entry.accumulation[0], acc);

        acc = apply_threat_features<+1>(j, acc, active, featureTransformer);

        store_tile(j, accumulator.accumulation[perspective].data(), acc);
    }
}

}

}
