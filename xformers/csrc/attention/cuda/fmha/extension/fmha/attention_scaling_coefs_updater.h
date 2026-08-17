/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/functional.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_ppu.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_ppu.h"
#include "cutlass/matrix_shape.h"
#include "gemm_kernel_utils.h"


/* Iterates on the accumulator and corresponding position on result matrix

(1) Update `mi[r]` to the max value of the row `r`
(2) In a second iteration do the following:
    (a) accum   <- exp(accum - mi)
    (b) m_prime <- exp(m_prime - mi)
    (c) s_prime <- s_prime * m_prime + sum(accum)

All of this is done on registers, before we store all of this
on shared memory for the next matmul with Value.

We have multiple implementations, because each configuration has a different way
of iterating in the accumulators.
*/


template <typename BASE, typename T, typename accum_t, int kWarpSize>
struct RegisterOps {
  template <
      int kQueriesPerBlock,
      bool kFullColumns,
      bool kIsFirst,
      bool kKeepOutputInRF,
      int WarpCount>
  CUTLASS_DEVICE static void update(
      typename T::Fragment& frag_o, // output so far
      typename T::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock * WarpCount>& s_prime,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      typename T::TensorCoord const& tile_offset,
      float scaling) {
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    if (!kIsFirst) {
      if (thread_id < kQueriesPerBlock) {
        m_prime[thread_id] = mi[thread_id];
      }
      __syncthreads();
    }

    auto lane_offset = BASE::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (kFullColumns || accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max * scaling);
          });
    }
    frag = cutlass::multiplies<typename T::Fragment>()(scaling * kLog2e, frag);

    // Make sure we all share the update values for `mi`
    __syncthreads();

    if (thread_id < kQueriesPerBlock) {
      auto m_prime_exp = exp2f(kLog2e * (m_prime[thread_id] - mi[thread_id]));
      m_prime[thread_id] = m_prime_exp;
      s_prime[thread_id] *= m_prime_exp;
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !kIsFirst) {
      accum_t mp;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mp = m_prime[accum_m]; },
          [&](int accum_m, int accum_n, int idx) { frag_o[idx] *= mp; },
          [&](int accum_m) {});
      __syncthreads();
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = kLog2e * mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] = (kFullColumns || accum_n < max_col)
                ? exp2f(frag[idx] - mi_row)
                : accum_t(0.0);
          },
          [&](int accum_m) {});
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (BASE::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // save result to shared
              s_prime[accum_m + warp_id * kQueriesPerBlock] += total_row;
              //atomicAdd(&s_prime[accum_m], total_row);
            }
          });

      __syncthreads();
      // read partial sum and reduce in warp0
      accum_t row_sum(0.0);
      if (thread_id < kQueriesPerBlock) {

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < WarpCount; i++) {
          row_sum += s_prime[thread_id + i * kQueriesPerBlock];
        }
        s_prime[thread_id] = row_sum;

        //reset other warp's shared data to 0
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < WarpCount; i++) {
          s_prime[thread_id + i * kQueriesPerBlock] = accum_t(0);
        }
      }
    }
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterPPU0010
    : RegisterOps<
          AttentionScalingCoefsUpdaterPPU0010<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  static int const kElementsPerAccess = InstructionShape::kN / 4;
  static int const kRowsPerTile = 8;
  static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
#if SAIL_PPU_MMA
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad +
            tile_offset.column() * Shape::kColumn);
#else
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad * kElementsPerAccess +
            tile_offset.column() * Shape::kColumn);
#endif
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    // See cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kAccumulatorRows; ++row) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            row * kRowsPerTile + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
              (mma_n * Policy::MmaIterations::kRow + mma_m);
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
#if SAIL_PPU_MMA
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col * 4 + lane_offset.column();
#else
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col + lane_offset.column();
#endif
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            op(accum_m, accum_n, idx);
          }
        }

        endRow(accum_m);
      }
    }
  }

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // In each warp, 4 threads will work on the same row
    // - the ones with the same `quad`
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1);
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 2);
    myValue = fn(myValue, otherV);
    int lane_in_quad = (lane_id & 3);
    return lane_in_quad == 0;
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSimt
    : RegisterOps<
          AttentionScalingCoefsUpdaterSimt<T, accum_t, kWarpSize>,
          T,
          accum_t,
          kWarpSize> {
  using Policy = typename T::Policy;
  using Iterations = typename T::Iterations;
  using Element = typename T::Element;
  using Delta = typename T::Delta;
  using Shape = typename T::Shape;
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    CUTLASS_PRAGMA_UNROLL
    for (int bit = 1; bit < Policy::WarpShape::kColumn; bit *= 2) {
      auto otherV = __shfl_xor_sync(0xffffffff, myValue, bit);
      myValue = fn(myValue, otherV);
    }
    return (lane_id & (Policy::WarpShape::kColumn - 1)) == 0;
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
        int accum_m = mma_m * Delta::kRow + m + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
          int accum_n =
              mma_n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN +
              lane_offset.column();
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
            int idx = n +
                Policy::LaneMmaShape::kN *
                    (mma_n +
                     Iterations::kColumn *
                         (m + mma_m * Policy::LaneMmaShape::kM));
            op(accum_m, accum_n + n, idx);
          }
        }
        endRow(accum_m);
      }
    }
  }

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    static_assert(
        cutlass::platform::is_same<
            typename Policy::LaneLayout,
            cutlass::layout::RowMajorInterleaved<1>>::value,
        "");
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    cutlass::MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
        cutlass::MatrixCoord(Policy::LaneMmaShape::kM,
                             Policy::LaneMmaShape::kN);
    return lane_offset +
        tile_offset * cutlass::MatrixCoord(Shape::kRow, Shape::kColumn);
  }
};

template <typename T, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater;

// Simt
template <typename S, typename P, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaSimtTileIterator<
        S,
        cutlass::gemm::Operand::kC,
        accum_t,
        cutlass::layout::RowMajor,
        P,
        1,
        1>,
    accum_t,
    kWarpSize> {
  using Iterator = typename cutlass::gemm::warp::MmaSimtTileIterator<
      S,
      cutlass::gemm::Operand::kC,
      accum_t,
      cutlass::layout::RowMajor,
      P,
      1,
      1>;
  using Updater =
      AttentionScalingCoefsUpdaterSimt<Iterator, accum_t, kWarpSize>;
};

// TensorOp - PPU+
template <
    typename S1,
    typename S2,
    typename S3,
    typename accum_t,
    int kWarpSize>
struct DefaultAttentionScalingCoefsUpdater<
    cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
        S1,
        accum_t,
        cutlass::layout::RowMajor,
        S2,
        S3>,
    accum_t,
    kWarpSize> {
  using Iterator =
      typename cutlass::gemm::warp::MmaTensorOpAccumulatorTileIterator<
          S1,
          accum_t,
          cutlass::layout::RowMajor,
          S2,
          S3>;
  using Updater =
      AttentionScalingCoefsUpdaterPPU0010<Iterator, accum_t, kWarpSize>;
};
