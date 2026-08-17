/*
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../kernel_backward.h"
// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== bf16 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k96_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_bf16_ppu0010(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_ppu0010);
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_bf16_aligned_128x64_k96_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_ppu0010);
}

// ======== f16 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k96_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_ppu0010(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_ppu0010);
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_f16_aligned_128x64_k96_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_ppu0010);
}

// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)

}

// ======== f32 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k32_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k64_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x64_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k128_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k32_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k64_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x64_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k128_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k65536_dropout_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_ppu0010(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 128>(), fmha_cutlassB_f32_aligned_128x64_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f32_aligned_128x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 32>(), fmha_cutlassB_f32_aligned_64x64_k32_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 64>(), fmha_cutlassB_f32_aligned_64x64_k64_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 128>(), fmha_cutlassB_f32_aligned_128x64_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 128>(), fmha_cutlassB_f32_aligned_64x64_k128_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f32_aligned_128x64_k65536_dropout_ppu0010);
    cb(AttentionBackwardKernel<cutlass::arch::PPU0010, float, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f32_aligned_64x64_k65536_dropout_ppu0010);
}


template <typename DT, typename T>
void dispatch_cutlassB(T cb, int cc = 0) {

    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_bf16_ppu0010(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_f16_ppu0010(cb, cc);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_f32_ppu0010(cb, cc);
    }
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
