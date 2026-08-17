/*
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "accutlass.h"
#include "extension/fmha/kernel_backward.h"
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32, true>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH >= 800
#if COMPATIBLE_ARCH < 1000
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_ppu0010` is for ppu0010+, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH >= 800
#if COMPATIBLE_ARCH < 1000
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::half_t, true, false, true, 128, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_128x128_k32_ppu0010` is for ppu0010+, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
