/*
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"

#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "accutlass.h"
#include "extension/fmha/kernel_backward.h"
#include "extension/fmha/kernel_backward_fa2.h"
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::attention_kernel(p);
  return;
#elif COMPATIBLE_ARCH == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassB_bf16_aligned_128x128_k65536_ppu0010` is for ppu0010/ppu0015, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
#endif  // COMPATIBLE_ARCH
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k65536_ppu0010(typename AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::PPU0010, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::attention_kernel(p);
  return;
#elif COMPATIBLE_ARCH == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassB_bf16_aligned_64x128_k65536_ppu0010` is for ppu0010/ppu0015, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
#endif  // COMPATIBLE_ARCH
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
