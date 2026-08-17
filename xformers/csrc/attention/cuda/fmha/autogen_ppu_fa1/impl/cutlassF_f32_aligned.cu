/*
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "accutlass.h"
#include "extension/fmha/kernel_forward.h"
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH >= 800
#if COMPATIBLE_ARCH < 1000
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_rf_ppu0010` is for ppu0010+, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH >= 800
#if COMPATIBLE_ARCH < 1000
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_32x128_rf_ppu0010` is for ppu0010+, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
}

__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH >= 800
#if COMPATIBLE_ARCH < 1000
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_32x128_gmem_ppu0010` is for ppu0010+, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
