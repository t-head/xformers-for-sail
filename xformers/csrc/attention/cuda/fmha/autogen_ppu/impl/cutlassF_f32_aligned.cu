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
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "accutlass.h"
#include "extension/fmha/kernel_forward.h"
#include "extension/fmha/kernel_forward_fa2.h"
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_k64_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true, 64>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, true, true, true, 64>::attention_kernel(p);
  return;
#elif COMPATIBLE_ARCH == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_k64_rf_ppu0010` is for ppu0010/ppu0015, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
#endif  // COMPATIBLE_ARCH
}
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_k128_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true, 128>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, true, true, true, 128>::attention_kernel(p);
  return;
#elif COMPATIBLE_ARCH == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_32x128_k128_rf_ppu0010` is for ppu0010/ppu0015, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
#endif  // COMPATIBLE_ARCH
}
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_k65536_gmem_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true, 65536>::Params p) {
#ifdef COMPATIBLE_ARCH
#if COMPATIBLE_ARCH == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, false, true, true, 65536>::attention_kernel(p);
  return;
#elif COMPATIBLE_ARCH == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f32_aligned_32x128_k65536_gmem_ppu0010` is for ppu0010/ppu0015, but was built for sm%d\n",
        int(COMPATIBLE_ARCH + 0) / 10);
#endif
#endif  // COMPATIBLE_ARCH
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
