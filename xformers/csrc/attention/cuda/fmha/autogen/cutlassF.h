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
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "../kernel_forward.h"
// ======== bf16 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_ppu0010(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x128_rf_ppu0010(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_ppu0010(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_bf16_ppu0010(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>(), fmha_cutlassF_bf16_aligned_64x64_rf_ppu0010);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>(), fmha_cutlassF_bf16_aligned_64x128_rf_ppu0010);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>(), fmha_cutlassF_bf16_aligned_32x128_gmem_ppu0010);
}

// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f16 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f16 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_ppu0010(typename AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x128_rf_ppu0010(typename AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_ppu0010(typename AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_ppu0010(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_ppu0010);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_64x128_rf_ppu0010);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_ppu0010);
}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f32 / ppu ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)

}

// ======== f32 / ppu0010 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x128_rf_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_ppu0010(typename AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_ppu0010(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_ppu0010);
    cb(AttentionKernel<float, cutlass::arch::PPU0010, true, 64, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_64x128_rf_ppu0010);
    cb(AttentionKernel<float, cutlass::arch::PPU0010, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_ppu0010);
}


template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc = 0) {

    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_bf16_ppu0010(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
    }
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
    }
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f16_ppu0010(cb, cc);
    }
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
    }
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
    }
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f32_ppu0010(cb, cc);
    }
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
