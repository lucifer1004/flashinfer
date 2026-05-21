// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

// ModelType determines KV cache layout, dimensions, and scale format.
// V32:    DeepSeek V3.2, GLM 5.1 — d_nope=512, FP32 scale inline, 656B/token
// MODEL1: DeepSeek V4 Flash/Pro  — d_nope=448, UE8M0 scale footer, 584B/token
enum class ModelType { V32, MODEL1 };

// ComputeMode determines the MMA precision path.
//
// FP8:  QK and XV use FP8 MMA (block-scaled with UE8M0).
//       Highest throughput. Q is quantized to FP8 on the fly.
//       KV remains FP8 in smem — no dequant needed.
//
// BF16: QK and XV use BF16 MMA. FP8 KV is dequantized to BF16 in smem.
//       Matches FlashMLA's precision behavior (which always uses BF16 MMA).
//       ~40-50% throughput of FP8 path, but higher accuracy.
//       FlashMLA's prefill always uses this mode.
//
// For FlashMLA compatibility, the default should match FlashMLA:
//   - prefill: BF16 (matching FlashMLA sparse_fwd)
//   - decode: configurable (FP8 for performance, BF16 for precision)
enum class ComputeMode { FP8, BF16 };
