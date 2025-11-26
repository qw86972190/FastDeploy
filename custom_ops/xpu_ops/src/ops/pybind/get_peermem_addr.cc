// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ops/pybind/cuda_compat.h"
#include "paddle/extension.h"
#include "xpu/runtime.h"

uintptr_t xpu_get_peer_mem_addr(uintptr_t ptr) {
#if FASTDEPLOY_XPU_HAS_CUDA
  struct cudaPointerAttributes pointerAttr;
  cudaPointerGetAttributes(&pointerAttr, reinterpret_cast<void*>(ptr));
  PD_CHECK(pointerAttr.hostPointer != nullptr,
           "Failed to get host pointer from device pointer");
  uintptr_t ptr_out = reinterpret_cast<uintptr_t>(pointerAttr.hostPointer);
  return ptr_out;
#else
  (void)ptr;
  PD_THROW("cudaPointerGetAttributes is not available in this build.");
#endif
}
