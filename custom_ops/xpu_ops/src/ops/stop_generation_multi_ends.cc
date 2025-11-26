// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <fcntl.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include "paddle/extension.h"
#include "xpu/plugin.h"

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens,
                       const paddle::Tensor &pre_ids,
                       const paddle::Tensor &step_idx,
                       const paddle::Tensor &stop_seqs,
                       const paddle::Tensor &stop_seqs_len,
                       const bool beam_search) {
  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
  PD_CHECK(pre_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_seqs.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_seqs_len.dtype() == paddle::DataType::INT32);
  PD_CHECK(step_idx.dtype() == paddle::DataType::INT64);
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  std::vector<int64_t> shape = topk_ids.shape();
  int64_t bs_now = shape[0];
  int64_t end_length = end_ids.shape()[0];
  int64_t stop_seqs_bs = stop_seqs.shape()[1];
  int64_t stop_seqs_max_len = stop_seqs.shape()[2];
  int64_t pre_ids_len = pre_ids.shape()[1];
  PD_CHECK(stop_seqs_len.shape()[0] == bs_now,
           "stop_seqs_len batch size must match topk_ids.");
  PD_CHECK(stop_seqs_len.shape()[1] == stop_seqs_bs,
           "stop_seqs_len shape mismatch with stop_seqs.");
  PD_CHECK(step_idx.shape()[0] == bs_now,
           "step_idx batch size must match topk_ids.");
  int r = baidu::xpu::api::plugin::set_stop_value_multi_ends<int64_t>(
      xpu_ctx->x_context(),
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(topk_ids.data<int64_t>()),
      const_cast<int64_t *>(next_tokens.data<int64_t>()),
      end_ids.data<int64_t>(),
      seq_lens.data<int>(),
      pre_ids.data<int64_t>(),
      static_cast<int>(pre_ids_len),
      step_idx.data<int64_t>(),
      stop_seqs.data<int64_t>(),
      stop_seqs_len.data<int>(),
      bs_now,
      end_length,
      static_cast<int>(stop_seqs_bs),
      static_cast<int>(stop_seqs_max_len),
      beam_search);
  PD_CHECK(r == 0, "xpu::plugin::set_stop_value_multi_ends failed.");
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids",
             "stop_flags",
             "seq_lens",
             "end_ids",
             "next_tokens",
             "pre_ids",
             "step_idx",
             "stop_seqs",
             "stop_seqs_len"})
    .Attrs({"beam_search: bool"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti));
