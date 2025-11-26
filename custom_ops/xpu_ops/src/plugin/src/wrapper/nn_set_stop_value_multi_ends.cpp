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

#include <algorithm>
#include <numeric>

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu3 {
namespace plugin {
template <typename T>
__attribute__((global)) void set_stop_value_multi_ends(
    bool *stop_flags,
    T *topk_ids,
    T *next_tokens,
    const T *end_ids,
    const int *seq_lens,
    const T *pre_ids,
    const int pre_ids_len,
    const T *step_idx,
    const T *stop_seqs,
    const int *stop_seqs_len,
    const int bs,
    const int end_length,
    const int stop_seqs_bs,
    const int stop_seqs_max_len,
    const bool beam_search,
    const bool prefill_one_step_stop);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
__inline__ bool is_in_end(const T id, const T *end_ids, int length) {
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return false;
}

template <typename T>
static int cpu_wrapper(Context *ctx,
                       bool *stop_flags,
                       T *topk_ids,
                       T *next_tokens,
                       const T *end_ids,
                       const int *seq_lens,
                       const T *pre_ids,
                       const T *step_idx,
                       const T *stop_seqs,
                       const int *stop_seqs_len,
                       const int bs,
                       const int end_length,
                       const int stop_seqs_bs,
                       const int stop_seqs_max_len,
                       const int pre_ids_len,
                       const bool beam_search,
                       const bool prefill_one_step_stop) {
  for (int i = 0; i < bs; i++) {
    if (prefill_one_step_stop) {
      stop_flags[i] = true;
      if (seq_lens[i] == 0) {
        topk_ids[i] = -1;
      }
      next_tokens[i] = topk_ids[i];
    } else {
      if (stop_flags[i]) {
        if (seq_lens[i] == 0) {
          topk_ids[i] = -1;
        } else {
          topk_ids[i] = end_ids[0];
          next_tokens[i] = end_ids[0];
        }
      } else {
        next_tokens[i] = topk_ids[i];
      }
      if (!beam_search && is_in_end(topk_ids[i], end_ids, end_length)) {
        stop_flags[i] = true;
        topk_ids[i] = end_ids[0];
        next_tokens[i] = end_ids[0];
      }
    }
  }
  // Handle stop sequences
  for (int bid = 0; bid < bs; bid++) {
    if (stop_flags[bid]) continue;
    const T* pre_ids_now = pre_ids + bid * pre_ids_len;
    const T step_idx_now = step_idx[bid];
    for (int tid = 0; tid < stop_seqs_bs; tid++) {
      const int stop_seq_len = stop_seqs_len[bid * stop_seqs_bs + tid];
      if (stop_seq_len <= 0) continue;
      const T* stop_seq_now = stop_seqs + bid * stop_seqs_bs * stop_seqs_max_len + tid * stop_seqs_max_len;
      bool is_end = true;
      int count = 1;
      for (int k = stop_seq_len - 1; k >= 0; --k) {
        if ((step_idx_now - count) < 0 ||
            pre_ids_now[step_idx_now - count++] != stop_seq_now[k]) {
          is_end = false;
          break;
        }
      }
      if (is_end) {
        stop_flags[bid] = true;
        topk_ids[bid] = end_ids[0];
        next_tokens[bid] = end_ids[0];
        break;
      }
    }
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(Context *ctx,
                        bool *stop_flags,
                        T *topk_ids,
                        T *next_tokens,
                        const T *end_ids,
                        const int *seq_lens,
                        const T *pre_ids,
                        const T *step_idx,
                        const T *stop_seqs,
                        const int *stop_seqs_len,
                        const int bs,
                        const int end_length,
                        const int stop_seqs_bs,
                        const int stop_seqs_max_len,
                        const int pre_ids_len,
                        const bool beam_search,
                        const bool prefill_one_step_stop) {
  using XPU_TID = typename XPUIndexType<T>::type;
  auto set_stop_value_multi_ends =
      xpu3::plugin::set_stop_value_multi_ends<XPU_TID>;
  set_stop_value_multi_ends<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      stop_flags,
      reinterpret_cast<XPU_TID *>(topk_ids),
      reinterpret_cast<XPU_TID *>(next_tokens),
      reinterpret_cast<const XPU_TID *>(end_ids),
      seq_lens,
      reinterpret_cast<const XPU_TID *>(pre_ids),
      pre_ids_len,
      reinterpret_cast<const XPU_TID *>(step_idx),
      reinterpret_cast<const XPU_TID *>(stop_seqs),
      stop_seqs_len,
      bs,
      end_length,
      stop_seqs_bs,
      stop_seqs_max_len,
      beam_search,
      prefill_one_step_stop);
  return api::SUCCESS;
}

template <typename T>
int set_stop_value_multi_ends(Context *ctx,
                              bool *stop_flags,
                              T *topk_ids,
                              T *next_tokens,
                              const T *end_ids,
                              const int *seq_lens,
                              const T *pre_ids,
                              const T *step_idx,
                              const T *stop_seqs,
                              const int *stop_seqs_len,
                              const int bs,
                              const int end_length,
                              const int stop_seqs_bs,
                              const int stop_seqs_max_len,
                              const int pre_ids_len,
                              const bool beam_search) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "set_stop_value_multi_ends", T);
  WRAPPER_DUMP_PARAM5(ctx, stop_flags, topk_ids, next_tokens, end_ids, seq_lens);
  WRAPPER_DUMP_PARAM4(ctx, pre_ids, step_idx, stop_seqs, stop_seqs_len);
  WRAPPER_DUMP_PARAM6(ctx, bs, end_length, stop_seqs_bs, stop_seqs_max_len, pre_ids_len, beam_search);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, bool, bs, stop_flags);
  WRAPPER_CHECK_PTR(ctx, T, bs, topk_ids);
  WRAPPER_CHECK_PTR(ctx, T, end_length, end_ids);
  WRAPPER_CHECK_PTR(ctx, T, bs, seq_lens);
  WRAPPER_CHECK_PTR(ctx, T, bs * pre_ids_len, pre_ids);
  WRAPPER_CHECK_PTR(ctx, T, bs, step_idx);
  WRAPPER_CHECK_PTR(ctx, T, bs * stop_seqs_bs * stop_seqs_max_len, stop_seqs);
  WRAPPER_CHECK_PTR(ctx, int, bs * stop_seqs_bs, stop_seqs_len);
  WRAPPER_ASSERT_LE(ctx, end_length, 1024);  // assume end_length <= 1024
  bool prefill_one_step_stop = false;
  if (const char *env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
    // std::cout << "Your PATH is: " << env_p << '\n';
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          stop_flags,
                          topk_ids,
                          next_tokens,
                          end_ids,
                          seq_lens,
                          pre_ids,
                          step_idx,
                          stop_seqs,
                          stop_seqs_len,
                          bs,
                          end_length,
                          stop_seqs_bs,
                          stop_seqs_max_len,
                          pre_ids_len,
                          beam_search,
                          prefill_one_step_stop);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(ctx,
                           stop_flags,
                           topk_ids,
                           next_tokens,
                           end_ids,
                           seq_lens,
                           pre_ids,
                           step_idx,
                           stop_seqs,
                           stop_seqs_len,
                           bs,
                           end_length,
                           stop_seqs_bs,
                           stop_seqs_max_len,
                           pre_ids_len,
                           beam_search,
                           prefill_one_step_stop);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int set_stop_value_multi_ends<int64_t>(Context *ctx,
                                                bool *stop_flags,
                                                int64_t *topk_ids,
                                                int64_t *next_tokens,
                                                const int64_t *end_ids,
                                                const int *seq_lens,
                                                const int64_t *pre_ids,
                                                const int64_t *step_idx,
                                                const int64_t *stop_seqs,
                                                const int *stop_seqs_len,
                                                const int bs,
                                                const int end_length,
                                                const int stop_seqs_bs,
                                                const int stop_seqs_max_len,
                                                const int pre_ids_len,
                                                const bool beam_search);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
