/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a text prompt, encode it using tokenizer and prefill the KV cache of a
// LLM.

#include <executorch/extension/llm/runner/text_prefiller.h>
#include <algorithm>

namespace executorch {
namespace extension {
namespace llm {

TextPrefiller::TextPrefiller(
    TextDecoderRunner* text_decoder_runner,
    bool use_kv_cache,
    bool enable_parallel_prefill,
    int64_t max_seq_len)
    : text_decoder_runner_(text_decoder_runner),
      use_kv_cache_(use_kv_cache),
      enable_parallel_prefill_(enable_parallel_prefill),
      max_seq_len_(max_seq_len > 0 ? max_seq_len : 128) {}

// ============================================================================
// [한글 주석] TextPrefiller::prefill() — 프롬프트 Prefill 처리
// ============================================================================
// 프롬프트의 모든 토큰을 모델에 입력하여 KV 캐시를 채우고,
// 프롬프트 다음에 올 첫 번째 토큰을 예측하여 반환
//
// [예시 가정값]
//   prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338] (7개)
//   start_pos = 0 (KV 캐시의 시작 위치)
//   max_seq_len_ = 2048
//
// [동작 흐름]
//   num_prompt_tokens(7) <= max_seq_len_(2048)이므로 chunking 없이 처리
//   → prefill_chunk(prompt_tokens, start_pos=0) 호출
//   → 반환: 첫 예측 토큰 (예: 29871)
//   → start_pos가 참조로 전달되어 7로 업데이트됨
::executorch::runtime::Result<uint64_t> TextPrefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");
  if (!text_decoder_runner_->is_method_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
  }

  int32_t num_prompt_tokens = prompt_tokens.size();
  // 예시: num_prompt_tokens = 7

  // ── Chunking 판단 ──
  // 프롬프트 토큰 수가 max_seq_len_을 초과하면 청크로 분할
  // 예시: 7 <= 2048이므로 else 분기 (직접 처리)
  //
  // 만약 프롬프트가 5000 토큰이고 max_seq_len=2048이면:
  //   청크1: tokens[0:2048] → prefill_chunk → start_pos += 2048
  //   청크2: tokens[2048:4096] → prefill_chunk → start_pos += 2048
  //   청크3: tokens[4096:5000] → prefill_chunk → start_pos += 904
  //   마지막 청크의 예측 토큰을 반환
  if (num_prompt_tokens > max_seq_len_) {
    uint64_t cur_token = 0;
    int num_tokens_to_process = 0;

    while (num_tokens_to_process < num_prompt_tokens) {
      // 이번 청크에서 처리할 토큰 수 = min(남은 토큰, max_seq_len)
      auto num_tokens_to_prefill_with = std::min<int>(
          num_prompt_tokens - num_tokens_to_process, max_seq_len_);

      // 청크 토큰 복사
      std::vector<uint64_t> prompt_tokens_to_process(
          num_tokens_to_prefill_with);
      std::copy(
          prompt_tokens.begin() + num_tokens_to_process,
          prompt_tokens.begin() + num_tokens_to_process +
              num_tokens_to_prefill_with,
          prompt_tokens_to_process.begin());

      // 청크 처리 → KV 캐시 채움 → start_pos 업데이트
      auto chunk_result = prefill_chunk(prompt_tokens_to_process, start_pos);
      ET_CHECK_OK_OR_RETURN_ERROR(chunk_result.error());
      cur_token = chunk_result.get();

      num_tokens_to_process += num_tokens_to_prefill_with;
    }

    return cur_token;  // 마지막 청크의 예측 토큰
  } else {
    // 토큰 수가 max_seq_len_ 이하 → 한 번에 처리
    // 예시: prefill_chunk([450, 1234, 304, 278, 8494, 1139, 338], start_pos=0)
    return prefill_chunk(prompt_tokens, start_pos);
  }
}

// ============================================================================
// [한글 주석] TextPrefiller::prefill_chunk() — 실제 Prefill 실행
// ============================================================================
// 두 가지 모드로 동작:
//   (A) Parallel Prefill: enable_parallel_prefill_=true 또는 KV캐시 미사용
//       → 모든 토큰을 shape [1, N]으로 묶어 한 번에 모델 실행
//       → 효율적 (GPU/XNNPACK에서 배치 처리 가능)
//   (B) Sequential Prefill: use_kv_cache_=true && enable_parallel_prefill_=false
//       → 토큰을 하나씩 순차적으로 모델에 입력
//       → 모델이 incremental KV 캐시만 지원할 때 사용
//
// [예시 — Parallel Prefill (일반적인 경우)]
//   prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338] (7개)
//   start_pos = 0
//   enable_parallel_prefill_ = true (enable_dynamic_shape=true)
//   use_kv_cache_ = true
::executorch::runtime::Result<uint64_t> TextPrefiller::prefill_chunk(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  // KV 캐시를 사용하지 않으면 start_pos는 무시됨
  int32_t num_prompt_tokens = prompt_tokens.size();
  // 예시: num_prompt_tokens = 7

  uint64_t cur_token;

  if (enable_parallel_prefill_ || !use_kv_cache_) {
    // ══════════════════════════════════════════════════════════════
    // [경로 A] Parallel Prefill — 모든 토큰을 한 번에 처리
    // ══════════════════════════════════════════════════════════════

    // [A-1] 토큰 텐서 생성
    // from_blob: 기존 데이터 버퍼를 감싸는 텐서 생성 (복사 없음)
    // shape = {1, 7} (batch=1, seq_len=7)
    // dtype = Long (int64_t)
    // 데이터: [[450, 1234, 304, 278, 8494, 1139, 338]]
    auto tokens = from_blob(
        prompt_tokens.data(),
        {1, num_prompt_tokens},
        executorch::aten::ScalarType::Long);

    // [A-2] 모델 forward pass 실행
    // text_decoder_runner_->step(tokens, start_pos=0) 내부:
    //   (1) module_->method_meta("forward") → num_inputs=2 → use_kv_cache=true
    //   (2) populate_start_pos_or_cache_position() → start_pos 텐서 [0]
    //   (3) io_manager_->prepare_decode(tokens[1,7], pos[1]) → [{tokens}, {pos}]
    //   (4) module_->execute("forward", inputs)
    //       → XNNPACK delegate가 LLaMA의 전체 forward pass 실행:
    //         embedding → 32 transformer layers → output projection
    //       → KV 캐시의 position 0~6에 key/value 벡터 저장
    //       → logits 텐서 반환: shape [1, 7, 32000]
    //         [1, 7, 32000]의 의미:
    //           dim0=1: 배치 크기
    //           dim1=7: 각 입력 위치에 대한 예측
    //           dim2=32000: 각 위치에서 vocab의 모든 토큰에 대한 점수
    //   (5) io_manager_->update_decode() → no-op
    auto outputs_res = text_decoder_runner_->step(tokens, start_pos);

    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_LOG(
        Info, "Prefill token result numel(): %zu", outputs_res.get().numel());
    // 예시: numel() = 1 * 7 * 32000 = 224000

    // [A-3] start_pos 업데이트 (참조로 전달 → 호출자의 pos_도 업데이트)
    // start_pos: 0 → 7
    // 이제 KV 캐시에는 position 0~6이 채워져 있음
    start_pos += num_prompt_tokens;

    // [A-4] Logits → Token 변환
    // logits_to_token(logits_tensor [1, 7, 32000]):
    //   (1) 3D 텐서이므로 마지막 시퀀스 위치(position 6)의 logits 추출
    //       → logits 포인터를 (7-1) * 32000 = 192000 오프셋으로 이동
    //       → shape [32000]의 1D logits 배열
    //   (2) Sampler(vocab_size=32000, temperature=0.0) 생성
    //       (prefill에서는 temp=0, 즉 greedy 샘플링)
    //   (3) sample_argmax(logits) → 가장 높은 logit의 인덱스 반환
    //       → 예: 29871 (토크나이저에서 " 42"에 해당)
    cur_token = text_decoder_runner_->logits_to_token(outputs_res.get());
    // 예시: cur_token = 29871

  } else {
    // ══════════════════════════════════════════════════════════════
    // [경로 B] Sequential Prefill — 토큰을 하나씩 순차 처리
    // ══════════════════════════════════════════════════════════════
    // 이 경로는 enable_parallel_prefill_=false일 때 사용
    // 모델이 동적 입력 shape을 지원하지 않는 경우에 해당
    //
    // [예시]
    //   prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338]
    //   반복 0: step(token=450, pos=0) → logits → 무시
    //   반복 1: step(token=1234, pos=1) → logits → 무시
    //   ...
    //   반복 6: step(token=338, pos=6) → logits → 샘플링 → cur_token
    //   총 7번의 forward pass 필요 (parallel의 1번 대비 7배 느림)

    int64_t pos = 0;
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    cur_token = prompt_tokens[0];  // 첫 토큰: 450

    // shape [1, 1]의 토큰 텐서 생성
    // cur_token 변수의 주소를 직접 참조하므로 cur_token 변경 시 텐서 데이터도 변경됨
    auto tokens =
        from_blob(&cur_token, {1, 1}, executorch::aten::ScalarType::Long);

    // 첫 번째 토큰 실행 (BOS 토큰으로 가정하여 결과는 무시)
    // step(token=[[450]], start_pos=0) → logits [1, 1, 32000]
    auto logits_result = text_decoder_runner_->step(tokens, start_pos);
    if (!logits_result.ok()) {
      return logits_result.error();
    }
    auto logits_tensor = std::move(*logits_result);

    pos += 1;        // pos: 0 → 1
    start_pos += 1;  // start_pos: 0 → 1

    // 나머지 토큰들을 순차적으로 처리
    while (pos < num_prompt_tokens) {
      // 다음 토큰으로 업데이트 (cur_token 변수 변경 → 텐서 데이터도 자동 업데이트)
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      cur_token = prompt_tokens[pos];
      // 예: pos=1일 때 cur_token = 1234, tokens = [[1234]]

      // step(token=[[1234]], start_pos=1) → logits [1, 1, 32000]
      // KV 캐시의 position 1에 key/value 저장
      auto step_result = text_decoder_runner_->step(tokens, start_pos);
      if (!step_result.ok()) {
        return step_result.error();
      }
      logits_tensor = std::move(*step_result);

      pos++;        // 다음 위치로
      start_pos++;  // KV 캐시 위치도 증가
    }

    // 마지막 토큰(position 6)의 logits에서 다음 토큰 샘플링
    // logits_tensor [1, 1, 32000] → sample → cur_token = 29871
    cur_token = text_decoder_runner_->logits_to_token(logits_tensor);
  }

  // 반환: 프롬프트 다음에 올 첫 번째 예측 토큰
  // 예시: 29871
  return cur_token;
}

} // namespace llm
} // namespace extension
} // namespace executorch
