/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/kernels/portable/cpu/util/arange_util.h>

#include <ctime>

#include <executorch/extension/llm/runner/stats.h>

namespace executorch {
namespace extension {
namespace llm {

// [한글 주석] 생성자: Module과 IOManager의 원시 포인터를 참조로 저장
// TextDecoderRunner는 이들의 소유권을 가지지 않음
// (TextLLMRunner가 module_과 io_manager_의 unique_ptr를 소유)
//
// 참고: iPhone 15에서 FileDataLoader가 MmapDataLoader+UseMlockIgnoreErrors 대비
//       ~2배 로딩 성능 향상, Galaxy S22에서 ~5% 향상 관측됨
TextDecoderRunner::TextDecoderRunner(
    Module* module,
    IOManager* io_manager,
    std::string method_name)
    : module_(module),
      io_manager_(io_manager),
      method_name_(std::move(method_name)) {}

// ============================================================================
// [한글 주석] TextDecoderRunner::step() — 모델 Forward Pass 1회 실행
// ============================================================================
// 이 함수는 "함수형(functional)"으로, 입력 상태를 변경하지 않음
// 동일한 입력으로 여러 번 호출해도 안전
// 상태 관리(pos_ 업데이트 등)는 호출자(TextPrefiller, TextTokenGenerator)의 책임
//
// [예시 — Prefill 호출 시]
//   tokens: shape [1, 7], data=[450, 1234, 304, 278, 8494, 1139, 338]
//   start_pos: 0
//   반환: logits 텐서 shape [1, 7, 32000]
//
// [예시 — Decode 호출 시 (생성 루프)]
//   tokens: shape [1, 1], data=[29871]
//   start_pos: 7
//   반환: logits 텐서 shape [1, 1, 32000]
::executorch::runtime::Result<executorch::aten::Tensor> TextDecoderRunner::step(
    TensorPtr& tokens,
    int64_t start_pos) {

  // [단계 1] 메서드 메타데이터 조회
  // method_meta에서 입력 개수, 텐서 shape 등의 정보를 얻음
  auto method_meta_result = module_->method_meta(method_name_);
  if (!method_meta_result.ok()) {
    return method_meta_result.error();
  }
  auto method_meta = std::move(*method_meta_result);

  // [단계 2] KV 캐시 사용 여부 판단
  // 입력이 2개 이상이면 KV 캐시 사용 (token_ids + cache_position)
  // 입력이 1개면 KV 캐시 미사용 (token_ids만)
  // 예시: num_inputs = 2 → use_kv_cache = true
  bool use_kv_cache = method_meta.num_inputs() > 1;

  std::vector<int64_t> cache_positions;

  if (use_kv_cache) {
    // ══════════════════════════════════════════════════════════════
    // [경로 A] KV 캐시 사용 — 일반적인 LLM 추론 경로
    // ══════════════════════════════════════════════════════════════

    // [A-1] 캐시 위치 텐서 생성
    // populate_start_pos_or_cache_position() (util.h):
    //   - method_meta의 두 번째 입력 텐서 shape 조회
    //   - numel == 1이면: start_pos 스칼라를 shape [1] 텐서로 감쌈
    //     예시 (Prefill): start_pos_tensor = [0]
    //     예시 (Decode):  start_pos_tensor = [7]
    //   - numel > 1이면: [start_pos, start_pos+1, ..., start_pos+seq_len-1]
    //     배열 생성 (cache_position 방식을 사용하는 모델용)
    //     예시 (Prefill, numel=7): [0, 1, 2, 3, 4, 5, 6]
    auto start_pos_tensor_result = populate_start_pos_or_cache_position(
        module_,
        start_pos,
        cache_positions,
        tokens->numel(),      // Prefill: 7, Decode: 1
        method_name_.c_str()); // "forward"
    if (!start_pos_tensor_result.ok()) {
      return start_pos_tensor_result.error();
    }
    auto start_pos_tensor = std::move(*start_pos_tensor_result);

    // [A-2] IOManager를 통해 입력 준비
    // 기본 CPU IOManager:
    //   method_meta.num_inputs() == 2 확인
    //   return {tokens_tensor, start_pos_tensor}
    //
    // Prefill 예시:
    //   inputs[0] = tokens: shape [1, 7], data=[450, 1234, ..., 338]
    //   inputs[1] = start_pos: shape [1], data=[0]
    //
    // Decode 예시:
    //   inputs[0] = tokens: shape [1, 1], data=[29871]
    //   inputs[1] = start_pos: shape [1], data=[7]
    std::vector<runtime::EValue> inputs;
    auto inputs_res =
        io_manager_->prepare_decode(tokens, start_pos_tensor, method_name_);
    ET_CHECK_OK_OR_RETURN_ERROR(inputs_res.error());
    inputs = inputs_res.get();

    // [A-3] 모델 실행 — ExecuTorch 런타임을 통한 forward pass
    // module_->execute("forward", inputs):
    //   (1) load_method("forward") — 이미 로드됨, 스킵
    //   (2) method->set_inputs(inputs) — 입력 텐서 설정
    //   (3) method->execute() — ExecuTorch 실행 계획에 따라 연산 수행
    //       → XNNPACK delegate가 처리하는 연산들:
    //         - Token Embedding: [1,7] → [1,7,4096] (Prefill)
    //         - RoPE 위치 인코딩
    //         - Multi-Head Self-Attention (32 heads, head_dim=128)
    //           - Q = X @ W_q, K = X @ W_k, V = X @ W_v
    //           - Attention = softmax(Q @ K^T / sqrt(128)) @ V
    //           - KV 캐시에 K, V 저장 (position 0~6)
    //         - Feed-Forward Network (SwiGLU activation)
    //         - RMSNorm
    //         - ... (32개 트랜스포머 레이어 반복)
    //         - Output projection: [1,7,4096] → [1,7,32000] (logits)
    //   (4) method->get_outputs() → [{logits tensor}]
    //
    // Prefill 결과: logits shape [1, 7, 32000]
    //   각 위치(0~6)에서 다음 토큰에 대한 32000개 점수
    //   position 6의 logits가 프롬프트 다음 토큰을 예측
    //
    // Decode 결과: logits shape [1, 1, 32000]
    //   현재 위치에서 다음 토큰에 대한 32000개 점수
    auto outputs_res = module_->execute(method_name_, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());

    // [A-4] 출력 후처리
    // 기본 CPU IOManager의 update_decode(): no-op
    // (KV 캐시는 모델 내부에서 자동 관리됨)
    // 커스텀 IOManager는 여기서 GPU→CPU 전송 등을 수행할 수 있음
    auto update_err =
        io_manager_->update_decode(outputs_res.get(), method_name_);
    ET_CHECK_OK_OR_RETURN_ERROR(update_err);

    // [A-5] 출력 검증
    // 출력이 정확히 1개의 텐서여야 함 (logits)
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More than one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // logits 텐서 반환
    // Prefill: [1, 7, 32000] — 모든 위치의 예측 (마지막 위치만 사용됨)
    // Decode:  [1, 1, 32000] — 현재 위치의 예측
    return outputs_res.get()[0].toTensor();

  } else {
    // ══════════════════════════════════════════════════════════════
    // [경로 B] KV 캐시 없음 — 전체 시퀀스를 매번 입력
    // ══════════════════════════════════════════════════════════════
    // start_pos는 사용되지 않음 (KV 캐시가 없으므로 위치 추적 불필요)
    // 매 step마다 전체 시퀀스를 입력해야 하므로 O(n²) 복잡도
    (void)start_pos;

    // 토큰만 입력 (cache_position 없음)
    std::vector<runtime::EValue> inputs{tokens};
    auto outputs_res = module_->execute(method_name_, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More than one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    return outputs_res.get()[0].toTensor();
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch
