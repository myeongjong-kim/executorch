/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

// ============================================================================
// [한글 주석] TextLLMRunner 생성자
// ============================================================================
// 모든 컴포넌트를 unique_ptr로 소유권 이전받아 조립
// 멤버 초기화 순서가 파괴 순서의 역순을 결정하므로 중요:
//   tokenizer_ → metadata_ → module_ → text_decoder_runner_ →
//   text_prefiller_ → io_manager_ → text_token_generator_ → stats_
// module_이 text_decoder_runner_보다 늦게 파괴되어야 하므로 순서가 맞음 ✓
//
// [예시 가정값]
//   metadata = {"get_max_seq_len":2048, "get_max_context_len":2048,
//               "use_kv_cache":1, "enable_dynamic_shape":1, ...}
//   temperature = 0.8 (또는 -1.0이면 GenerationConfig의 값 사용)
//   pos_ = 0 (KV 캐시의 현재 위치, 생성 시작 시 0)
TextLLMRunner::TextLLMRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<::executorch::extension::Module> module,
    std::unique_ptr<TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<TextPrefiller> text_prefiller,
    std::unique_ptr<IOManager> io_manager,
    std::unique_ptr<TextTokenGenerator> text_token_generator,
    std::unique_ptr<Stats> stats,
    float temperature)
    : tokenizer_(std::move(tokenizer)),
      metadata_(std::move(metadata)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      io_manager_(std::move(io_manager)),
      text_token_generator_(std::move(text_token_generator)),
      stats_(std::move(stats)),
      temperature_(temperature),
      pos_(0) {
  // text_prefiller와 text_token_generator는 이미 Module과 TextDecoderRunner에
  // 대한 원시 포인터 참조를 가지고 있다고 가정함
  // (create_text_llm_runner에서 .get()으로 전달됨)
}

// [한글 주석] 모델이 로드되었는지 확인
// text_prefiller와 text_token_generator 모두 내부적으로
// module_->is_method_loaded("forward")를 확인
bool TextLLMRunner::is_loaded() const {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

// ============================================================================
// [한글 주석] 모델 로드
// ============================================================================
// 호출 체인:
//   text_prefiller_->load()
//     → text_decoder_runner_->load()
//       → module_->load_method("forward")
//         → module_->load()  ← .pte 파일에서 Program 로드 (flatbuffer 파싱)
//         → Method 로드 → XNNPACK delegate 초기화, 가중치 매핑, 메모리 할당
//   io_manager_->load() → 기본 CPU IOManager는 no-op
//   text_token_generator_->load() → (이미 로드됨, 스킵)
Error TextLLMRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(text_prefiller_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(io_manager_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator_->load());
  return Error::Ok;
}

// Don't print with the same priority during warmup
#define RUNNER_ET_LOG(warmup, format, ...) \
  if (warmup) {                            \
    ET_LOG(Debug, format, __VA_ARGS__);    \
  } else {                                 \
    ET_LOG(Info, format, __VA_ARGS__);     \
  }

// ============================================================================
// [한글 주석] TextLLMRunner::generate() — 텍스트 생성의 핵심 메서드
// ============================================================================
// 전체 흐름:
//   (1) 모델 로드 (최초 1회)
//   (2) 프롬프트 토큰화: "The answer..." → [450, 1234, 304, 278, 8494, 1139, 338]
//   (3) Prefill: 7개 토큰을 모델에 입력 → KV 캐시 채움 → 첫 예측 토큰 얻음
//   (4) max_new_tokens 결정: min(seq_len=128, max_context_len=2048) - 7 = 121
//   (5) 첫 토큰 디코드 & 출력
//   (6) 자기회귀 생성 루프: 최대 120회 반복 (121 - 1, prefill에서 1개 이미 생성)
//   (7) 통계 리포트 출력
//
// [예시 값]
//   prompt = "The answer to the ultimate question is"
//   config.temperature = 0.8, config.seq_len = 128
//   metadata["get_max_context_len"] = 2048
//   pos_ = 0 (초기 KV 캐시 위치)
Error TextLLMRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {

  // ── [단계 1] 모델 로드 (최초 1회) ──
  // 모델이 아직 로드되지 않았으면 load() 호출
  // load() → module_->load_method("forward") → XNNPACK delegate 초기화
  // 예시: model_load_start_ms=1000, model_load_end_ms=2500 (1.5초 소요)
  if (!is_loaded()) {
    stats_->model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_->model_load_end_ms = time_in_ms();
  }

  if (config.warming) {
    ET_LOG(Info, "Doing a warmup run...");
  }

  RUNNER_ET_LOG(
      config.warming,
      "RSS after loading model: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // ── [단계 2] 토큰 콜백 래핑 ──
  // 생성된 각 토큰을 (a) stdout에 출력하고 (b) 사용자 콜백에 전달
  // 워밍업 모드에서는 stdout 출력을 생략
  // safe_printf(): 출력 불가능한 제어 문자를 필터링
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback, config](const std::string& piece) {
        if (!config.warming) {
          llm::safe_printf(piece.c_str());
          fflush(stdout);  // 즉시 버퍼 플러시 → 스트리밍 출력 효과
        }
        if (token_callback) {
          token_callback(piece);
        }
      };

  // ── [단계 3] 추론 시작 시각 기록 ──
  // inference_start_ms: 토큰화부터 마지막 토큰까지의 시간 측정 시작점
  // 예시: inference_start_ms = 2500
  stats_->inference_start_ms = time_in_ms();
  shouldStop_ = false;

  // ── [단계 4] 남은 KV 캐시 용량 계산 ──
  // max_context_len = 모델의 최대 컨텍스트 길이 - 이미 사용된 위치
  // 예시: 2048 - 0 = 2048 (처음 호출 시 pos_=0)
  // 이전에 prefill()을 호출했다면 pos_ > 0일 수 있음
  int64_t max_context_len = metadata_.at(kMaxContextLen) - pos_;

  uint64_t cur_token = 0;
  int num_prompt_tokens = 0;
  std::vector<uint64_t> prompt_tokens;

  if (!prompt.empty()) {
    // ── [단계 5] 프롬프트 토큰화 ──
    // "The answer to the ultimate question is"
    //   → encode(prompt, bos=0, eos=0)
    //   → [450, 1234, 304, 278, 8494, 1139, 338]
    // num_bos=0이므로 BOS 토큰(1)은 추가되지 않음
    // num_eos=0이므로 EOS 토큰(2)도 추가되지 않음
    ::tokenizers::Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
        prompt, /*bos=*/config.num_bos, /*eos=*/config.num_eos);

    if (!encode_res.ok()) {
      ET_LOG(
          Error,
          "Failed to encode prompt %s. Tokenizers error code %d",
          prompt.c_str(),
          static_cast<uint32_t>(encode_res.error()));
      return Error::InvalidArgument;
    }

    // 토큰화 결과 저장
    // 예시: prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338]
    //       num_prompt_tokens = 7
    prompt_tokens = encode_res.get();
    num_prompt_tokens = prompt_tokens.size();

    // ── [단계 6] 유효성 검증 ──
    // (a) 최소 1개 토큰이 있어야 함
    // (b) 프롬프트 토큰 수가 남은 KV 캐시 용량을 초과하면 안 됨
    //     예시: 7 < 2048 → 통과 ✓
    ET_CHECK_OR_RETURN_ERROR(
        num_prompt_tokens >= 1,
        InvalidArgument,
        "Expected at least 1 prompt token");
    ET_CHECK_OR_RETURN_ERROR(
        num_prompt_tokens < max_context_len,
        InvalidArgument,
        "num_prompt_tokens %d >= max_context_len %" PRId64
        ", Max seq length exceeded - please increase max seq len value in your export script",
        num_prompt_tokens,
        max_context_len);

    // echo=true이면 프롬프트 텍스트를 먼저 출력
    if (config.echo) {
      wrapped_callback(prompt);
    }

    // ── [단계 7] Prefill 실행 ──
    // 모든 프롬프트 토큰을 모델에 입력하여 KV 캐시를 채움
    // text_prefiller_->prefill(prompt_tokens, pos_):
    //   → tokens 텐서 생성: shape [1, 7], data=[450, 1234, 304, 278, 8494, 1139, 338]
    //   → text_decoder_runner_->step(tokens, start_pos=0)
    //     → module_->execute("forward", [{tokens}, {start_pos=0}])
    //     → XNNPACK가 7개 토큰을 한 번에 처리
    //     → KV 캐시의 position 0~6에 key/value 저장
    //     → logits 반환: shape [1, 7, 32000]
    //   → start_pos += 7 → pos_ = 7 (참조로 전달되므로 외부도 업데이트)
    //   → logits_to_token(logits) → 마지막 위치(pos=6)의 logits에서 샘플링
    //     → cur_token = 29871 (예시 — 모델이 예측한 다음 토큰)
    auto prefill_res = text_prefiller_->prefill(prompt_tokens, pos_);
    ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
    cur_token = prefill_res.get();
    // 예시: cur_token = 29871, pos_ = 7
    prefill_next_token_.reset();  // 이전 prefill 결과 소비 완료
  } else {
    // 빈 프롬프트: 이전 prefill() 호출에서 저장된 토큰을 사용
    // 예: 먼저 prefill("시스템 프롬프트")을 호출하고
    //     이후 generate("")로 생성을 시작하는 패턴
    ET_CHECK_OR_RETURN_ERROR(
        prefill_next_token_.has_value(),
        InvalidState,
        "Empty prompt requires a prior prefill() call");
    cur_token = prefill_next_token_.value();
    prefill_next_token_.reset();
  }

  // ── [단계 8] max_new_tokens 결정 ──
  // resolve_max_new_tokens(max_context_len=2048, num_prompt_tokens=7):
  //   seq_len=128, max_new_tokens=-1이므로:
  //   result = min(128, 2048) - 7 = 121
  //
  // 의미: 최대 121개의 새 토큰을 생성할 수 있음
  //       (프롬프트 7개 + 생성 121개 = 총 128개, seq_len 이내)
  int max_new_tokens =
      config.resolve_max_new_tokens(max_context_len, num_prompt_tokens);

  ET_LOG(
      Info,
      "Max new tokens resolved: %d, given pos_ %" PRId64
      ", num_prompt_tokens %d, max_context_len %" PRId64,
      max_new_tokens,
      pos_,
      num_prompt_tokens,
      max_context_len);
  ET_CHECK_OR_RETURN_ERROR(
      max_new_tokens > 0,
      InvalidArgument,
      "Max new tokens %d is less than or equal to 0",
      max_new_tokens);

  // ── [단계 9] 시간 기록 ──
  // first_token_ms: 첫 생성 토큰이 나온 시각 (TTFT 계산용)
  // prompt_eval_end_ms: prefill 완료 시각 (프롬프트 처리량 계산용)
  // 예시: first_token_ms = 2700 (inference_start_ms=2500에서 200ms 후)
  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();

  // ── [단계 10] 첫 번째 토큰 디코드 & 출력 ──
  // prefill에서 얻은 첫 토큰을 텍스트로 변환
  // decode(prev_token=29871, cur_token=29871) → " 42"
  // prev_token이 없으므로 cur_token을 prev로도 사용
  // (토크나이저는 이전 토큰과의 컨텍스트로 공백 처리 등을 결정)
  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  wrapped_callback(std::move(*decode_result));
  // stdout에 " 42" 출력됨

  RUNNER_ET_LOG(
      config.warming,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // ── [단계 11] 토큰 생성 루프 준비 ──
  // prompt_tokens에 첫 생성 토큰을 추가
  // [450, 1234, 304, 278, 8494, 1139, 338] + [29871]
  // → [450, 1234, 304, 278, 8494, 1139, 338, 29871]
  // KV 캐시 사용 시: 생성 루프에서는 마지막 토큰(29871)만 사용
  // KV 캐시 미사용 시: 전체 시퀀스를 매번 입력
  prompt_tokens.push_back(cur_token);

  // ignore_eos 설정: false이면 EOS 토큰(id=2) 만나면 생성 중단
  text_token_generator_->set_ignore_eos(config.ignore_eos);

  // ── [단계 12] 자기회귀 토큰 생성 루프 ──
  // max_new_tokens - 1 = 120개를 더 생성 (prefill에서 이미 1개 생성했으므로)
  // temperature: -1.0이면 config.temperature 사용 (0.8)
  //
  // 내부 루프 (TextTokenGenerator::generate):
  //   while (pos < 7 + 120):
  //     token_data = [cur_token]  → shape [1, 1]
  //     logits = text_decoder_runner_->step(token, pos)
  //       → module_->execute("forward", [{token}, {pos}])
  //       → XNNPACK: position pos에서 token 처리, KV 캐시 업데이트
  //       → logits [1, 1, 32000]
  //     cur_token = sampler.sample(logits, temp=0.8)
  //       → softmax → top-p 샘플링
  //     decode(prev, cur) → 텍스트 → stdout 출력
  //     pos++
  //     EOS 체크: cur_token == 2? → break
  //
  // 반환: 생성된 토큰 수 (예: 120 또는 EOS까지의 수)
  auto generate_result = text_token_generator_->generate(
      prompt_tokens,
      pos_,                                                    // 7
      max_new_tokens - 1,                                      // 120
      temperature_ == -1.0f ? config.temperature : temperature_, // 0.8
      wrapped_callback);
  if (!generate_result.ok()) {
    return generate_result.error();
  }
  int64_t num_generated_tokens = generate_result.get();

  // ── [단계 13] pos_ 업데이트 ──
  // 예시: pos_ = 7 + 120 = 127
  // 다음 generate() 호출 시 이 위치부터 이어서 생성
  pos_ += num_generated_tokens;

  // ── [단계 14] 추론 종료 시각 기록 ──
  // 예시: inference_end_ms = 15000 (추론에 12.5초 소요)
  stats_->inference_end_ms = time_in_ms();
  if (!config.warming) {
    printf("\n");  // 생성 완료 후 줄바꿈
  }
  RUNNER_ET_LOG(
      config.warming,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  if (num_generated_tokens == max_new_tokens) {
    RUNNER_ET_LOG(config.warming, "Max new tokens %i reached!", max_new_tokens);
  }

  // ── [단계 15] 통계 기록 ──
  // 예시:
  //   num_prompt_tokens = 7
  //   num_generated_tokens = 120
  stats_->num_prompt_tokens =
      prompt.empty() ? static_cast<int64_t>(pos_) : num_prompt_tokens;
  stats_->num_generated_tokens = num_generated_tokens;

  // ── [단계 16] 성능 리포트 출력 ──
  // print_report()는 JSON + 사람이 읽기 쉬운 형태로 출력:
  //   - 모델 로드 시간: 1.5초
  //   - 전체 추론 시간: 12.5초, 처리율: 9.6 tok/s
  //   - 프롬프트 평가: 0.2초, 처리율: 35.0 tok/s
  //   - 생성: 12.3초, 처리율: 9.76 tok/s
  //   - 첫 토큰까지 시간(TTFT): 0.2초
  //   - 샘플링 시간: 0.05초
  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    print_report(*stats_);
  }
  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

Result<uint64_t> TextLLMRunner::prefill(
    const std::vector<MultimodalInput>& inputs,
    int32_t num_bos,
    int32_t num_eos) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  for (const auto& input : inputs) {
    if (input.is_text()) {
      auto encode_res = tokenizer_->encode(
          input.get_text(), /*bos=*/num_bos, /*eos=*/num_eos);
      ET_CHECK_TK_OK_OR_RETURN_ERROR(
          encode_res.error(),
          "Failed to encode prompt %s",
          input.get_text().c_str());
      std::vector<uint64_t> tokens = encode_res.get();
      auto prefill_res = text_prefiller_->prefill(tokens, pos_);
      ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
      prefill_next_token_ = prefill_res.get();
      num_bos = 0;
      num_eos = 0;
    }
    // Skip non-text inputs — text-only runner
  }

  if (!prefill_next_token_.has_value()) {
    return Error::InvalidArgument;
  }
  return prefill_next_token_.value();
}

Result<uint64_t> TextLLMRunner::prefill(
    const std::string& prompt,
    int32_t num_bos,
    int32_t num_eos) {
  std::vector<MultimodalInput> inputs;
  inputs.emplace_back(MultimodalInput(prompt));
  return prefill(inputs, num_bos, num_eos);
}

Result<uint64_t> TextLLMRunner::prefill(
    const std::string& prompt,
    const GenerationConfig& config) {
  return prefill(prompt, config.num_bos, config.num_eos);
}

Error TextLLMRunner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  GenerationConfig config;
  config.echo = false;
  config.max_new_tokens = max_new_tokens;
  config.warming = true;

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup, not resetting the std::unique_ptr!
  reset();
  return err;
}

void TextLLMRunner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  } else {
    ET_LOG(Error, "Token generator is not loaded, cannot stop");
  }
}

void TextLLMRunner::reset() {
  stats_->reset();
  pos_ = 0;
  prefill_next_token_.reset();
}

} // namespace executorch::extension::llm
