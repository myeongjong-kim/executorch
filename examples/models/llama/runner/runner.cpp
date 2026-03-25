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

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/extension/module/module.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>

namespace example {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace llm = ::executorch::extension::llm;

// ============================================================================
// [한글 주석] LLaMA 토크나이저 로드
// ============================================================================
// LLaMA 모델 버전에 맞는 특수 토큰(special tokens)을 설정하고 토크나이저 로드
//
// Version::Default: LLaMA 2/3의 기본 특수 토큰 사용
//   - <s> (BOS), </s> (EOS), <unk> 등
//
// load_tokenizer()가 다음 순서로 포맷을 시도:
//   Tekken → HuggingFace JSON → TikToken → SentencePiece → BPE(Llama2c)
//
// [예시]
//   tokenizer_path = "tokenizer.model" → SentencePiece로 로드 성공
//   vocab_size = 32000, bos_tok = 1, eos_tok = 2
std::unique_ptr<::tokenizers::Tokenizer> load_llama_tokenizer(
    const std::string& tokenizer_path,
    Version version) {
  // get_special_tokens(version): LLaMA 버전별 특수 토큰 목록 반환
  // 예: ["<|begin_of_text|>", "<|end_of_text|>", ...] (LLaMA 3)
  //     또는 nullptr (LLaMA 2 SentencePiece는 특수 토큰이 내장됨)
  auto special_tokens = get_special_tokens(version);
  return llm::load_tokenizer(tokenizer_path, std::move(special_tokens));
}

// ── 단일 data_path를 받는 오버로드 (편의용) ──
std::unique_ptr<llm::TextLLMRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer,
    const std::string& method_name) {
  if (data_path.has_value()) {
    std::vector<std::string> data_files;
    data_files.push_back(data_path.value());
    return create_llama_runner(
        model_path,
        tokenizer_path,
        std::move(data_files),
        temperature,
        std::move(event_tracer),
        method_name);
  }
  return create_llama_runner(
      model_path,
      tokenizer_path,
      std::vector<std::string>(),
      temperature,
      std::move(event_tracer),
      method_name);
}

// ============================================================================
// [한글 주석] create_llama_runner() — LLaMA Runner 생성 메인 함수
// ============================================================================
// main.cpp에서 호출되는 진입점
// 역할: LLaMA 전용 토크나이저를 로드한 후, 범용 create_text_llm_runner()에 위임
//
// [호출 체인]
//   create_llama_runner()
//     ├── load_llama_tokenizer() → SentencePiece/TikToken 토크나이저
//     └── llm::create_text_llm_runner()
//           ├── Module 생성 (mmap)
//           ├── 메타데이터 추출 (max_seq_len, use_kv_cache 등)
//           ├── EOS ID 추출
//           ├── IOManager, TextDecoderRunner, TextPrefiller, TextTokenGenerator 생성
//           └── TextLLMRunner 조립 & 반환
//
// [예시]
//   model_path = "llama2.pte"
//   tokenizer_path = "tokenizer.model"
//   data_files = {} (빈 벡터)
//   temperature = 0.8
//   method_name = "forward"
std::unique_ptr<llm::TextLLMRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::vector<std::string> data_files,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer,
    const std::string& method_name) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());

  // [1] LLaMA 토크나이저 로드
  // Version::Default로 LLaMA 2/3 기본 특수 토큰 설정
  // 예시: SentencePiece 토크나이저 로드 → vocab_size=32000
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      load_llama_tokenizer(tokenizer_path, Version::Default);

  if (tokenizer == nullptr) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken, Sentencepiece or Llama2.c tokenizer, make sure the artifact is one of these types",
        tokenizer_path.c_str());
    return nullptr;
  }

  // [2] 범용 TextLLMRunner 팩토리에 위임
  // 여기서 Module, 메타데이터, 모든 컴포넌트가 생성되고 조립됨
  // 반환: 완전히 초기화된 TextLLMRunner (아직 모델은 로드되지 않음)
  //       → generate() 최초 호출 시 load()가 트리거됨
  return llm::create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      data_files,
      temperature,
      std::move(event_tracer),
      method_name);
}

} // namespace example
