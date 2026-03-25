/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
// Implementation of helper utilities for creating and configuring LLM runners

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens,
    std::optional<std::string> pattern,
    size_t bos_token_index,
    size_t eos_token_index) {
  runtime::runtime_init();
  auto tekken_tokenizer = std::make_unique<tokenizers::Tekken>();
  // Prevent the case where tekken tokenizer accidentally successfully loads a
  // HuggingFace tokenizer, which is also .json.
  static constexpr std::string_view tekken_name = "tekken.json";
  if (tokenizer_path.size() >= tekken_name.size() &&
      tokenizer_path.rfind(tekken_name) ==
          tokenizer_path.size() - tekken_name.size()) {
    if (tekken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
      ET_LOG(Info, "Loaded tekken tokenizer");
      return tekken_tokenizer;
    }
  }
  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }
  std::unique_ptr<::tokenizers::Tiktoken> tiktoken_tokenizer;
  if (special_tokens != nullptr && !pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        std::move(special_tokens), bos_token_index, eos_token_index);
  } else if (special_tokens != nullptr && pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        pattern.value(),
        std::move(special_tokens),
        bos_token_index,
        eos_token_index);
  } else {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>();
  }
  if (tiktoken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded TikToken tokenizer");
    return tiktoken_tokenizer;
  }

  auto sp_tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  if (sp_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded Sentencepiece tokenizer");
    return sp_tokenizer;
  }

  auto bpe_tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  if (bpe_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded BPE tokenizer");
    return bpe_tokenizer;
  }

  return nullptr;
}

::executorch::runtime::Result<std::unordered_map<std::string, int64_t>>
get_llm_metadata(tokenizers::Tokenizer* tokenizer, Module* module) {
  // Initialize metadata with default values
  std::unordered_map<std::string, int64_t> metadata({
      {llm::kEnableDynamicShape, false},
      {llm::kMaxSeqLen, 128},
      {llm::kMaxContextLen, 128},
      {llm::kUseKVCache, true},
      {llm::kUseSDPAWithKVCache, false},
  });

  // Read metadata from the model
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return ::executorch::runtime::Error::InvalidArgument;
  }
  const auto& method_names = method_names_result.get();

  // Error out if the max seq len metadata method is not present, since
  // it is hard to figure out from just the .pte itself.
  if (!method_names.count(llm::kMaxSeqLen)) {
    ET_LOG(
        Error,
        "Required metadata method %s not found in model",
        llm::kMaxSeqLen);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  for (auto& pair : metadata) {
    const auto& method_name = pair.first;
    auto& value = pair.second;

    if (method_names.count(method_name)) {
      auto get_result = module->get(method_name);
      value = get_result.get().toScalar().to<decltype(metadata)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Method %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }

  // If kMaxContextLen method not found but kMaxSeqLen is
  // available, set kMaxContextLen to the value of kMaxSeqLen.
  if (!method_names.count(llm::kMaxContextLen) &&
      method_names.count(llm::kMaxSeqLen)) {
    metadata[llm::kMaxContextLen] = metadata[llm::kMaxSeqLen];
    ET_LOG(
        Info,
        "Setting kMaxContextLen to kMaxSeqLen value: %" PRId64,
        metadata[llm::kMaxContextLen]);
  }

  // Set tokenizer-related metadata
  metadata[llm::kBosId] = tokenizer->bos_tok();
  metadata[llm::kVocabSize] = tokenizer->vocab_size();
  return metadata;
}

std::unordered_set<uint64_t> get_eos_ids(
    tokenizers::Tokenizer* tokenizer,
    Module* module) {
  std::unordered_set<uint64_t> eos_ids = {tokenizer->eos_tok()};
  // Get EOS IDs if available
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return eos_ids;
  }
  const auto& method_names = method_names_result.get();

  if (method_names.count(llm::kEosIds)) {
    eos_ids.clear();
    auto execute_result = module->execute(llm::kEosIds);
    if (execute_result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute %s", llm::kEosIds);
      return eos_ids;
    }
    for (const auto& eos_id : execute_result.get()) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids.emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }
  return eos_ids;
}

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    float temperature,
    const std::string& method_name,
    Module::LoadMode load_mode) {
  if (data_path.has_value()) {
    std::vector<std::string> data_files;
    data_files.push_back(data_path.value());
    return create_text_llm_runner(
        model_path,
        std::move(tokenizer),
        std::move(data_files),
        temperature,
        nullptr,
        method_name,
        load_mode);
  }
  return create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      std::vector<std::string>(),
      temperature,
      nullptr,
      method_name,
      load_mode);
}

// ============================================================================
// [한글 주석] create_text_llm_runner() — TextLLMRunner 팩토리 함수
// ============================================================================
// 모든 컴포넌트를 생성하고 의존성을 주입하여 TextLLMRunner를 조립
//
// [생성되는 컴포넌트와 소유권]
//   Module(unique_ptr) ─── TextLLMRunner가 소유
//     ↓ (raw ptr)
//   IOManager(unique_ptr) ─── TextLLMRunner가 소유
//     ↓ (raw ptr)
//   TextDecoderRunner(unique_ptr) ─── TextLLMRunner가 소유
//     ↓ (raw ptr)            ↓ (raw ptr)
//   TextPrefiller(unique_ptr)  TextTokenGenerator(unique_ptr)
//     └── TextLLMRunner 소유    └── TextLLMRunner 소유
//
// [예시 가정값]
//   model_path = "llama2.pte"
//   tokenizer: SentencePiece, vocab_size=32000
//   load_mode = MmapUseMlockIgnoreErrors
//   method_name = "forward"
//   temperature = 0.8
std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::vector<std::string> data_files,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer,
    const std::string& method_name,
    Module::LoadMode load_mode) {

  // [1] 토크나이저 유효성 검사
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // [2] Module 생성 — .pte 모델 파일 로딩 준비
  // Module은 ExecuTorch의 Program(모델)을 감싸는 래퍼 클래스
  // MmapUseMlockIgnoreErrors: mmap으로 파일 매핑 + mlock 시도 (실패 시 무시)
  // data_files: 외부 가중치 파일(.ptd) — 보통 빈 벡터
  std::unique_ptr<Module> module;
  if (data_files.size() > 0) {
    module = std::make_unique<Module>(
        model_path, data_files, load_mode, std::move(event_tracer));
  } else {
    module = std::make_unique<Module>(
        model_path, load_mode, std::move(event_tracer));
  }

  // [3] 모델 메타데이터 추출
  // .pte 파일에 직렬화된 메타데이터 메서드를 실행하여 설정값 읽기
  // 예시 결과:
  //   get_max_seq_len = 2048
  //   get_max_context_len = 2048
  //   use_kv_cache = 1 (true)
  //   enable_dynamic_shape = 1 (true → parallel prefill 가능)
  //   use_sdpa_with_kv_cache = 0 (false)
  //   get_bos_id = 1, get_vocab_size = 32000
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  // [4] EOS 토큰 ID 집합 추출
  // 모델의 get_eos_ids 메서드 실행 → {2}
  // 없으면 토크나이저의 기본 eos_tok 사용
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      llm::get_eos_ids(tokenizer.get(), module.get()));

  // [5] IOManager 생성
  // 기본 CPU IOManager: prepare_decode/prefill에서 입력을 그대로 전달
  // 커스텀 백엔드(GPU 등)는 파생 클래스로 구현
  std::unique_ptr<IOManager> io_manager = std::make_unique<IOManager>(*module);

  // [6] TextDecoderRunner 생성
  // Module과 IOManager의 원시 포인터를 참조 (소유하지 않음)
  // 역할: step() 메서드로 모델 forward pass 1회 실행
  ET_LOG(Info, "Using method: %s", method_name.c_str());
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(),       // Module* — module이 먼저 파괴되면 안 됨
      io_manager.get(),   // IOManager*
      method_name);       // "forward"

  // [7] TextPrefiller 생성
  // 역할: 프롬프트 토큰들을 모델에 입력하여 KV 캐시를 채움
  // use_kv_cache=true: KV 캐시 사용
  // enable_parallel_prefill=true (=enable_dynamic_shape): 토큰을 한 번에 처리
  // max_seq_len=2048: 한 번에 처리할 수 있는 최대 토큰 수
  auto text_prefiller = std::make_unique<TextPrefiller>(
      text_decoder_runner.get(),           // TextDecoderRunner* (비소유)
      metadata.at(kUseKVCache),            // true
      metadata.at(kEnableDynamicShape),    // true (parallel prefill)
      metadata.at(kMaxSeqLen));            // 2048

  // [8] Stats & TextTokenGenerator 생성
  // Stats: 성능 측정 (모델 로드 시간, prefill 시간, 생성 시간 등)
  // TextTokenGenerator: 자기회귀 토큰 생성 루프
  //   - tokenizer: 토큰→텍스트 변환용
  //   - text_decoder_runner: 매 반복 모델 실행용
  //   - eos_ids: {2} — 이 토큰이 생성되면 루프 종료
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),              // Tokenizer* (비소유)
      text_decoder_runner.get(),    // TextDecoderRunner* (비소유)
      metadata.at(kUseKVCache),     // true
      std::move(eos_ids),           // {2} — 소유권 이전
      stats.get());                 // Stats* (비소유)

  // [9] TextLLMRunner 조립 — 모든 unique_ptr 소유권 이전
  // TextLLMRunner가 모든 컴포넌트의 생명주기를 관리
  // 멤버 선언 순서에 따라 파괴 순서가 결정됨:
  //   stats_ → text_token_generator_ → io_manager_ →
  //   text_prefiller_ → text_decoder_runner_ → module_ →
  //   metadata_ → tokenizer_
  return std::make_unique<TextLLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats),
      temperature);
}

std::unique_ptr<MultimodalRunner> create_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    Module::LoadMode load_mode) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_path.has_value()) {
    module = std::make_unique<Module>(model_path, data_path.value(), load_mode);
  } else {
    module = std::make_unique<Module>(model_path, load_mode);
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      get_eos_ids(tokenizer.get(), module.get()));

  // Create IOManager
  std::unique_ptr<IOManager> io_manager = std::make_unique<IOManager>(*module);

  // Create text_decoder_runner
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());

  // Create multimodal_prefiller
  auto multimodal_prefiller = std::make_unique<MultimodalPrefiller>(
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      io_manager.get());

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the MultimodalRunner instance
  return std::make_unique<MultimodalRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(multimodal_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));
}

} // namespace executorch::extension::llm
