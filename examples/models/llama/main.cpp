/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/examples/models/llama/runner/runner.h>
#include <gflags/gflags.h>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    data_paths,
    "",
    "Data files for the model. If multiple files are provided, they should be comma separated.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");
DEFINE_string(
    prompt_file,
    "",
    "Optional path to a file containing the prompt. If set, this overrides --prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "DEPRECATED: Please use max_seq_len instead. Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    max_new_tokens,
    -1,
    "Total number of tokens to generate, excluding the prompt, will be capped by max_seq_len - # prompt tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_int32(
    num_bos,
    0,
    "Number of BOS tokens to prepend to the prompt. Defaults to 0. If > 0, the prompt will be prepended with BOS tokens. This is useful for models that expect one or more BOS token at the start.");

DEFINE_int32(
    num_eos,
    0,
    "Number of EOS tokens to append to the prompt. Defaults to 0. If > 0, the prompt will be appended with EOS tokens. This is useful for models that expect one or more EOS token at the end.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

DEFINE_bool(
    ignore_eos,
    false,
    "Whether to ignore EOS token and continue generating until max_new_tokens is reached.");

DEFINE_string(
    etdump_path,
    "etdump.in",
    "If an etdump path is provided, generate an ETDump file at the specified path for profiling purposes.");

DEFINE_string(
    method_name,
    "forward",
    "Method name to execute in the model (e.g., 'forward', 'lora_forward').");

// Helper function to parse comma-separated string lists
std::vector<std::string> parseStringList(const std::string& input) {
  std::vector<std::string> result;
  if (input.empty()) {
    return result;
  }

  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // Trim whitespace
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

bool readFileToString(const std::string& path, std::string& out) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    return false;
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  out = ss.str();
  return true;
}

// ============================================================================
// [한글 주석] LLaMA Text LLM Runner 진입점
// ============================================================================
// 전체 흐름 요약:
//   1. 커맨드라인 인자 파싱 (모델경로, 토크나이저, 프롬프트, temperature 등)
//   2. 스레드풀 설정 (선택)
//   3. TextLLMRunner 생성 (모델+토크나이저+디코더+프리필러+토큰생성기 조립)
//   4. 워밍업 실행 (선택)
//   5. GenerationConfig 구성
//   6. runner->generate() 호출 → 프롬프트 토큰화 → prefill → 토큰 생성 루프
//   7. 프로파일링 데이터 저장 (선택)
//
// [예시 가정값]
//   모델: LLaMA 7B (XNNPACK 백엔드), vocab_size=32000, max_seq_len=2048
//   프롬프트: "The answer to the ultimate question is"
//   → 토큰화 결과: [450, 1234, 304, 278, 8494, 1139, 338] (7개 토큰)
//   temperature=0.8, seq_len=128
// ============================================================================
int32_t main(int32_t argc, char** argv) {
  // [단계 1] 커맨드라인 인자 파싱
  // 예: --model_path=llama2.pte --tokenizer_path=tokenizer.model
  //     --prompt="The answer to the ultimate question is"
  //     --temperature=0.8 --seq_len=128
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // [단계 2] 모델 파일 경로 설정
  // model_path = "llama2.pte" (ExecuTorch flatbuffer 형식으로 직렬화된 모델)
  // 이 파일에는 XNNPACK 백엔드로 위임된 연산 그래프와 가중치가 포함됨
  const char* model_path = FLAGS_model_path.c_str();

  // data_paths: 외부 가중치 파일(.ptd) 경로 목록 (쉼표 구분)
  // 대부분의 경우 빈 벡터 (가중치가 .pte에 포함됨)
  std::vector<std::string> data_paths = parseStringList(FLAGS_data_paths);

  // tokenizer_path = "tokenizer.model" 또는 "tokenizer.bin"
  // SentencePiece(.model), TikToken, HuggingFace JSON 등 다양한 형식 지원
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  // [단계 3] 프롬프트 텍스트 준비
  // prompt_file이 지정되면 파일에서 읽고, 아니면 --prompt 플래그 값 사용
  // 예시 프롬프트: "The answer to the ultimate question is"
  std::string prompt_storage;
  const char* prompt = FLAGS_prompt.c_str();
  if (!FLAGS_prompt_file.empty()) {
    if (!readFileToString(FLAGS_prompt_file, prompt_storage)) {
      ET_LOG(
          Error,
          "Failed to read prompt file at path: %s",
          FLAGS_prompt_file.c_str());
      return 1;
    }
    prompt = prompt_storage.c_str();
  }

  // temperature = 0.8 (높을수록 랜덤, 0이면 greedy argmax)
  float temperature = FLAGS_temperature;

  // seq_len = 128 (프롬프트 + 생성 토큰의 총 길이 상한)
  // max_new_tokens가 지정되면 seq_len 대신 사용됨
  int32_t seq_len = FLAGS_seq_len;

  // cpu_threads = -1 → 자동 감지 (기기의 고성능 코어 수)
  int32_t cpu_threads = FLAGS_cpu_threads;

  // warmup = false → 워밍업 생략
  bool warmup = FLAGS_warmup;

  // [단계 4] 스레드풀 설정 (ET_USE_THREADPOOL이 정의된 경우)
  // ARM big.LITTLE 아키텍처 등에서 고성능 코어만 사용하도록 설정
  // 예: 8코어 중 4개 big 코어 감지 → 스레드풀 크기 4로 설정
  // XNNPACK 백엔드가 이 스레드풀을 사용하여 행렬 연산을 병렬화
#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  // [단계 5] 프로파일링용 ETDump 생성기 (ET_EVENT_TRACER_ENABLED일 때만)
  // 모델 실행 중 각 연산의 타이밍을 기록하여 .etdump 파일로 저장 가능
#ifdef ET_EVENT_TRACER_ENABLED
  auto etdump_gen_ptr = std::make_unique<executorch::etdump::ETDumpGen>();
  executorch::etdump::ETDumpGen* etdump_gen = etdump_gen_ptr.get();
#endif

  // ============================================================================
  // [단계 6] TextLLMRunner 생성 — 핵심 초기화 단계
  // ============================================================================
  // create_llama_runner() 내부에서 수행되는 작업:
  //   (a) load_llama_tokenizer(): 토크나이저 로드
  //       → SentencePiece/TikToken/HuggingFace JSON 순으로 시도
  //       → vocab_size=32000, bos_id=1, eos_id=2
  //   (b) Module 생성: .pte 파일을 mmap으로 메모리 매핑
  //       → LoadMode::MmapUseMlockIgnoreErrors
  //   (c) get_llm_metadata(): 모델 메타데이터 추출
  //       → max_seq_len=2048, use_kv_cache=true, enable_dynamic_shape=true
  //   (d) get_eos_ids(): EOS 토큰 ID 집합 추출 → {2}
  //   (e) IOManager, TextDecoderRunner, TextPrefiller, TextTokenGenerator 생성
  //   (f) TextLLMRunner 조립 (모든 컴포넌트의 소유권 이전)
  //
  // 반환: TextLLMRunner unique_ptr (nullptr이면 생성 실패)
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
      example::create_llama_runner(
          model_path,           // "llama2.pte"
          tokenizer_path,       // "tokenizer.model"
          data_paths,           // {} (보통 빈 벡터)
          temperature,          // 0.8
#ifdef ET_EVENT_TRACER_ENABLED
          std::move(etdump_gen_ptr),  // 프로파일링 트레이서
#else
          nullptr,              // 프로파일링 비활성
#endif
          FLAGS_method_name);   // "forward"

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create llama runner");
    return 1;
  }

  // [단계 7] 워밍업 실행 (선택)
  // 워밍업은 generate()를 한 번 실행하고 결과를 버림
  // 목적: XNNPACK 내부 캐시/메모리 할당을 사전에 수행하여
  //        실제 추론 시 지연 시간 감소
  // 워밍업 후 reset()으로 KV 캐시와 통계를 초기화
  if (warmup) {
    int32_t warmup_max_new_tokens =
        FLAGS_max_new_tokens != -1 ? FLAGS_max_new_tokens : seq_len;
    auto error =
        runner->warmup(prompt, /*max_new_tokens=*/warmup_max_new_tokens);
    if (error != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to warmup llama runner");
      return 1;
    }
  }

  // ============================================================================
  // [단계 8] GenerationConfig 구성
  // ============================================================================
  // GenerationConfig는 텍스트 생성의 모든 파라미터를 담는 구조체
  // temperature=0.8: 샘플링 시 logits를 1/0.8=1.25배로 스케일링
  //   → 확률 분포가 더 뾰족해짐 → 높은 확률 토큰이 더 자주 선택됨
  //   → temperature=0이면 항상 가장 높은 확률의 토큰 선택 (greedy)
  executorch::extension::llm::GenerationConfig config{
      .temperature = temperature};   // 0.8

  config.ignore_eos = FLAGS_ignore_eos;  // false → EOS 토큰 만나면 생성 중단
  config.num_bos = FLAGS_num_bos;        // 0 → BOS 토큰 추가하지 않음
  config.num_eos = FLAGS_num_eos;        // 0 → EOS 토큰 추가하지 않음

  // max_new_tokens vs seq_len 결정
  // max_new_tokens=-1 (미지정)이면 seq_len=128을 사용
  // resolve_max_new_tokens()에서 최종 계산:
  //   min(seq_len=128, max_context_len=2048) - num_prompt_tokens(7) = 121
  if (FLAGS_max_new_tokens != -1) {
    config.max_new_tokens = FLAGS_max_new_tokens;
  } else {
    ET_LOG(
        Info,
        "max_new_tokens not provided, falling back to seq_len=%d. "
        "Consider using --max_new_tokens instead of --seq_len for specifying generation length.",
        seq_len);
    config.seq_len = seq_len;  // 128
  }

  // ============================================================================
  // [단계 9] 텍스트 생성 실행 — generate()
  // ============================================================================
  // generate() 내부 흐름:
  //   (1) load() → 모델 로드 (최초 1회)
  //       → Module::load_method("forward") → XNNPACK delegate 초기화
  //   (2) tokenizer_->encode(prompt) → [450, 1234, 304, 278, 8494, 1139, 338]
  //   (3) text_prefiller_->prefill(tokens, pos_=0)
  //       → 7개 토큰을 shape [1,7]로 묶어 모델에 한 번에 입력 (parallel prefill)
  //       → KV 캐시에 position 0~6의 key/value 저장
  //       → logits [1,7,32000]에서 마지막 위치 샘플링 → 첫 토큰 (예: 29871)
  //       → pos_ = 7로 업데이트
  //   (4) 첫 토큰 디코드: tokenizer_->decode(29871) → " 42"
  //       → stdout에 " 42" 출력
  //   (5) text_token_generator_->generate(tokens, pos_=7, max=120, temp=0.8)
  //       → 자기회귀 루프: step() → sample → decode → callback 반복
  //       → 매 반복: shape [1,1] 토큰 + 현재 pos → 모델 실행 → logits → 샘플링
  //       → EOS(id=2) 만나거나 max_new_tokens 도달 시 종료
  //   (6) print_report() → JSON 형태 성능 통계 출력
  //       → 프롬프트 처리량, 생성 처리량(tok/s), TTFT 등
  auto error = runner->generate(prompt, config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to run llama runner");
    return 1;
  }

  // [단계 10] 프로파일링 데이터 저장 (ET_EVENT_TRACER_ENABLED일 때만)
  // generate() 실행 중 수집된 각 연산의 타이밍을 .etdump 파일로 저장
  // 이 파일은 ExecuTorch Inspector 도구로 분석 가능
#ifdef ET_EVENT_TRACER_ENABLED
  if (etdump_gen != nullptr) {
    executorch::etdump::ETDumpResult result = etdump_gen->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      FILE* f = fopen(FLAGS_etdump_path.c_str(), "w+");
      if (f == nullptr) {
        ET_LOG(
            Error,
            "Failed to open etdump file at path: %s",
            FLAGS_etdump_path.c_str());
      } else {
        fwrite((uint8_t*)result.buf, 1, result.size, f);
        fclose(f);
        ET_LOG(Info, "ETDump file written to: %s", FLAGS_etdump_path.c_str());
      }
      free(result.buf);
    }
  }
#endif

  return 0;
}
