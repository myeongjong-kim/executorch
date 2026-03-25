# ExecuTorch TextLLMRunner 코드 흐름 상세 분석서

> **가정 (예시 값)**
> - 모델: LLaMA 7B (XNNPACK 백엔드로 export)
> - 프롬프트: `"The answer to the ultimate question is"` → 토큰화 결과 8개 토큰
> - 토큰 ID: `[1, 450, 1234, 304, 278, 8494, 1139, 338]` (BOS=1 포함)
> - vocab_size: 32000
> - max_seq_len (=max_context_len): 2048
> - temperature: 0.8
> - max_new_tokens: 128 (seq_len=128)
> - use_kv_cache: true
> - enable_dynamic_shape (=enable_parallel_prefill): true

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [main() 진입점](#2-main-진입점)
3. [Runner 생성 (create_llama_runner)](#3-runner-생성-create_llama_runner)
4. [모델 로드 (TextLLMRunner::load)](#4-모델-로드-textllmrunnerload)
5. [텍스트 생성 (TextLLMRunner::generate)](#5-텍스트-생성-textllmrunnergenerate)
6. [프롬프트 토큰화](#6-프롬프트-토큰화)
7. [Prefill 단계 (TextPrefiller::prefill)](#7-prefill-단계-textprefillerprefill)
8. [TextDecoderRunner::step — 모델 Forward Pass](#8-textdecoderrunnerstepcall--모델-forward-pass)
9. [IOManager — 입출력 준비](#9-iomanager--입출력-준비)
10. [Module::execute — ExecuTorch 런타임 실행](#10-moduleexecute--executorch-런타임-실행)
11. [Logits → Token 변환 (Sampler)](#11-logits--token-변환-sampler)
12. [첫 번째 토큰 출력](#12-첫-번째-토큰-출력)
13. [토큰 생성 루프 (TextTokenGenerator::generate)](#13-토큰-생성-루프-texttokengeneratorgenerate)
14. [통계 리포트 출력](#14-통계-리포트-출력)
15. [객체 소유권 및 생명주기](#15-객체-소유권-및-생명주기)
16. [전체 호출 스택 요약 (Call Graph)](#16-전체-호출-스택-요약-call-graph)

---

## 1. 전체 아키텍처 개요

```
main.cpp
  └── create_llama_runner()                    ← Runner 팩토리
        ├── load_llama_tokenizer()             ← 토크나이저 로드
        └── create_text_llm_runner()           ← 핵심 팩토리
              ├── Module 생성 (mmap 로딩)
              ├── get_llm_metadata() 메타데이터 추출
              ├── get_eos_ids() EOS 토큰 추출
              ├── IOManager 생성
              ├── TextDecoderRunner 생성
              ├── TextPrefiller 생성
              ├── TextTokenGenerator 생성
              └── TextLLMRunner 생성

TextLLMRunner::generate(prompt, config)
  ├── load()  ← 모델 로드 (최초 1회)
  ├── tokenizer_->encode(prompt)              ← 프롬프트 토큰화
  ├── text_prefiller_->prefill(tokens, pos_)  ← Prefill 단계
  │     └── text_decoder_runner_->step()      ← 모델 forward (전체 프롬프트 1회)
  │           ├── io_manager_->prepare_decode()
  │           ├── module_->execute("forward")  ← XNNPACK delegate 실행
  │           └── io_manager_->update_decode()
  ├── logits_to_token(logits)                 ← 첫 토큰 샘플링
  ├── tokenizer_->decode()                    ← 토큰→텍스트
  └── text_token_generator_->generate()       ← 자기회귀 루프
        └── [반복: step() → sample → decode → callback]
```

### 클래스 관계도

```
IRunner (인터페이스)
  └── TextLLMRunner
        ├── owns: Module (ExecuTorch 프로그램 래퍼)
        ├── owns: Tokenizer (텍스트↔토큰 변환)
        ├── owns: TextDecoderRunner (forward pass 실행)
        │     ├── uses: Module* (비소유 포인터)
        │     └── uses: IOManager* (비소유 포인터)
        ├── owns: TextPrefiller (프롬프트 prefill)
        │     └── uses: TextDecoderRunner* (비소유 포인터)
        ├── owns: TextTokenGenerator (자기회귀 생성 루프)
        │     ├── uses: TextDecoderRunner* (비소유 포인터)
        │     └── uses: Tokenizer* (비소유 포인터)
        ├── owns: IOManager (입출력 관리)
        └── owns: Stats (성능 통계)
```

---

## 2. main() 진입점

**파일**: `examples/models/llama/main.cpp`

### 2.1 커맨드라인 파싱

```
실행 예시:
  llama_main --model_path=llama2.pte --tokenizer_path=tokenizer.model \
             --prompt="The answer to the ultimate question is" \
             --temperature=0.8 --seq_len=128
```

gflags를 통해 다음 인자를 파싱합니다:

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `model_path` | `"llama2.pte"` | ExecuTorch 직렬화 모델 파일 |
| `tokenizer_path` | `"tokenizer.bin"` | 토크나이저 파일 |
| `prompt` | `"The answer to the..."` | 입력 프롬프트 |
| `temperature` | `0.8` | 샘플링 온도 (0=greedy) |
| `seq_len` | `128` | 총 시퀀스 길이 (프롬프트+생성) |
| `max_new_tokens` | `-1` | 생성할 최대 토큰 수 |
| `cpu_threads` | `-1` | CPU 스레드 수 (자동) |
| `num_bos` | `0` | 프롬프트 앞에 추가할 BOS 토큰 수 |
| `num_eos` | `0` | 프롬프트 뒤에 추가할 EOS 토큰 수 |
| `warmup` | `false` | 워밍업 실행 여부 |
| `ignore_eos` | `false` | EOS 토큰 무시 여부 |
| `method_name` | `"forward"` | 모델의 실행 메서드 이름 |

### 2.2 스레드풀 설정

```cpp
// ET_USE_THREADPOOL이 정의된 경우
// cpu_threads=-1이면 기기의 고성능 코어 수를 자동 감지
uint32_t num_performant_cores = cpuinfo::get_num_performant_cores();
// 예: ARM big.LITTLE에서 4개의 big 코어 감지 → 스레드 4개로 설정
threadpool::get_threadpool()->_unsafe_reset_threadpool(num_performant_cores);
```

### 2.3 Runner 생성

```cpp
std::unique_ptr<TextLLMRunner> runner = example::create_llama_runner(
    model_path,           // "llama2.pte"
    tokenizer_path,       // "tokenizer.model"
    data_paths,           // {} (빈 벡터)
    temperature,          // 0.8
    nullptr,              // event_tracer (프로파일링 비활성)
    FLAGS_method_name);   // "forward"
```

### 2.4 Generation Config 구성 및 실행

```cpp
GenerationConfig config{.temperature = 0.8};
config.ignore_eos = false;
config.num_bos = 0;     // BOS 토큰 자동 추가하지 않음
config.num_eos = 0;     // EOS 토큰 자동 추가하지 않음
config.seq_len = 128;   // max_new_tokens=-1이므로 seq_len 사용

runner->generate(prompt, config);
// 내부적으로: prefill → 첫 토큰 → 생성 루프 → 통계 출력
```

---

## 3. Runner 생성 (create_llama_runner)

**파일**: `examples/models/llama/runner/runner.cpp`

### 3.1 토크나이저 로드

```
create_llama_runner()
  └── load_llama_tokenizer(tokenizer_path, Version::Default)
        └── llm::load_tokenizer(path, special_tokens)
```

`load_tokenizer()`는 다음 순서로 토크나이저 포맷을 시도합니다:

1. **Tekken** (`.json`으로 끝나는 파일명이 `tekken.json`인 경우만)
2. **HuggingFace JSON** (`tokenizer.json` 포맷)
3. **TikToken** (LLaMA 3 등에서 사용하는 바이트페어 인코딩)
4. **SentencePiece** (LLaMA 1/2에서 사용, `.model` 확장자)
5. **BPE (Llama2c)** (`.bin` 확장자, Karpathy의 llama2.c 포맷)

```
예시: tokenizer.model → SentencePiece 로드 성공
      vocab_size = 32000
      bos_token = 1 (<s>)
      eos_token = 2 (</s>)
```

### 3.2 핵심 팩토리: create_text_llm_runner()

**파일**: `extension/llm/runner/llm_runner_helper.cpp`

#### 3.2.1 Module 생성

```cpp
auto module = std::make_unique<Module>(
    model_path,                            // "llama2.pte"
    load_mode: MmapUseMlockIgnoreErrors,   // mmap + mlock (실패 시 무시)
    event_tracer: nullptr                  // 프로파일링 비활성
);
```

**LoadMode::MmapUseMlockIgnoreErrors**:
- `mmap()`으로 파일을 메모리에 매핑 (전체 모델을 RAM에 로드하지 않음)
- `mlock()`으로 페이지를 RAM에 고정 시도 (낮은 추론 지연)
- mlock 실패 시 graceful fallback (mmap만 사용)

#### 3.2.2 메타데이터 추출 (get_llm_metadata)

모델 `.pte` 파일에 직렬화된 메타데이터 메서드를 실행하여 설정값을 읽습니다:

```
module->method_names()로 사용 가능한 메서드 목록 조회:
  → {"forward", "get_max_seq_len", "get_max_context_len",
     "use_kv_cache", "enable_dynamic_shape", "get_eos_ids", ...}

각 메타데이터 메서드 실행:
  module->get("get_max_seq_len")     → 2048
  module->get("get_max_context_len") → 2048
  module->get("use_kv_cache")        → 1 (true)
  module->get("enable_dynamic_shape")→ 1 (true)
  module->get("use_sdpa_with_kv_cache") → 0 (false)

토크나이저에서:
  bos_id   = 1
  vocab_size = 32000

최종 metadata 맵:
  {
    "get_max_seq_len": 2048,
    "get_max_context_len": 2048,
    "use_kv_cache": 1,
    "enable_dynamic_shape": 1,
    "use_sdpa_with_kv_cache": 0,
    "get_bos_id": 1,
    "get_vocab_size": 32000
  }
```

#### 3.2.3 EOS ID 추출 (get_eos_ids)

```
module->execute("get_eos_ids") → [2]  (EOS 토큰 ID = 2)
eos_ids = {2}
```

#### 3.2.4 컴포넌트 조립

```cpp
// 1. IOManager: 입출력 텐서 준비 (CPU 기본 구현)
auto io_manager = make_unique<IOManager>(*module);

// 2. TextDecoderRunner: 모델 forward pass 실행기
//    module과 io_manager의 원시 포인터를 참조 (비소유)
auto text_decoder_runner = make_unique<TextDecoderRunner>(
    module.get(),        // Module* (비소유)
    io_manager.get(),    // IOManager* (비소유)
    "forward"            // 실행할 메서드 이름
);

// 3. TextPrefiller: 프롬프트 prefill 처리
auto text_prefiller = make_unique<TextPrefiller>(
    text_decoder_runner.get(),  // TextDecoderRunner* (비소유)
    use_kv_cache: true,         // KV 캐시 사용
    enable_parallel_prefill: true, // 병렬 prefill (동적 shape)
    max_seq_len: 2048           // 최대 시퀀스 길이
);

// 4. Stats: 성능 통계 추적
auto stats = make_unique<Stats>();

// 5. TextTokenGenerator: 자기회귀 토큰 생성 루프
auto text_token_generator = make_unique<TextTokenGenerator>(
    tokenizer.get(),             // Tokenizer* (비소유)
    text_decoder_runner.get(),   // TextDecoderRunner* (비소유)
    use_kv_cache: true,
    eos_ids: {2},                // EOS 토큰 ID 집합
    stats.get()                  // Stats* (비소유)
);

// 6. 최종 TextLLMRunner 조립 — 모든 unique_ptr 소유권 이전
return make_unique<TextLLMRunner>(
    metadata, tokenizer, module,
    text_decoder_runner, text_prefiller,
    io_manager, text_token_generator,
    stats, temperature
);
```

---

## 4. 모델 로드 (TextLLMRunner::load)

**파일**: `extension/llm/runner/text_llm_runner.cpp:57`

```
TextLLMRunner::load()
  ├── text_prefiller_->load()
  │     └── text_decoder_runner_->load()
  │           └── module_->load_method("forward")
  │                 ├── module_->load()  ← Program 로드 (최초 1회)
  │                 │     └── DataLoader에서 .pte 파일 읽기
  │                 │         → Program::load() → flatbuffer 파싱
  │                 └── Method 로드
  │                       ├── MethodMeta에서 메모리 계획 크기 조회
  │                       ├── HierarchicalAllocator 생성
  │                       ├── MemoryManager 생성
  │                       └── Program::load_method() 호출
  │                             → XNNPACK delegate 초기화
  │                             → 가중치 로드 & 연산 그래프 준비
  ├── io_manager_->load()   ← 기본 IOManager는 no-op
  └── text_token_generator_->load()
        └── text_decoder_runner_->load()  ← 이미 로드됨, 스킵
```

`module_->load_method("forward")`가 핵심입니다. 이 호출이:
1. `.pte` 파일에서 `forward` 메서드의 실행 계획을 로드
2. 필요한 메모리를 할당 (활성화 텐서, KV 캐시 등)
3. XNNPACK 백엔드 delegate를 초기화
4. 가중치를 메모리에 매핑

---

## 5. 텍스트 생성 (TextLLMRunner::generate)

**파일**: `extension/llm/runner/text_llm_runner.cpp:75`

### 전체 흐름 (예시 값 기준)

```
generate("The answer to the ultimate question is", config)
│
├── [1] 모델 로드 확인
│   stats_->model_load_start_ms = 1000  (현재 시각 ms)
│   load() → 모델 로드
│   stats_->model_load_end_ms = 2500    (1.5초 로드)
│
├── [2] 추론 시작 시간 기록
│   stats_->inference_start_ms = 2500
│   shouldStop_ = false
│
├── [3] 남은 KV 캐시 용량 계산
│   max_context_len = metadata["get_max_context_len"] - pos_
│                   = 2048 - 0 = 2048
│
├── [4] 프롬프트 토큰화
│   tokenizer_->encode("The answer to the ultimate question is", bos=0, eos=0)
│   → prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338]
│   num_prompt_tokens = 7
│
├── [5] 유효성 검증
│   num_prompt_tokens(7) >= 1  ✓
│   num_prompt_tokens(7) < max_context_len(2048)  ✓
│
├── [6] Prefill 실행
│   text_prefiller_->prefill(prompt_tokens, pos_=0)
│   → cur_token = 29871  (모델이 예측한 다음 토큰)
│   → pos_ = 7 (prefill 후 위치 업데이트)
│
├── [7] max_new_tokens 결정
│   config.resolve_max_new_tokens(max_context_len=2048, num_prompt_tokens=7)
│   → seq_len=128, max_new_tokens=-1이므로:
│     result = min(128, 2048) - 7 = 121
│   max_new_tokens = 121
│
├── [8] 시간 기록
│   stats_->first_token_ms = 2700
│   stats_->prompt_eval_end_ms = 2700
│
├── [9] 첫 토큰 디코드 & 출력
│   tokenizer_->decode(29871, 29871) → " 42"
│   wrapped_callback(" 42") → stdout에 " 42" 출력
│
├── [10] 토큰 생성 루프 실행
│   prompt_tokens.push_back(29871) → [450, 1234, 304, 278, 8494, 1139, 338, 29871]
│   text_token_generator_->generate(
│       tokens=prompt_tokens,
│       start_pos=7,          // pos_ (prefill 후)
│       max_new_tokens=120,   // 121-1 (첫 토큰은 이미 생성)
│       temperature=0.8,
│       wrapped_callback
│   )
│   → 120개 토큰 생성 (또는 EOS 도달 시 조기 종료)
│
├── [11] pos_ 업데이트
│   pos_ += num_generated_tokens  (예: pos_ = 7 + 120 = 127)
│
├── [12] 통계 기록 & 출력
│   stats_->inference_end_ms = 15000
│   stats_->num_prompt_tokens = 7
│   stats_->num_generated_tokens = 120
│   print_report(stats_) → JSON + 처리량 정보 출력
```

---

## 6. 프롬프트 토큰화

**사용 라이브러리**: `pytorch/tokenizers` (SentencePiece 또는 TikToken)

```
tokenizer_->encode("The answer to the ultimate question is", bos=0, eos=0)

처리 과정:
1. 텍스트를 유니코드 정규화
2. BPE(Byte Pair Encoding) 또는 SentencePiece 알고리즘으로 서브워드 분할
3. 각 서브워드를 vocab ID로 매핑

예시 결과 (실제 값은 토크나이저에 따라 다름):
  "The"       → 450
  " answer"   → 1234
  " to"       → 304
  " the"      → 278
  " ultimate" → 8494
  " question" → 1139
  " is"       → 338

prompt_tokens = [450, 1234, 304, 278, 8494, 1139, 338]
num_prompt_tokens = 7
```

**참고**: `num_bos=0`이므로 BOS 토큰(1)이 앞에 추가되지 않습니다. `num_bos=1`이면 `[1, 450, 1234, ...]`이 됩니다.

---

## 7. Prefill 단계 (TextPrefiller::prefill)

**파일**: `extension/llm/runner/text_prefiller.cpp:29`

### 7.1 Chunking 판단

```
num_prompt_tokens = 7
max_seq_len_ = 2048

7 <= 2048 이므로 chunking 없이 단일 prefill_chunk() 호출
```

만약 프롬프트가 max_seq_len(2048)보다 길면, max_seq_len 크기의 청크로 분할하여 순차적으로 prefill합니다.

### 7.2 prefill_chunk() — Parallel Prefill

**파일**: `extension/llm/runner/text_prefiller.cpp:72`

`enable_parallel_prefill_=true`이므로 모든 프롬프트 토큰을 한 번에 처리합니다:

```
enable_parallel_prefill_ = true
use_kv_cache_ = true

[1] 토큰 텐서 생성
    tokens = from_blob(prompt_tokens.data(), shape={1, 7}, dtype=Long)
    // 텐서 데이터: [[450, 1234, 304, 278, 8494, 1139, 338]]
    // shape: [batch=1, seq_len=7]

[2] 모델 forward pass 실행
    outputs_res = text_decoder_runner_->step(tokens, start_pos=0)
    // → 모델이 7개 토큰을 한 번에 처리
    // → KV 캐시에 7개 위치의 key/value 저장
    // → logits 텐서 반환: shape [1, 7, 32000]

[3] start_pos 업데이트
    start_pos += 7  →  start_pos = 7
    (참조로 전달되므로 호출자의 pos_도 7로 업데이트)

[4] Logits → Token 샘플링
    cur_token = logits_to_token(logits_tensor)
    // logits 텐서에서 마지막 토큰(position 6)의 logits 추출
    // shape [32000]의 logits에서 다음 토큰 샘플링
    // → cur_token = 29871 (예시)

반환: cur_token = 29871
```

### Sequential Prefill (참고)

`enable_parallel_prefill_=false`인 경우에는 토큰을 한 개씩 순차적으로 처리합니다:

```
pos=0: step(token=450, start_pos=0) → logits → 무시
pos=1: step(token=1234, start_pos=1) → logits → 무시
...
pos=6: step(token=338, start_pos=6) → logits → 샘플링 → cur_token
```

이 방식은 모델이 incremental KV 캐시 업데이트만 지원할 때 사용됩니다. Parallel prefill에 비해 7배 느립니다.

---

## 8. TextDecoderRunner::step() — 모델 Forward Pass

**파일**: `extension/llm/runner/text_decoder_runner.cpp:36`

### 8.1 메서드 메타데이터 조회

```
method_meta = module_->method_meta("forward")
num_inputs = method_meta.num_inputs()  → 2 (token_ids, cache_position)
use_kv_cache = (num_inputs > 1) → true
```

### 8.2 캐시 위치 텐서 생성

**파일**: `extension/llm/runner/util.h:111` — `populate_start_pos_or_cache_position()`

```
method_meta의 두 번째 입력 텐서 메타데이터 조회:
  second_input_info = method_meta.input_tensor_meta(1)
  second_input_sizes = second_input_info.sizes()
  numel = second_input_sizes[0]

[경우 1] numel == 1인 경우 (일반적인 KV 캐시 모델):
  start_pos_tensor = from_blob(&start_pos, shape={1}, dtype=Long)
  // 데이터: [0]  (첫 prefill 시)
  //   또는: [7]  (생성 루프 시)

[경우 2] numel > 1인 경우 (cache_position 배열이 필요한 모델):
  cache_positions = [0, 1, 2, 3, 4, 5, 6]  (prefill 시, seq_len=7)
  start_pos_tensor = from_blob(cache_positions.data(), shape={7}, dtype=Long)
```

### 8.3 IOManager를 통한 입력 준비

```
inputs = io_manager_->prepare_decode(tokens, start_pos_tensor, "forward")

기본 IOManager (CPU):
  1. method_meta 조회 → num_inputs == 2 확인
  2. inputs = [tokens_tensor, start_pos_tensor] 반환

  Prefill 시:
    inputs[0]: shape [1, 7], dtype=Long, data=[450, 1234, 304, 278, 8494, 1139, 338]
    inputs[1]: shape [1], dtype=Long, data=[0]

  Decode 시 (생성 루프):
    inputs[0]: shape [1, 1], dtype=Long, data=[29871]  (현재 토큰)
    inputs[1]: shape [1], dtype=Long, data=[7]          (현재 위치)
```

### 8.4 모델 실행

```
outputs_res = module_->execute("forward", inputs)

내부 흐름:
  module_->load_method("forward")  ← 이미 로드됨, 스킵
  method->set_inputs(inputs)       ← 입력 텐서 설정
  method->execute()                ← ExecuTorch 런타임 실행
                                      → XNNPACK delegate가 attention, FFN 등 연산 수행
                                      → KV 캐시 업데이트
  method->get_outputs()            ← 출력 텐서 수집

outputs_res.get() = [logits_tensor]
  Prefill 시: logits_tensor.shape = [1, 7, 32000]
  Decode 시:  logits_tensor.shape = [1, 1, 32000]
```

### 8.5 출력 후처리

```
io_manager_->update_decode(outputs_res.get(), "forward")
// 기본 IOManager: no-op (KV 캐시는 모델 내부에서 관리)

// 검증
outputs_res.get().size() == 1  ✓ (출력이 1개)
outputs_res.get()[0].isTensor() ✓ (텐서 타입)

return outputs_res.get()[0].toTensor()  // logits 텐서 반환
```

---

## 9. IOManager — 입출력 준비

**파일**: `extension/llm/runner/io_manager/io_manager.h`

기본 `IOManager`는 CPU 추론을 위한 최소한의 구현입니다:

```
prepare_decode(input, start_pos, method_name):
  1. method_meta 조회
  2. num_inputs == 2 확인 (토큰 + 캐시 위치)
  3. return {input, start_pos}  ← 입력을 그대로 전달

update_decode(outputs, method_name):
  → no-op (CPU에서는 별도 후처리 불필요)
```

**커스텀 IOManager**는 GPU 백엔드, 양자화 모델, 또는 특수한 캐시 관리가 필요한 경우 파생 클래스로 구현할 수 있습니다.

---

## 10. Module::execute — ExecuTorch 런타임 실행

**파일**: `extension/module/module.h`

```
Module::execute("forward", inputs)
│
├── load_method("forward") ← 이미 로드된 경우 스킵
│
├── method = methods_["forward"].method.get()
│
├── method->set_inputs(inputs)
│   // 입력 텐서를 Method의 내부 버퍼에 복사/참조 설정
│   // Prefill 예시:
│   //   input[0] = tokens [1, 7] → [450, 1234, 304, 278, 8494, 1139, 338]
│   //   input[1] = start_pos [1] → [0]
│
├── method->execute()
│   // ExecuTorch 런타임이 실행 계획(execution plan)에 따라 연산 수행
│   // XNNPACK 백엔드로 위임된 연산들:
│   //   - Token Embedding: [1,7] → [1,7,4096]
│   //   - RoPE (위치 인코딩)
│   //   - Multi-Head Attention (32 heads)
│   //     - Q, K, V 프로젝션
│   //     - Scaled Dot-Product Attention
│   //     - KV 캐시 업데이트 (position 0~6에 key/value 저장)
│   //   - Feed-Forward Network (SwiGLU)
│   //   - RMSNorm
│   //   - ... (32 레이어 반복)
│   //   - Output projection → logits [1, 7, 32000]
│   //
│   // ※ 이 내부는 XNNPACK delegate가 처리하므로
│   //   본 분석의 범위 바깥입니다.
│
└── return method->get_outputs()
    // outputs = [logits_tensor]
    // logits_tensor.shape = [1, 7, 32000]
    // logits_tensor.dtype = Float (또는 Half/BFloat16)
```

---

## 11. Logits → Token 변환 (Sampler)

**파일**: `extension/llm/sampler/util.h:25` + `extension/llm/sampler/sampler.cpp`

### 11.1 logits_to_token()

```
logits_to_token(logits_tensor, temperature=0.0)  ← prefill에서 호출 (temp=0)
logits_to_token(logits_tensor, temperature=0.8)  ← 생성 루프에서 호출

[1] 텐서 타입 분기
    scalar_type = Float → float* logits

[2] 3D 텐서인 경우 마지막 토큰의 logits 추출
    logits_tensor.shape = [1, 7, 32000]
    num_tokens = 7
    logits += (7-1) * 32000  → position 6의 logits 포인터로 이동
    // 이제 logits는 shape [32000]의 1D 배열을 가리킴

[3] Sampler 생성 & 샘플링
    Sampler sampler(vocab_size=32000, temperature=0.8);
    result = sampler.sample(logits);
```

### 11.2 Sampler::sample()

```
sample(logits):
  inv_temperature_ = 1.0 / 0.8 = 1.25

  [경우 1] temperature == 0 (greedy):
    return sample_argmax(logits)
    // 32000개 logits 중 최댓값의 인덱스 반환

  [경우 2] temperature > 0:
    // (a) 온도 스케일링
    for (q = 0; q < 32000; q++):
      logits[q] *= 1.25   // 높은 inv_temp → logits 차이 증폭 → 더 확정적

    // (b) Softmax
    softmax(logits, 32000):
      max_val = max(logits)
      for i: logits[i] = exp(logits[i] - max_val)
      sum = sum(logits)
      for i: logits[i] /= sum
      // 이제 logits는 확률 분포 (합=1)

      // 예시 (상위 5개):
      // logits[29871] = 0.15   ← " 42"
      // logits[29906] = 0.12   ← " 4"
      // logits[29946] = 0.08   ← " 8"
      // logits[29900] = 0.07   ← " 0"
      // logits[29941] = 0.05   ← " 5"
      // ... (나머지 합산 = 0.53)

    // (c) 랜덤 코인 생성
    coin = random_f32(&rng_state_)  // [0, 1) 범위의 난수
    // 예: coin = 0.23

    // (d) Top-p 샘플링 (topp_ = 0.9)
    return sample_topp(logits, coin=0.23)
```

### 11.3 Top-p (Nucleus) 샘플링

```
sample_topp(probabilities, coin=0.23):
  topp_ = 0.9
  cutoff = (1.0 - 0.9) / (32000 - 1) = 0.0000031

  [1] cutoff 이상인 토큰만 후보로 선별
      probindex = [(0.15, 29871), (0.12, 29906), (0.08, 29946), ...]
      // cutoff가 매우 작으므로 대부분의 토큰이 포함

  [2] 확률 내림차순 정렬
      [(0.15, 29871), (0.12, 29906), (0.08, 29946), (0.07, 29900), ...]

  [3] 누적 확률이 0.9를 초과하는 지점 찾기
      cumulative = 0.15 → 0.27 → 0.35 → ... → 0.91 (초과!)
      last_idx = 해당 인덱스

  [4] 잘린 분포에서 샘플링
      r = coin * cumulative_prob = 0.23 * 0.91 = 0.2093
      CDF 순회하며 r 위치의 토큰 선택
      → 29871 반환 (확률 0.15/0.91에 해당)
```

---

## 12. 첫 번째 토큰 출력

**파일**: `extension/llm/runner/text_llm_runner.cpp:193`

```
// prefill에서 반환된 토큰을 텍스트로 디코드
cur_token = 29871  (prefill 결과)

decode_result = tokenizer_->decode(prev_token=29871, cur_token=29871)
// → " 42" (예시 — 실제로는 토크나이저에 따라 다름)

wrapped_callback(" 42"):
  safe_printf(" 42")   → stdout에 " 42" 출력
  fflush(stdout)        → 즉시 버퍼 플러시 (스트리밍 출력)
  token_callback(" 42") → (사용자 콜백이 있으면 호출)
```

---

## 13. 토큰 생성 루프 (TextTokenGenerator::generate)

**파일**: `extension/llm/runner/text_token_generator.h:55`

### 13.1 초기 설정

```
tokens = [450, 1234, 304, 278, 8494, 1139, 338, 29871]
start_pos = 7
max_new_tokens = 120
temperature = 0.8

use_kv_cache_ = true이므로:
  token_data = {29871}     // 마지막 토큰만
  token_shape = {1, 1}     // batch=1, seq=1

tokens_managed = from_blob(token_data.data(), {1,1}, Long)
// 텐서: [[29871]]

should_stop_ = false
pos = 7  (=start_pos)
```

### 13.2 생성 루프 (반복)

```
while (pos < 7 + 120):  // pos < 127

  ─── 반복 1 (pos=7) ───
  │
  │ [1] 모델 forward pass
  │   logits_res = text_decoder_runner_->step(tokens_managed, pos=7)
  │   // tokens_managed = [[29871]], start_pos = 7
  │   // → module->execute("forward", [{[29871]}, {7}])
  │   // → XNNPACK: position 7에서 토큰 29871 처리
  │   //   KV 캐시의 position 7에 key/value 저장
  │   // → logits: [1, 1, 32000]
  │
  │ [2] 샘플링
  │   prev_token = 29871
  │   stats_->on_sampling_begin()  // 샘플링 시간 측정 시작
  │   cur_token = logits_to_token(logits, temperature=0.8)
  │   // → 예: cur_token = 13  ("\n")
  │   stats_->on_sampling_end()    // 샘플링 시간 누적
  │
  │ [3] pos 증가
  │   pos = 8
  │
  │ [4] 토큰 데이터 업데이트 (KV 캐시 사용 시)
  │   token_data[0] = 13  // 다음 반복에서 사용할 입력 토큰
  │   // tokens_managed는 token_data를 가리키므로 자동으로 업데이트됨
  │
  │ [5] 토큰 디코드 & 콜백
  │   decode_result = tokenizer_->decode(prev=29871, cur=13) → "\n"
  │   token_callback("\n") → stdout에 "\n" 출력
  │
  │ [6] 종료 조건 확인
  │   should_stop_ = false? → 계속
  │   ignore_eos_ = false, eos_ids_ = {2}
  │   cur_token(13) ∈ {2}? → NO → 계속
  │
  ─── 반복 2 (pos=8) ───
  │
  │ tokens_managed = [[13]]  (token_data[0]이 13으로 업데이트됨)
  │ step(tokens_managed, pos=8)
  │ → logits → sample → cur_token = 1576 ("The")
  │ pos = 9
  │ token_data[0] = 1576
  │ decode("The") → stdout 출력
  │
  ─── 반복 3 (pos=9) ───
  │ ...
  │
  ─── ... (계속) ───
  │
  ─── 반복 N: EOS 토큰 생성 시 ───
  │
  │ cur_token = 2 (EOS)
  │ eos_ids_.find(2) != eos_ids_.end() → true!
  │ "Reached to the end of generation" 출력
  │ break!
  │
  ─── 또는: pos >= 127 도달 시 ───
  │ while 조건 불충족 → 루프 종료

return pos - start_pos
// 예: 30개 토큰 생성 후 EOS → return 30
// 또는: 120개 모두 생성 → return 120
```

### 13.3 KV 캐시 없이 동작하는 경우 (참고)

```
use_kv_cache_ = false인 경우:
  token_data = [450, 1234, 304, 278, 8494, 1139, 338, 29871]
  token_shape = {1, 8}

  매 반복마다:
    token_data.push_back(new_token)
    resize_tensor_ptr(tokens_managed, {1, token_data.size()})
    // 텐서 크기가 매 반복마다 1씩 증가
    // 전체 시퀀스를 다시 모델에 입력
    // → O(n²) 복잡도로 매우 비효율적
```

---

## 14. 통계 리포트 출력

**파일**: `extension/llm/runner/stats.h:136`

```
generate() 종료 후:

stats_->inference_end_ms = 15000
stats_->num_prompt_tokens = 7
stats_->num_generated_tokens = 120

print_report(stats_):

  JSON 출력:
  PyTorchObserver {"prompt_tokens":7,"generated_tokens":120,
    "model_load_start_ms":1000,"model_load_end_ms":2500,
    "inference_start_ms":2500,"inference_end_ms":15000,
    "prompt_eval_end_ms":2700,"first_token_ms":2700,
    "aggregate_sampling_time_ms":50,
    "SCALING_FACTOR_UNITS_PER_SECOND":1000}

  상세 로그:
    Prompt Tokens: 7    Generated Tokens: 120
    Model Load Time:            1.500 (seconds)
    Total inference time:       12.500 (seconds)  Rate: 9.6 (tokens/second)
      Prompt evaluation:        0.200 (seconds)   Rate: 35.0 (tokens/second)
      Generated 120 tokens:     12.300 (seconds)  Rate: 9.76 (tokens/second)
    Time to first generated token: 0.200 (seconds)
    Sampling time over 127 tokens: 0.050 (seconds)
```

### 성능 지표 설명

| 지표 | 계산 | 의미 |
|------|------|------|
| Model Load Time | `model_load_end - model_load_start` | .pte 로드 + XNNPACK 초기화 시간 |
| Total Inference | `inference_end - inference_start` | 토큰화부터 마지막 토큰까지 |
| Prompt Evaluation | `prompt_eval_end - inference_start` | Prefill 단계 소요 시간 |
| Prompt eval rate | `num_prompt / prompt_eval_time` | Prefill 처리량 (tok/s) |
| Generation rate | `num_generated / gen_time` | Decode 처리량 (tok/s) |
| Time to first token | `first_token - inference_start` | 첫 토큰까지의 지연 (TTFT) |
| Sampling time | 누적 `on_sampling_begin~end` | 순수 샘플링 오버헤드 |

---

## 15. 객체 소유권 및 생명주기

```
TextLLMRunner (소유자)
│
├── tokenizer_          unique_ptr<Tokenizer>       ← 첫 번째 파괴
├── metadata_           unordered_map               ← 값 타입
├── module_             unique_ptr<Module>          ← 가장 오래 생존 필요
│                                                     (TextDecoderRunner보다 오래)
├── text_decoder_runner_ unique_ptr<TextDecoderRunner>
│   ├── module_          Module* (비소유)           ← module_을 참조
│   └── io_manager_      IOManager* (비소유)        ← io_manager_를 참조
│
├── text_prefiller_      unique_ptr<TextPrefiller>
│   └── text_decoder_runner_ TextDecoderRunner* (비소유)
│
├── io_manager_          unique_ptr<IOManager>
│   └── module_          Module& (비소유 참조)
│
├── text_token_generator_ unique_ptr<TextTokenGenerator>
│   ├── tokenizer_        Tokenizer* (비소유)
│   ├── text_decoder_runner_ TextDecoderRunner* (비소유)
│   └── stats_            Stats* (비소유)
│
└── stats_               unique_ptr<Stats>

파괴 순서 (C++ 멤버 역순):
  stats_ → text_token_generator_ → io_manager_ →
  text_prefiller_ → text_decoder_runner_ → module_ →
  metadata_ → tokenizer_

핵심: module_이 text_decoder_runner_보다 늦게 파괴됨 ✓
      text_decoder_runner_가 text_prefiller_/text_token_generator_보다 늦게 파괴됨 ✓
```

---

## 16. 전체 호출 스택 요약 (Call Graph)

```
main()
│
├── gflags::ParseCommandLineFlags()
├── example::create_llama_runner()
│   ├── load_llama_tokenizer()
│   │   └── llm::load_tokenizer() → SentencePiece/TikToken/HF 시도
│   └── llm::create_text_llm_runner()
│       ├── Module("llama2.pte", MmapUseMlockIgnoreErrors)
│       ├── get_llm_metadata() → metadata map
│       ├── get_eos_ids() → {2}
│       ├── IOManager(*module)
│       ├── TextDecoderRunner(module*, io_manager*, "forward")
│       ├── TextPrefiller(decoder_runner*, use_kv=true, parallel=true, 2048)
│       ├── Stats()
│       ├── TextTokenGenerator(tokenizer*, decoder_runner*, use_kv=true, {2}, stats*)
│       └── TextLLMRunner(metadata, tokenizer, module, decoder, prefiller, io, generator, stats)
│
├── runner->generate(prompt, config)
│   ├── load()
│   │   ├── text_prefiller_->load()
│   │   │   └── text_decoder_runner_->load()
│   │   │       └── module_->load_method("forward")
│   │   │           ├── module_->load() → Program 로드, flatbuffer 파싱
│   │   │           └── Method 로드 → XNNPACK delegate 초기화
│   │   ├── io_manager_->load() → no-op
│   │   └── text_token_generator_->load() → (이미 로드됨)
│   │
│   ├── tokenizer_->encode(prompt, bos=0, eos=0)
│   │   → [450, 1234, 304, 278, 8494, 1139, 338]
│   │
│   ├── text_prefiller_->prefill(tokens, pos_=0)
│   │   └── prefill_chunk(tokens, pos_)
│   │       ├── from_blob(tokens, {1,7}, Long) → 토큰 텐서
│   │       ├── text_decoder_runner_->step(tokens_tensor, start_pos=0)
│   │       │   ├── module_->method_meta("forward") → num_inputs=2
│   │       │   ├── populate_start_pos_or_cache_position() → [0] 텐서
│   │       │   ├── io_manager_->prepare_decode(tokens, pos, "forward")
│   │       │   │   → [{tokens}, {pos}]
│   │       │   ├── module_->execute("forward", inputs)
│   │       │   │   └── Method::execute() → XNNPACK 실행
│   │       │   │       → logits [1, 7, 32000]
│   │       │   └── io_manager_->update_decode() → no-op
│   │       ├── start_pos += 7 → pos_=7
│   │       └── logits_to_token(logits) → Sampler::sample() → 29871
│   │
│   ├── config.resolve_max_new_tokens(2048, 7) → 121
│   │
│   ├── tokenizer_->decode(29871, 29871) → " 42"
│   ├── wrapped_callback(" 42") → stdout 출력
│   │
│   ├── text_token_generator_->generate(tokens, pos_=7, max=120, temp=0.8)
│   │   └── while (pos < 127):
│   │       ├── text_decoder_runner_->step(token, pos)
│   │       │   ├── populate_start_pos_or_cache_position()
│   │       │   ├── io_manager_->prepare_decode()
│   │       │   ├── module_->execute("forward")
│   │       │   │   └── XNNPACK delegate 실행
│   │       │   └── io_manager_->update_decode()
│   │       ├── logits_to_token(logits, temp=0.8)
│   │       │   └── Sampler(32000, 0.8).sample(logits)
│   │       │       ├── logits *= inv_temperature
│   │       │       ├── softmax(logits, 32000)
│   │       │       └── sample_topp(probs, random_coin)
│   │       ├── tokenizer_->decode(prev, cur) → 텍스트
│   │       ├── token_callback(텍스트) → stdout 출력
│   │       ├── EOS 체크: cur_token ∈ {2}? → break
│   │       └── pos++
│   │
│   ├── pos_ += num_generated_tokens
│   └── print_report(stats_) → JSON + 성능 로그
│
└── return 0
```

---

## 부록: GenerationConfig::resolve_max_new_tokens 동작 표

| seq_len | max_new_tokens | max_context_len | num_prompt | 결과 |
|---------|---------------|-----------------|------------|------|
| -1 | -1 | 2048 | 7 | 2041 |
| 128 | -1 | 2048 | 7 | min(128,2048)-7 = 121 |
| -1 | 50 | 2048 | 7 | min(50, 2041) = 50 |
| 128 | 50 | 2048 | 7 | min(min(128,2048)-7, 50) = min(121,50) = 50 |
| -1 | -1 | 2048 | 2000 | 48 |
| 128 | -1 | 2048 | 200 | min(128,2048)-200 = -72 → max(0,-72) = 0 ⚠ |

---

## 부록: 텐서 Shape 변화 추적 (Prefill + Decode 1회)

```
[Prefill]
  Input tokens:     [1, 7]  (Long)   → [450, 1234, 304, 278, 8494, 1139, 338]
  Cache position:   [1]     (Long)   → [0]
  ─── module_->execute("forward") ───
  Output logits:    [1, 7, 32000] (Float)
  → 마지막 위치 logits: [32000] → Sampler → token_id = 29871

[Decode step 1]
  Input tokens:     [1, 1]  (Long)   → [29871]
  Cache position:   [1]     (Long)   → [7]
  ─── module_->execute("forward") ───
  Output logits:    [1, 1, 32000] (Float)
  → logits: [32000] → Sampler(temp=0.8) → token_id = 13

[Decode step 2]
  Input tokens:     [1, 1]  (Long)   → [13]
  Cache position:   [1]     (Long)   → [8]
  ─── module_->execute("forward") ───
  Output logits:    [1, 1, 32000] (Float)
  → logits: [32000] → Sampler(temp=0.8) → token_id = 1576

[... 반복 ...]
```
